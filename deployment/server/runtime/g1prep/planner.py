"""Recorded-scan planner using the official GraphMap and graph input builder.

Output is a path proposal. It is deliberately not connected to a robot transport.
"""
import ast
import math
import types

import numpy as np
import quaternion
import torch
from torch.nn.utils.rnn import pad_sequence

from .model import UPSTREAM, HABITAT


def definitions(path, names, context):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, str(path), "exec"), context)


def load_graph():
    context = {"np": np, "quaternion": quaternion}
    definitions(HABITAT / "habitat/utils/geometry_utils.py", {"quaternion_from_coeff", "quaternion_rotate_vector"}, context)
    definitions(HABITAT / "habitat/tasks/utils.py", {"cartesian_to_polar"}, context)
    path = UPSTREAM / "vlnce_baselines/models/graph_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module.startswith("habitat")) and not (isinstance(n, ast.Import) and any(a.name.startswith("matplotlib") for a in n.names))]
    # Habitat uses y up. The upstream function mistakenly uses horizontal dz
    # for elevation, making otherwise identical plans depend on world yaw.
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'calculate_vp_rel_pos_fts')
    elevation = next(n for n in function.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'elevation' for t in n.targets))
    numerator = elevation.value.args[0].left
    if not isinstance(numerator, ast.Name) or numerator.id != 'dz':
        raise ValueError('Unexpected upstream elevation calculation; review coordinates')
    numerator.id = 'dy'
    # A coincident node has no bearing. Encoding -base_heading for the current
    # node otherwise leaks the arbitrary world yaw into graph attention.
    if not isinstance(function.body[-1], ast.Return):
        raise ValueError('Unexpected upstream coordinate return')
    function.body.insert(-1, ast.parse('if xz_dist <= 1e-7:\n    heading = 0.0').body[0])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), context)
    return context["GraphMap"]


def habitat_pose(pose):
    x, y, yaw = pose
    return np.array([-y, 0, -x], dtype=np.float32), np.array([0, math.sin(yaw / 2), 0, math.cos(yaw / 2)])


def robot_point(point):
    return [-float(point[2]), -float(point[0])]


class ReplayPlanner:
    coordinate_revision = 'habitat-y-up-coincident-zero-v1'

    def __init__(self, policy, instruction):
        self.policy = policy
        self.device = policy.device
        self.model_id = policy.model_id
        self.config = types.SimpleNamespace(MODEL=policy.runtime_config)
        self.graph_step_id_mode = policy.graph_step_id_mode
        self.envs = types.SimpleNamespace(num_envs=1)
        self.gmaps = [load_graph()(False, 0.5, True, 0.0)]
        self.text, self.text_masks = policy.language(instruction)
        self.step_id = 0
        self.previous = None
        self.finished = False
        self.pending_execution = None
        self.last_execution = None
        self.blocked_ghosts = set()
        from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
        context = {"np": np, "torch": torch, "pad_sequence": pad_sequence, "pad_tensors_wgrad": pad_tensors_wgrad, "gen_seq_masks": gen_seq_masks, "MAX_DIST": 30}
        # The official builder hardcodes .cuda(). Use selected CPU/CUDA device only.
        tree = ast.parse(UPSTREAM.joinpath("vlnce_baselines/ss_trainer_ETP_R1.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLTrainer")
        method_ast = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_nav_gmap_variable")
        class DeviceSelection(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda" and not node.args and not node.keywords:
                    node.func.attr = "to"
                    node.args = [ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="device", ctx=ast.Load())]
                return node
        module = ast.Module(body=[DeviceSelection().visit(method_ast)], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(UPSTREAM / "vlnce_baselines/ss_trainer_ETP_R1.py"), "exec"), context)
        self._nav_gmap_variable = types.MethodType(context["_nav_gmap_variable"], self)

    def execution_feedback(self, feedback):
        """Commit only route prefixes that the stopped robot actually completed."""
        if self.pending_execution is None:
            if feedback == self.last_execution:
                return {"accepted": True, "duplicate": True}
            raise ValueError("No matching pending route")
        pending = self.pending_execution
        count = feedback.get('completed_waypoints')
        pose = feedback.get('actual_pose_xy_yaw')
        if (feedback.get('step') != pending['step'] or type(count) is not int
                or not 0 <= count <= pending['path_count']
                or type(feedback.get('route_complete')) is not bool
                or not isinstance(pose, list) or len(pose) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose)):
            raise ValueError("Invalid execution feedback")
        if feedback['route_complete'] and count != pending['path_count']:
            raise ValueError("Complete route requires every waypoint")
        blocked = feedback.get('blocked_by_measured_obstacle', False)
        if type(blocked) is not bool or blocked and feedback['route_complete']:
            raise ValueError("Invalid blocked-route feedback")
        reached = min(count, len(pending['route']))
        self.previous = pending['route'][reached - 1] if reached else pending['current']
        if blocked and pending['ghost'] is not None:
            self.blocked_ghosts.add(pending['ghost'])
        if feedback['route_complete'] and pending['ghost'] is not None:
            graph = self.gmaps[0]
            if self.model_id == 'st2nav':
                graph.delete_ghost(pending['ghost'], preserve_temporal=True)
            else:
                graph.delete_ghost(pending['ghost'])
        self.last_execution = dict(feedback)
        self.pending_execution = None
        return {"accepted": True, "previous_node": self.previous,
                "completed_waypoints": count}

    @torch.inference_mode()
    def plan(self, observations, pose, candidate_filter=None, path_validator=None):
        if self.pending_execution is not None:
            raise ValueError("Previous route requires actual execution feedback before planning")
        if self.finished:
            raise ValueError("episode finished; create a new planner")
        wp, pano, pano_masks = self.policy.encode_scan(observations, candidate_filter=candidate_filter)
        if candidate_filter is not None and not wp["cand_angles"][0]:
            return {"status": "observe_needed", "reason": "no_observed_candidate", "robot_connection": False}
        self.step_id += 1
        if self.step_id > 30:
            self.finished = True
            return {"status": "stop", "reason": "step_limit", "robot_connection": False}
        position, orientation = habitat_pose(pose)
        graph = self.gmaps[0]
        current, candidates, candidate_positions = graph.identify_node(position, orientation, wp["cand_angles"][0], wp["cand_distances"][0])
        average = (pano * pano_masks.unsqueeze(-1)).sum(1) / pano_masks.sum(1, keepdim=True)
        graph_options = {}
        if self.model_id == 'rpgnav':
            slots = wp['cand_img_idxes'][0]
            remaining = np.flatnonzero(~np.isin(np.arange(12), slots))
            ordered = torch.zeros((12, pano.size(-1)), device=pano.device, dtype=pano.dtype)
            ordered[torch.as_tensor(slots, device=pano.device)] = pano[0, :len(slots)]
            ordered[torch.as_tensor(remaining, device=pano.device)] = pano[0, len(slots):len(slots)+len(remaining)]
            graph_options = {'cur_pano_embeds': ordered, 'cur_ori': orientation}
        graph.update_graph(self.previous, self.step_id, current, position, average[0], candidates, candidate_positions, pano[0, :len(candidates)], None, **graph_options)
        inputs = self._nav_gmap_variable([current], [position], [orientation], 1)
        no_candidates = inputs.pop("no_vp_left")[0]
        for index, node in enumerate(inputs['gmap_vp_ids'][0]):
            if node in self.blocked_ghosts:
                inputs['gmap_visited_masks'][0, index] = True
        # Historical ghost nodes must also be rechecked against current memory.
        # A previously clear route is never granted validity by old graph edges alone.
        blocked = []
        if path_validator is not None:
            ids = inputs["gmap_vp_ids"][0]
            for index, node in enumerate(ids):
                if node is None or not node.startswith("g"):
                    continue
                _, front = graph.front_to_ghost_dist(node)
                route = graph.shortest_path[current][front][1:]
                path = [robot_point(graph.node_pos[v]) for v in route] + [robot_point(graph.ghost_aug_pos[node])]
                previous_point = list(pose[:2])
                for point in path:
                    if not path_validator(previous_point, point):
                        inputs["gmap_visited_masks"][0, index] = True
                        blocked.append(node)
                        break
                    previous_point = point
            active = inputs["gmap_masks"] & ~inputs["gmap_visited_masks"]
            if int(active.sum()) <= 1:
                # No path has been committed. Preserve the last measured graph node
                # as the predecessor of the next observation at this location.
                self.previous = current
                return {"status": "observe_needed", "reason": "all_routes_unobserved_or_blocked", "robot_connection": False}
        scores = self.policy(mode="navigation", txt_embeds=self.text, txt_masks=self.text_masks, **inputs)["global_logits"]
        valid = inputs["gmap_masks"] & ~inputs["gmap_visited_masks"]
        if not torch.isfinite(scores[valid]).all():
            raise RuntimeError("non-finite legal action scores")
        probabilities = scores.softmax(-1)[0]
        graph.node_stop_scores[current] = float(probabilities[0])
        selected = int(scores.argmax(-1)[0])
        ids = inputs["gmap_vp_ids"][0]
        audit = {"selected_action_index": selected, "selected_node_id": ids[selected],
                 "graph_coordinate_revision": self.coordinate_revision,
                 "measured_blocked_node_ids": sorted(self.blocked_ghosts),
                 "action_node_ids": ids,
                 "waypoints": [{"relative_left_deg": math.degrees(float(angle)) % 360,
                                "distance_m": float(distance), "node_id": node,
                                "world_xy_m": robot_point(point)}
                               for angle, distance, node, point in zip(wp["cand_angles"][0], wp["cand_distances"][0], candidates, candidate_positions)]}
        if selected == 0 or no_candidates or self.step_id == 30:
            best = max(graph.node_stop_scores, key=graph.node_stop_scores.get)
            route = graph.shortest_path[current][best][1:]
            path = [robot_point(graph.node_pos[v]) for v in route]
            if path_validator is not None:
                previous_point = list(pose[:2])
                for point in path:
                    if not path_validator(previous_point, point):
                        self.previous = current
                        return {"status": "observe_needed", "reason": "return_route_blocked", "robot_connection": False}
                    previous_point = point
            status = "stop" if not path else "return_then_stop"
            self.finished = True
        else:
            ghost = ids[selected]
            _, front = graph.front_to_ghost_dist(ghost)
            route = graph.shortest_path[current][front][1:]
            path = [robot_point(graph.node_pos[v]) for v in route] + [robot_point(graph.ghost_aug_pos[ghost])]
            status = "path_proposal"
        if path:
            self.pending_execution = {'step': self.step_id, 'current': current,
                                      'route': list(route), 'path_count': len(path),
                                      'ghost': ghost if status == 'path_proposal' else None}
            self.previous = current
        return {"status": status, "path_xy_m": path, "step": self.step_id, "candidate_count": len(candidates), "graph_nodes": len(ids), "action_probabilities": probabilities.cpu().tolist(), **audit, "robot_connection": False}
