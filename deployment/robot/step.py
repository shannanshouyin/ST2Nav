"""Execute one checked short forward leg from a saved model result, after turning.

The two cameras cannot certify the near floor. Human clearance is explicit and
required. This is a supervised experimental step, not autonomous navigation.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
from storage import results_dir
import signal
import time

import numpy as np
import robot
import scan
import dual_scan


def floor_plane(world, start):
    """Estimate only a broad, near-level floor; otherwise retain nominal height."""
    local = world - np.r_[start, 0.]
    distance = np.linalg.norm(local[:, :2], axis=1)
    candidates = local[(np.abs(local[:, 2]) < .25) & (distance > .4) & (distance < 4.)]
    if len(candidates) < 500:
        return None, {'accepted': False, 'reason': 'insufficient_floor_samples'}
    rng = np.random.default_rng(0)
    points = candidates[rng.choice(len(candidates), min(2048, len(candidates)), replace=False)]
    best, count = None, 0
    for _ in range(96):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        offset = -a @ normal
        if normal[2] < math.cos(math.radians(5)) or abs(offset) > .10:
            continue
        mask = np.abs(points @ normal + offset) < .025
        if int(mask.sum()) > count:
            best, count = mask, int(mask.sum())
    if best is None or count < max(500, .6 * len(points)):
        return None, {'accepted': False, 'reason': 'no_dominant_near_level_floor'}
    for _ in range(3):
        selected = points[best]
        coefficients = np.linalg.lstsq(np.c_[selected[:, :2], np.ones(len(selected))],
                                       selected[:, 2], rcond=None)[0]
        normal = np.r_[-coefficients[:2], 1.]
        norm = np.linalg.norm(normal)
        normal /= norm
        offset = -coefficients[2] / norm
        best = np.abs(points @ normal + offset) < .025
        if int(best.sum()) < max(500, .6 * len(points)):
            return None, {'accepted': False, 'reason': 'floor_refinement_unstable'}
    spread = np.linalg.eigvalsh(np.cov(points[best, :2].T))
    if (normal[2] < math.cos(math.radians(5)) or abs(offset) > .10
            or spread[0] < .12 ** 2):
        return None, {'accepted': False, 'reason': 'floor_extent_or_tilt_rejected'}
    return (normal, offset), {'accepted': True, 'normal_world': normal.tolist(),
                              'offset_at_start_m': float(offset),
                              'sample_inlier_fraction': float(best.mean())}


def corridor_points(depth, intrinsics, transform, start, target,
                    radius=robot.BODY_CLEARANCE_RADIUS_M + robot.MAX_PATH_DEVIATION_M):
    yy, xx = np.indices(depth.shape)
    z = depth[::2, ::2]
    valid = np.isfinite(z) & (z > .1) & (z <= 10)
    fx, fy, cx, cy = intrinsics
    camera = np.stack([(xx[::2, ::2] - cx) / fx * z,
                       (yy[::2, ::2] - cy) / fy * z, z], axis=-1)[valid]
    world = camera @ transform[:3, :3].T + transform[:3, 3]
    start, target = np.asarray(start), np.asarray(target)
    plane, floor_check = floor_plane(world, start)
    height = world[:, 2] if plane is None else (world - np.r_[start, 0.]) @ plane[0] + plane[1]
    world = world[(height > .12) & (height < 1.6)]
    delta = target - start
    length = float(np.linalg.norm(delta))
    if not .15 <= length <= 1.5:
        raise ValueError('Short leg distance must be within 0.15..1.5 m')
    t = np.clip((world[:, :2] - start) @ delta / length ** 2, 0, 1)
    separation = np.linalg.norm(world[:, :2] - (start + t[:, None] * delta), axis=1)
    # Lateral tracking allowance must not also extend the leg's endpoint.
    # Keep the full lateral envelope; the end cap covers body plus the existing
    # 0.30 m forward-overshoot bound enforced by robot.forward_progress.
    local = world[:, :2] - start
    along = local @ delta / length
    lateral = (local[:, 0] * delta[1] - local[:, 1] * delta[0]) / length
    past_end = np.maximum(np.maximum(-along, along - length), 0.)
    end_margin = robot.BODY_CLEARANCE_RADIUS_M + .30
    blocked = world[(lateral / radius) ** 2 + (past_end / end_margin) ** 2 < 1.]
    return {'measured_obstacle_points_in_corridor': len(blocked),
            'checked_radius_m': radius,
            'endpoint_margin_m': end_margin,
            'nearest_measured_surface_to_corridor_m': float(separation.min()) if len(world) else None,
            'floor_check': floor_check,
            'near_floor_automatically_verified': False}


def check_forward_leg(camera, directory, turner, calibration, prior, target, report):
    """Confirm transient depth hits while stopped; require two fresh clear pairs."""
    intrinsics = {c['serial']: c['intrinsics'] for c in calibration['cameras']}
    obstacle_seen = False
    clear_pairs = 0
    observation_history = []
    for attempt in range(3):
        if camera.process is None:
            try:
                camera.open(turner.read)
            except scan.CameraError as error:
                camera.recover(turner.read, error)
        begun = time.monotonic()
        frames = dual_scan.capture_pair(camera, directory, 0, turner, calibration,
                                        prefix='before' if attempt == 0 else 'before_retry%d' % attempt,
                                        compress=False)
        captured = time.monotonic()
        # This board loses locomotion replies while USB RGBD streams remain on.
        # Finish both streams before computing the corridor and starting a walk.
        camera.close()
        closed = time.monotonic()
        if camera.process is not None:
            raise RuntimeError('Camera worker still active; no walking permitted')
        pose = turner.read()
        if math.hypot(pose['position_m'][0] - prior[0], pose['position_m'][1] - prior[1]) > .30:
            raise RuntimeError('Robot position changed substantially since planning')
        start = pose['position_m'][:2]
        dx, dy = target[0] - start[0], target[1] - start[1]
        error = scan.wrap(math.atan2(dy, dx) - pose['rpy_rad'][2])
        distance = math.hypot(dx, dy)
        if abs(error) > math.radians(10):
            raise RuntimeError('Turn toward the model point before attempting this straight leg')
        radius = (robot.BODY_CLEARANCE_RADIUS_M + robot.MAX_PATH_DEVIATION_M
                  + distance * abs(math.sin(error)))
        checks = []
        for frame in frames:
            with np.load(directory / frame['file'], allow_pickle=False) as data:
                check = corridor_points(data['depth_m'], intrinsics[frame['serial']],
                    np.asarray(frame['world_camera']), start, target, radius=radius)
                checks.append({'serial': frame['serial'], **check})
        now = time.monotonic()
        age = now - min(f['capture_monotonic_s'] for f in frames)
        report.update(start_pose=pose, fresh_frames=frames, corridor_checks=checks,
                      target_distance_m=distance, target_heading_error_deg=math.degrees(error),
                      observation_attempts=attempt + 1, camera_closed_before_motion=True,
                      prewalk_timing={'capture_and_save_s': captured - begun,
                                      'camera_close_s': closed - captured,
                                      'corridor_check_s': now - closed,
                                      'oldest_frame_age_s': age})
        turner.read()
        fresh = time.monotonic() - min(f['capture_monotonic_s'] for f in frames) <= 5
        blocked = any(c['measured_obstacle_points_in_corridor'] for c in checks)
        obstacle_seen = obstacle_seen or blocked
        clear_pairs = clear_pairs + 1 if fresh and not blocked else 0
        observation_history.append({'attempt': attempt + 1, 'blocked': blocked,
                                    'fresh': fresh, 'consecutive_clear_pairs': clear_pairs,
                                    'frames': [f['file'] for f in frames],
                                    'corridor_checks': checks})
        report['observation_history'] = observation_history
        if fresh and not blocked and (not obstacle_seen or clear_pairs >= 2):
            report['stationary_obstacle_recheck_passed'] = obstacle_seen
            return pose, distance
        if attempt < 2:
            if obstacle_seen:
                print('行走前发现疑似障碍，保持停止并复拍；需连续两组新画面清空才放行（%d/3）。' %
                      (attempt + 1), flush=True)
            else:
                print('行走前画面处理超时，保持停止并重新拍照检查（%d/3）。' % (attempt + 1), flush=True)
    if obstacle_seen:
        raise RuntimeError('A fresh measured obstacle blocks the proposed short leg')
    raise RuntimeError('No timely pre-walk observation after three stationary captures; no movement sent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_result', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--human-clear-path', action='store_true')
    args = parser.parse_args()
    if args.execute and not args.human_clear_path:
        parser.error('This supervised experiment requires confirmed clear, level ground')
    raw = args.model_result.read_bytes()
    plan = json.loads(raw)
    if plan.get('motion_dispatch') is not False or plan.get('model_output', {}).get('status') != 'path_proposal':
        raise ValueError('A saved model path proposal is required')
    path = plan['model_output']['path_xy_m']
    if len(path) != 1 or not np.isfinite(path).all():
        raise ValueError('Exactly one finite short target is required')
    target = path[0]
    config = json.loads((robot.ROOT / 'config.json').read_text())
    calibration = json.loads(dual_scan.CALIBRATION.read_text())
    if calibration.get('status') != 'STATIC_LOCAL_CHECKS_PASSED':
        raise ValueError('Measured camera installation required')
    guard = open('/tmp/etpr1-walk.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = results_dir() / 'navigation' / str(time.time_ns())
    directory.mkdir(parents=True)
    state = robot.State(config['interface'])
    camera = loco = None
    report = {'model_result_sha256': hashlib.sha256(raw).hexdigest(), 'target_world_xy_m': target,
              'human_clearance_confirmed': args.human_clear_path, 'execute': args.execute,
              'complete': False, 'movement_started': False}
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        state.wait()
        loco = robot.client()
        scan.walking_mode(loco, state)
        state.arm()
        turner = scan.Turner(state, loco, None, .3)
        camera = dual_scan.PairCamera(tick=turner.read)
        prior = plan['observation']['reference_pose_xy_yaw']
        pose = None
        def prepare_walk():
            nonlocal pose
            pose, distance = check_forward_leg(camera, directory, turner, calibration,
                                               prior, target, report)
            state.permit()
            print(json.dumps(report), flush=True)
            if args.execute:
                consumed = Path.home() / '.local/share/etpr1/consumed-plans'
                consumed.mkdir(parents=True, exist_ok=True)
                with (consumed / report['model_result_sha256']).open('x') as output:
                    output.write(str(directory))
                report['movement_started'] = True
            return distance
        if args.execute:
            try:
                robot.walk(state, 10., .2, True, loco=loco, prewalk=prepare_walk)
            finally:
                if pose is not None:
                    time.sleep(1.)
                    report['final_displacement'] = scan.measure_after_stop(state, pose)
            final = report['final_displacement']['final_position_xy_m']
            report['distance_to_model_point_after_stop_m'] = math.hypot(target[0] - final[0], target[1] - final[1])
        else:
            prepare_walk()
        report['complete'] = True
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        state.disarm()
        if camera is not None:
            camera.close()
        robot.close_client(loco)
        state.close()
        guard.close()
        (directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({'step_result': report, 'directory': str(directory)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt, FileExistsError) as error:
        raise SystemExit(str(error) or 'Interrupted; stopped')
