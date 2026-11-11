"""Fresh measured 12-view survey before every model decision; no scene memory."""
import math
from dataclasses import replace

import cv2
import numpy as np

from .capture import wrap


METHOD = "actual_12_rgbd_views_per_decision"


class Survey30:
    def __init__(self, settle_s=0.25, max_duration_s=90):
        self.settle_s, self.max_duration_s = settle_s, max_duration_s
        self.anchor = None
        self.camera_anchor = None
        self.started = self.last_stamp = None
        self.stable_frame = None
        self.index = 0
        self.frames = {}
        self.last_frame = None

    @property
    def ready(self):
        return self.index == 13

    def status(self):
        return {"status": "scan_ready" if self.ready else "observe_rotate",
                "target_yaw_rad": None if self.anchor is None else wrap(self.anchor[2] + self.index * math.pi / 6),
                "confirmed_views": len(self.frames), "required_views": 12,
                "return_to_start_required": True, "motion_dispatch": False}

    def observe(self, frame):
        frame.validate()
        if self.ready:
            raise ValueError("completed scan must be consumed before a new scan")
        if self.last_stamp is not None and frame.stamp <= self.last_stamp:
            raise ValueError("scan timestamps must strictly increase")
        if self.anchor is None:
            self.anchor = tuple(frame.base_pose)
            self.camera_anchor = frame.world_camera[:3, 3].copy()
            self.started = frame.stamp
        if frame.stamp - self.started > self.max_duration_s:
            raise ValueError("full scan timeout")
        if np.linalg.norm(np.array(frame.base_pose[:2]) - self.anchor[:2]) > 0.08:
            raise ValueError("base translated more than 8 cm during scan")
        if np.linalg.norm(frame.world_camera[:3, 3] - self.camera_anchor) > 0.08:
            raise ValueError("camera translated more than 8 cm during scan")
        rotation = frame.world_camera[:3, :3]
        camera_yaw = math.atan2(rotation[1, 2], rotation[0, 2])
        if abs(wrap(camera_yaw - frame.base_pose[2])) > math.radians(2):
            raise ValueError("camera must be calibrated to face the base forward direction")
        if np.dot(rotation[:, 1], [0, 0, -1]) < math.cos(math.radians(3)):
            raise ValueError("camera must remain level during the measured scan")
        h, w = frame.depth_m.shape
        fx, fy, cx, cy = frame.intrinsics
        hfov = math.degrees(math.atan(cx / fx) + math.atan((w - cx) / fx))
        vfov = math.degrees(math.atan(cy / fy) + math.atan((h - cy) / fy))
        if not (88 <= hfov <= 92 and 88 <= vfov <= 92):
            raise ValueError("direct R2R capture requires calibrated near-90-degree horizontal and vertical FOV")
        valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0.1) & (frame.depth_m <= 10)
        if valid.mean() < 0.8:
            raise ValueError("too much invalid measured depth for full scan")
        target = self.anchor[2] + self.index * math.pi / 6
        at_target = abs(wrap(frame.base_pose[2] - target)) <= math.radians(2)
        if at_target:
            prior = self.stable_frame
            if prior is None or abs(wrap(frame.base_pose[2] - prior.base_pose[2])) > math.radians(0.5) or np.linalg.norm(frame.world_camera[:3, 3] - prior.world_camera[:3, 3]) > 0.01:
                self.stable_frame = frame
            elif frame.stamp - prior.stamp >= self.settle_s:
                if self.index < 12:
                    self.frames[self.index] = replace(frame, rgb=frame.rgb.copy(), depth_m=frame.depth_m.copy(), world_camera=frame.world_camera.copy())
                self.index += 1
                self.stable_frame = None
        else:
            self.stable_frame = None
        self.last_stamp, self.last_frame = frame.stamp, frame
        return self.status()

    def observations(self, now):
        if not self.ready or set(self.frames) != set(range(12)):
            raise ValueError("all 12 measured views and a measured return heading are required")
        if not 0 <= now - self.last_stamp <= 0.3 or now - self.started > self.max_duration_s:
            raise ValueError("full scan expired or clock moved backwards")
        observations = {}
        for slot, frame in sorted(self.frames.items()):
            suffix = "" if slot == 0 else f"_{slot * 30}"
            valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0.1) & (frame.depth_m <= 10)
            depth = np.where(valid, frame.depth_m / 10, 0).astype(np.float32)
            observations["rgb" + suffix] = cv2.resize(frame.rgb, (224, 224), interpolation=cv2.INTER_LINEAR)[None]
            observations["depth" + suffix] = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_NEAREST)[None, ..., None]
        return observations


class FullScanNavigator:
    def __init__(self, policy=None, instruction="", planner=None):
        if planner is None:
            from .planner import ReplayPlanner
            planner = ReplayPlanner(policy, instruction)
        self.planner = planner
        self.survey = None
        self.pending = []
        self.last_frame = None
        self.completed_scans = 0
        self.halted = None

    def observe(self, frame):
        if self.halted:
            return {"status": "halt", "reason": self.halted, "motion_dispatch": False}
        try:
            frame.validate()
            previous = self.last_frame
            if previous is not None:
                dt = frame.stamp - previous.stamp
                if dt <= 0:
                    raise ValueError("timestamps must strictly increase")
                if np.linalg.norm(np.array(frame.base_pose[:2]) - previous.base_pose[:2]) > 2 * dt + 0.15:
                    raise ValueError("localization jump; restart the episode")
            self.last_frame = frame
            if self.pending:
                if np.linalg.norm(np.array(frame.base_pose[:2]) - self.pending[0]) < 0.18:
                    self.pending.pop(0)
                if self.pending:
                    return {"status": "await_arrival", "path_xy_m": self.pending, "motion_dispatch": False}
            if self.planner.finished:
                return {"status": "stop", "reason": "episode_finished", "motion_dispatch": False}
            if self.survey is None:
                self.survey = Survey30()
            if self.survey.ready:
                # Do not let a completed scan survive subsequent motion.
                end = self.survey.last_frame
                if np.linalg.norm(np.array(frame.base_pose[:2]) - end.base_pose[:2]) > 0.02 or abs(wrap(frame.base_pose[2] - end.base_pose[2])) > math.radians(0.5) or np.linalg.norm(frame.world_camera[:3, 3] - end.world_camera[:3, 3]) > 0.01:
                    raise ValueError("robot moved after full scan; a new scan is required")
                return self.survey.status()
            return self.survey.observe(frame)
        except ValueError as error:
            self.halted = str(error)
            return {"status": "halt", "reason": self.halted, "motion_dispatch": False}

    def plan(self, now):
        if self.halted:
            return {"status": "halt", "reason": self.halted, "motion_dispatch": False}
        if self.last_frame is None or not math.isfinite(now) or not 0 <= now - self.last_frame.stamp <= 0.3:
            return {"status": "hold", "reason": "fresh_measured_pose_required", "motion_dispatch": False}
        if self.pending:
            return {"status": "await_arrival", "path_xy_m": self.pending, "motion_dispatch": False}
        if self.planner.finished:
            return {"status": "stop", "reason": "episode_finished", "motion_dispatch": False}
        if self.survey is None:
            return {"status": "scan_required", "reason": "fresh_12_view_scan_before_every_decision", "motion_dispatch": False}
        if not self.survey.ready:
            return self.survey.status()
        try:
            observations = self.survey.observations(now)
        except ValueError as error:
            self.halted = str(error)
            return {"status": "halt", "reason": self.halted, "motion_dispatch": False}
        stamps = [f.stamp for f in self.survey.frames.values()]
        result = self.planner.plan(observations, self.last_frame.base_pose)
        self.completed_scans += 1
        self.survey = None  # Consumed: no previous images can feed the next model call.
        self.pending = [list(p) for p in result.get("path_xy_m", [])]
        return {**result, "completed_scans": self.completed_scans, "actual_views": 12,
                "source_stamps": stamps, "oldest_view_age_s": now - min(stamps),
                "observation_method": METHOD, "history_reprojection": False,
                "robot_connection": False, "motion_dispatch": False}
