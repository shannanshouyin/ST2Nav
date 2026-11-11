"""Single-camera scan validation. All poses use x forward, y left, yaw CCW."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Frame:
    rgb: np.ndarray
    depth_m: np.ndarray
    x: float
    y: float
    yaw: float
    stamp: float
    depth_stamp: float
    pose_stamp: float
    hfov_deg: float


class Scan:
    def __init__(self, anchor=(0.0, 0.0, 0.0), max_duration=60.0):
        self.anchor = anchor
        self.max_duration = max_duration
        self.frames = {}
        self.started = None

    def add(self, frame):
        values = [frame.x, frame.y, frame.yaw, frame.stamp, frame.depth_stamp, frame.pose_stamp, frame.hfov_deg]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite capture metadata")
        if not 88.0 <= frame.hfov_deg <= 92.0:
            raise ValueError("R2R model requires calibrated ~90-degree horizontal field of view")
        if max(frame.stamp, frame.depth_stamp, frame.pose_stamp) - min(frame.stamp, frame.depth_stamp, frame.pose_stamp) > 0.1:
            raise ValueError("RGB/depth/pose timestamps differ by more than 0.1 s")
        if frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3:
            raise ValueError("RGB must be uint8 HWC RGB")
        if frame.depth_m.shape != frame.rgb.shape[:2]:
            raise ValueError("depth must be aligned to RGB")
        valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        if valid.mean() < 0.8:
            raise ValueError("too much invalid depth")
        if math.hypot(frame.x - self.anchor[0], frame.y - self.anchor[1]) > 0.08:
            raise ValueError("camera translated more than 8 cm during scan")
        relative = wrap(frame.yaw - self.anchor[2]) % (2 * math.pi)
        slot = round(relative / (math.pi / 6)) % 12
        error = abs(wrap(relative - slot * math.pi / 6))
        if error > math.radians(2):
            raise ValueError("camera yaw is more than 2 degrees from scan slot")
        if self.started is None:
            self.started = frame.stamp
        if frame.stamp < self.started or frame.stamp - self.started > self.max_duration:
            raise ValueError("scan timestamp out of range")
        self.frames[slot] = frame
        return slot

    def observations(self, current_pose, now):
        if set(self.frames) != set(range(12)):
            raise ValueError("scan incomplete: all 12 actual views are required")
        if not all(math.isfinite(v) for v in (*current_pose, now)):
            raise ValueError("non-finite current pose or time")
        if abs(wrap(current_pose[2] - self.anchor[2])) > math.radians(2):
            raise ValueError("return to anchor heading before planning")
        if math.hypot(current_pose[0] - self.anchor[0], current_pose[1] - self.anchor[1]) > 0.08:
            raise ValueError("scan position changed")
        if now < max(f.stamp for f in self.frames.values()) or now - self.started > self.max_duration:
            raise ValueError("scan expired or clock moved backwards")
        observations = {}
        for slot in range(12):
            frame = self.frames[slot]
            rgb = cv2.resize(frame.rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            # Invalid depth is kept at zero; it is never filled as free space.
            meters = np.where(np.isfinite(frame.depth_m) & (frame.depth_m > 0), frame.depth_m, 0)
            depth = cv2.resize(np.clip(meters / 10.0, 0, 1).astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
            suffix = "" if slot == 0 else f"_{slot * 30}"
            observations["rgb" + suffix] = rgb[None]
            observations["depth" + suffix] = depth[None, ..., None]
        return observations


def fixture():
    """Deterministic artificial data for interface checks, never a navigation benchmark."""
    scan = Scan()
    yy, xx = np.indices((256, 256))
    for i in range(12):
        rgb = np.stack([(xx + i * 17) % 256, (yy + i * 13) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
        depth = np.full((256, 256), 2.0 + i / 10, dtype=np.float32)
        scan.add(Frame(rgb, depth, 0, 0, i * math.pi / 6, 100 + i, 100 + i, 100 + i, 90))
    return scan
