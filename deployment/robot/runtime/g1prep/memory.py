from dataclasses import dataclass
import math
import numpy as np

def camera_transform(x, y, z, yaw):
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, [0, 0, -1], forward], axis=1)
    matrix[:3, 3] = [x, y, z]
    return matrix

@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: tuple
    world_camera: np.ndarray
    base_pose: tuple
    stamp: float
    depth_stamp: float
    pose_stamp: float

    def validate(self):
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError('RGB must be uint8 HWC RGB')
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError('RGB and metric depth must be aligned')
        fx, fy, cx, cy = self.intrinsics
        values = (*self.intrinsics, *self.base_pose, self.stamp, self.depth_stamp, self.pose_stamp)
        if not np.isfinite(values).all() or fx <= 0 or fy <= 0:
            raise ValueError('invalid intrinsics, pose, or timestamp')
        h, w = self.depth_m.shape
        if not 0 <= cx < w or not 0 <= cy < h:
            raise ValueError('principal point outside image')
        if max(self.stamp, self.depth_stamp, self.pose_stamp) - min(self.stamp, self.depth_stamp, self.pose_stamp) > 0.1:
            raise ValueError('RGB/depth/pose are not synchronized')
        transform = np.asarray(self.world_camera)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or (not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError('invalid optical-camera transform')
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.0001) or not np.isclose(np.linalg.det(rotation), 1, atol=0.0001):
            raise ValueError('camera rotation must be orthonormal and right handed')
        if ((self.depth_m > 0.1) & (self.depth_m <= 10) & np.isfinite(self.depth_m)).mean() < 0.1:
            raise ValueError('insufficient valid metric depth')
