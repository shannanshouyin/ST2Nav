"""Measured RGBD scene memory, with no learned replacement for official ETP-R1.

World coordinates are x/y horizontal and z up. Camera coordinates are optical:
x right, y down, z forward. Unknown pixels always have depth zero and a mask.
"""
from dataclasses import dataclass
import math

import cv2
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

    def validate(self, *, max_depth=10., min_valid_depth_fraction=.1):
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("RGB must be uint8 HWC RGB")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("RGB and metric depth must be aligned")
        fx, fy, cx, cy = self.intrinsics
        values = (*self.intrinsics, *self.base_pose, self.stamp, self.depth_stamp, self.pose_stamp)
        if not np.isfinite(values).all() or fx <= 0 or fy <= 0:
            raise ValueError("invalid intrinsics, pose, or timestamp")
        h, w = self.depth_m.shape
        if not 0 <= cx < w or not 0 <= cy < h:
            raise ValueError("principal point outside image")
        if max(self.stamp, self.depth_stamp, self.pose_stamp) - min(self.stamp, self.depth_stamp, self.pose_stamp) > 0.1:
            raise ValueError("RGB/depth/pose are not synchronized")
        transform = np.asarray(self.world_camera)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("invalid optical-camera transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError("camera rotation must be orthonormal and right handed")
        if ((self.depth_m > 0.1) & (self.depth_m <= max_depth) & np.isfinite(self.depth_m)).mean() < min_valid_depth_fraction:
            raise ValueError("insufficient valid metric depth")


class SceneMemory:
    def __init__(self, voxel_m=0.02, max_points=300000, max_age_s=120, max_frames=32, stride=2):
        self.voxel_m, self.max_points, self.max_age_s = voxel_m, max_points, max_age_s
        self.max_frames, self.stride = max_frames, stride
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.times = np.empty(0, dtype=np.float64)
        self.frames = []
        self.last_stamp = None

    @staticmethod
    def project(points, frame):
        camera = np.einsum("ni,ij->nj", points - frame.world_camera[:3, 3], frame.world_camera[:3, :3])
        fx, fy, cx, cy = frame.intrinsics
        z = camera[:, 2]
        safe_z = np.where(z > 0.01, z, 1)
        u = np.rint(camera[:, 0] / safe_z * fx + cx).astype(np.int32)
        v = np.rint(camera[:, 1] / safe_z * fy + cy).astype(np.int32)
        h, w = frame.depth_m.shape
        inside = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u, v, z, inside

    def update(self, frame):
        frame.validate()
        if self.last_stamp is not None and frame.stamp <= self.last_stamp:
            raise ValueError("timestamps must increase; reset memory between episodes or clocks")
        if self.frames:
            previous = self.frames[-1]
            movement = np.linalg.norm(np.array(frame.base_pose[:2]) - previous.base_pose[:2])
            if movement > 2 * (frame.stamp - previous.stamp) + 0.15:
                raise ValueError("localization jump; reset episode before continuing")
        keep = frame.stamp - self.times <= self.max_age_s
        # A new valid ray removes older surfaces now measured to be free space.
        # Occluded surfaces are retained rather than guessed away.
        if len(self.points):
            u, v, z, inside = self.project(self.points, frame)
            ids = np.flatnonzero(inside)
            measured = frame.depth_m[v[ids], u[ids]]
            cleared = np.isfinite(measured) & (measured > z[ids] + 0.12) & (measured <= 10)
            keep[ids[cleared]] = False
        self.points, self.colors, self.times = self.points[keep], self.colors[keep], self.times[keep]
        yy, xx = np.indices(frame.depth_m.shape)
        xx, yy = xx[::self.stride, ::self.stride], yy[::self.stride, ::self.stride]
        depth = frame.depth_m[::self.stride, ::self.stride]
        valid = np.isfinite(depth) & (depth > 0.1) & (depth <= 10)
        fx, fy, cx, cy = frame.intrinsics
        camera = np.stack([(xx - cx) / fx * depth, (yy - cy) / fy * depth, depth], axis=-1)[valid]
        points = np.einsum("ni,ij->nj", camera, frame.world_camera[:3, :3].T) + frame.world_camera[:3, 3]
        colors = frame.rgb[::self.stride, ::self.stride][valid]
        self.points = np.concatenate([self.points, points.astype(np.float32)])
        self.colors = np.concatenate([self.colors, colors])
        self.times = np.concatenate([self.times, np.full(len(points), frame.stamp)])
        quantized = np.floor(self.points / self.voxel_m).astype(np.int32)
        _, reversed_ids = np.unique(quantized[::-1], axis=0, return_index=True)
        ids = np.sort(len(quantized) - 1 - reversed_ids)
        if len(ids) > self.max_points:
            ids = ids[np.argsort(self.times[ids], kind="stable")[-self.max_points:]]
        self.points, self.colors, self.times = self.points[ids], self.colors[ids], self.times[ids]
        self.frames = [f for f in self.frames if frame.stamp - f.stamp <= self.max_age_s][-self.max_frames + 1:] + [frame]
        self.last_stamp = frame.stamp

    def render_view(self, position, yaw, now, size=256):
        if self.last_stamp is None or now < self.last_stamp:
            raise ValueError("missing memory or invalid render clock")
        transform = camera_transform(*position, yaw)
        points = np.einsum("ni,ij->nj", self.points - transform[:3, 3], transform[:3, :3])
        z = points[:, 2]
        valid = (z > 0.1) & (z <= 10) & (now - self.times <= self.max_age_s)
        points, colors, z = points[valid], self.colors[valid], z[valid]
        focal = size / 2.0  # Explicit 90 degrees; source intrinsics are used only for back-projection.
        u = np.rint(points[:, 0] / z * focal + (size - 1) / 2).astype(np.int32)
        v = np.rint(points[:, 1] / z * focal + (size - 1) / 2).astype(np.int32)
        uv, depths, rgb = [], [], []
        # Bounded pixel footprint for sampled measured surfaces, never a wide inpainting fill.
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                us, vs = u + dx, v + dy
                inside = (us >= 0) & (us < size) & (vs >= 0) & (vs < size)
                uv.append(vs[inside] * size + us[inside])
                depths.append(z[inside])
                rgb.append(colors[inside])
        pixels, distances, colors = np.concatenate(uv), np.concatenate(depths), np.concatenate(rgb)
        output_depth = np.zeros(size * size, dtype=np.float32)
        output_rgb = np.zeros((size * size, 3), dtype=np.uint8)
        if len(pixels):
            ordered = np.lexsort((distances, pixels))
            _, first = np.unique(pixels[ordered], return_index=True)
            selected = ordered[first]
            output_depth[pixels[selected]] = distances[selected]
            output_rgb[pixels[selected]] = colors[selected]
        return output_rgb.reshape(size, size, 3), output_depth.reshape(size, size)

    def observations(self, pose, height_m, now):
        output, masks, coverage = {}, {}, []
        for slot in range(12):
            rgb, depth = self.render_view((pose[0], pose[1], height_m), pose[2] + slot * math.pi / 6, now)
            suffix = "" if slot == 0 else f"_{slot * 30}"
            output["rgb" + suffix] = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)[None]
            output["depth" + suffix] = np.clip(depth / 10, 0, 1)[None, ..., None]
            masks[slot] = depth > 0
            coverage.append(float(masks[slot].mean()))
        return output, masks, coverage

    def path_observed(self, start, target, now, floor_z, body_radius=0.25, max_age_s=10, visibility_z=None):
        start, target = np.asarray(start), np.asarray(target)
        distance = float(np.linalg.norm(target - start))
        if not np.isfinite([*start, *target, now, floor_z]).all() or distance > 3.5:
            return False
        if distance < 0.05:
            return True
        steps = max(2, int(math.ceil(distance / 0.1)))
        centers = start + np.linspace(0.05, 1, steps)[:, None] * (target - start)
        # Reject measured obstacles within the swept body footprint.
        points = self.points[(now - self.times <= self.max_age_s) & (self.points[:, 2] > floor_z + 0.12) & (self.points[:, 2] < floor_z + 1.5)]
        if len(points):
            bounds = (points[:, :2] >= np.minimum(start, target) - body_radius).all(1) & (points[:, :2] <= np.maximum(start, target) + body_radius).all(1)
            points = points[bounds, :2]
            segment = target - start
            t = np.clip(np.einsum("ni,i->n", points - start, segment) / max(distance ** 2, 1e-8), 0, 1)
            if len(points) and (np.linalg.norm(points - (start + t[:, None] * segment), axis=1) < body_radius).any():
                return False
        direction = (target - start) / distance
        left = np.array([-direction[1], direction[0]])
        samples = np.concatenate([centers + left * offset for offset in [-body_radius, 0, body_radius]])
        # Visibility is checked in the camera-height plane, not claimed as footstep
        # or floor certification. Full 3-D measured obstacles still reject paths.
        samples = np.column_stack([samples, np.full(len(samples), floor_z + 1.2 if visibility_z is None else visibility_z)])
        # The robot already occupies its current footprint; camera minimum range
        # must not turn that starting footprint into an unobservable obstacle.
        seen_free = np.linalg.norm(samples[:, :2] - start, axis=1) < body_radius
        for frame in reversed(self.frames):
            if not 0 <= now - frame.stamp <= max_age_s:
                continue
            u, v, z, inside = self.project(samples, frame)
            ids = np.flatnonzero(inside)
            measured = frame.depth_m[v[ids], u[ids]]
            good = np.isfinite(measured) & (measured > z[ids] + 0.1) & (measured <= 10)
            seen_free[ids[good]] = True
        return bool(seen_free.all())
