"""Streaming recording for single-camera history; never pickle incoming arrays."""
import json
from pathlib import Path
import numpy as np

from .memory import RGBDFrame


class SequenceWriter:
    def __init__(self, directory, synthetic=False, floor_z=None, kind="single_rgbd_history"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError("choose a new empty recording directory")
        self.count = 0
        self.last_stamp = None
        self.synthetic = synthetic
        self.metadata = {"format": 2, "kind": kind, "synthetic": synthetic, "complete": False, "frames": 0, "floor_z": floor_z}
        self._metadata()
        self.stream = (self.directory / "frames.jsonl").open("w", encoding="utf-8")

    def _metadata(self):
        self.metadata["frames"] = self.count
        (self.directory / "sequence.json").write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def append(self, frame, request_plan=False):
        frame.validate()
        if self.last_stamp is not None and frame.stamp <= self.last_stamp:
            raise ValueError("recording timestamps must increase")
        filename = f"frame_{self.count:06d}.npz"
        np.savez_compressed(self.directory / filename, rgb=frame.rgb, depth_m=frame.depth_m)
        entry = {"file": filename, "intrinsics": list(frame.intrinsics), "world_camera": np.asarray(frame.world_camera).tolist(),
                 "base_pose": list(frame.base_pose), "stamp": frame.stamp, "depth_stamp": frame.depth_stamp,
                 "pose_stamp": frame.pose_stamp, "request_plan": bool(request_plan)}
        self.stream.write(json.dumps(entry) + "\n")
        self.stream.flush()
        self.count += 1
        self.last_stamp = frame.stamp

    def close(self):
        self.stream.close()
        self.metadata["complete"] = True
        self._metadata()


def read_sequence(directory):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "sequence.json").read_text(encoding="utf-8"))
    if metadata.get("format") != 2 or not metadata.get("complete"):
        raise ValueError("recording incomplete or unsupported")
    with (directory / "frames.jsonl").open(encoding="utf-8") as source:
        count, last_stamp = 0, None
        for line in source:
            entry = json.loads(line)
            path = (directory / entry["file"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("recording path escapes directory")
            with np.load(path, allow_pickle=False) as data:
                frame = RGBDFrame(data["rgb"], data["depth_m"], tuple(entry["intrinsics"]), np.asarray(entry["world_camera"]),
                                  tuple(entry["base_pose"]), entry["stamp"], entry["depth_stamp"], entry["pose_stamp"])
            frame.validate()
            if last_stamp is not None and frame.stamp <= last_stamp:
                raise ValueError("recording clock moved backwards")
            count, last_stamp = count + 1, frame.stamp
            yield frame, bool(entry.get("request_plan", False))
        if count != metadata["frames"]:
            raise ValueError("recording frame count mismatch")
