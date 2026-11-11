"""Experimental twelve actual RGBD inputs. Official weights; no motion imports.

Retains all captured RGB, including color without valid range. Missing depth is
zero. The native lens is NOT equivalent to the model's training camera. Measured
geometry is used only to audit paths, not to invent missing model observations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from storage import results_dir
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'runtime'))


def prepare(directory):
    import cv2
    import numpy as np
    from g1prep.memory import RGBDFrame, SceneMemory
    record = json.loads((directory / 'scan.json').read_text())
    views = record.get('views', [])
    if (record.get('kind') != 'raw_measured_12_rgbd_views' or not record.get('complete')
            or len(views) != 12 or 'return_pose' not in record or record.get('synthetic')):
        raise ValueError('A complete measured circle with twelve distinct views is required')
    if record.get('stop_error') or record.get('final_measurement_error'):
        raise ValueError('Capture did not stop and measure successfully')
    if not record.get('final_displacement', {}).get('stationary_by_sensors'):
        raise ValueError('A measured stationary final pose is required')
    elapsed = float(record['elapsed_s'])
    if (not np.isfinite([elapsed, record['max_base_drift_m']]).all()
            or not 0 < elapsed <= 240 or not 0 <= float(record['max_base_drift_m']) <= .40):
        raise ValueError('Experimental capture envelope exceeded (240 s, 0.40 m)')
    camera = record['camera']
    if not camera.get('depth_aligned_to_color'):
        raise ValueError('Aligned metric RGBD required')
    memory = SceneMemory(max_age_s=240, max_points=600000)
    hashes, timestamps, frames = set(), [], []
    for i, view in enumerate(views):
        if (view.get('index') != i or view['target_relative_yaw_deg'] != i * 30
                or not view.get('capture_stable') or view['rgb_depth_skew_s'] > .1):
            raise ValueError('Invalid heading, capture stability, or synchronization')
        filename = view['file']
        if Path(filename).name != filename or not filename.endswith('.npz'):
            raise ValueError('Invalid capture file')
        path = directory / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Repeated capture file refused')
        hashes.add(digest)
        pose = view['pose']
        stamp = float(view['capture_monotonic_s'])
        if abs(pose['position_monotonic_s'] - stamp) > .1:
            raise ValueError('Image/position time mismatch')
        if abs(math.atan2(math.sin(math.radians(pose['relative_yaw_deg'] - i * 30)),
                          math.cos(math.radians(pose['relative_yaw_deg'] - i * 30)))) > math.radians(5):
            raise ValueError('Measured heading mismatch')
        with np.load(path, allow_pickle=False) as data:
            frame = RGBDFrame(data['rgb'], data['depth_m'], tuple(camera['intrinsics']),
                              np.asarray(view['world_camera']),
                              (*pose['position_m'][:2], pose['rpy_rad'][2]),
                              stamp, stamp - view['rgb_depth_skew_s'], pose['pose_monotonic_s'])
            if frame.rgb.shape[:2] != (camera['height'], camera['width']):
                raise ValueError('Image dimensions differ from actual camera')
            memory.update(frame)
            frames.append(frame)
        timestamps.append(stamp)
    final = record['final_displacement']
    if abs(final['heading_change_deg']) > 5:
        raise ValueError('Circle did not return to its initial heading')
    pose = (*final['final_position_xy_m'],
            record['start_pose']['rpy_rad'][2] + math.radians(final['heading_change_deg']))
    if not np.isfinite(pose).all():
        raise ValueError('Invalid final pose')
    height = float(np.median([v['world_camera'][2][3] for v in views]))
    now = timestamps[-1]
    observations, coverage = {}, []
    for slot, frame in enumerate(frames):
        suffix = '' if slot == 0 else '_' + str(slot * 30)
        observations['rgb' + suffix] = cv2.resize(frame.rgb, (224, 224), interpolation=cv2.INTER_AREA)[None]
        valid = np.isfinite(frame.depth_m) & (frame.depth_m > .1) & (frame.depth_m <= 10)
        depth = np.where(valid, frame.depth_m, 0)
        observations['depth' + suffix] = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_NEAREST)[None, ..., None] / 10
        coverage.append(float(valid.mean()))
    audit = {'method': 'current_circle_twelve_native_rgbd_frames',
             'experimental': True, 'training_view_equivalence': False,
             'prior_circles_used': 0, 'measured_source_views': 12,
             'source_fov_deg': camera['fov_deg'],
             'output_fov_deg': camera['fov_deg'], 'view_mode': 'native-frames',
             'measured_pixel_fraction': coverage,
             'unknown_rgb': 'captured color retained even without depth', 'unknown_depth': 0,
             'source_span_s': timestamps[-1] - timestamps[0],
             'capture_elapsed_s': elapsed, 'capture_max_drift_m': record['max_base_drift_m'],
             'reference_pose_xy_yaw': pose, 'virtual_optical_height_m': height,
             'motion_dispatch': False,
             'limitations': ['unobserved vertical regions remain unknown',
                             'odometry-based reprojection includes localization uncertainty',
                             'scan is a planning snapshot, not a fresh obstacle check',
                             'near-floor blind region requires independent clearance confirmation',
                             'native lens coverage differs from waypoint predictor training camera',
                             'per-heading translations are not corrected in native images']}
    return observations, memory, pose, now, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import cv2
    import numpy as np
    observations, memory, pose, stamp, audit = prepare(args.directory)
    output = results_dir() / 'plans' / (str(time.time_ns()) + '-measured')
    output.mkdir(parents=True)
    tiles = []
    for slot in range(12):
        suffix = '' if slot == 0 else '_' + str(slot * 30)
        tile = observations['rgb' + suffix][0].copy()
        cv2.putText(tile, '%d deg / %.0f%% measured' % (slot * 30, audit['measured_pixel_fraction'][slot] * 100),
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 220, 0), 1)
        tiles.append(tile)
    montage = np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0, 12, 4)], axis=0)
    cv2.imwrite(str(output / 'model-views.jpg'), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    np.savez_compressed(output / 'observations.npz', **observations)
    (output / 'observation.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'observation': audit, 'output_directory': str(output)}), flush=True)
    if args.prepare_only:
        return
    import torch
    from g1prep.model import OfficialPolicy
    from g1prep.planner import ReplayPlanner
    torch.set_num_threads(4)
    started = time.monotonic()
    with torch.inference_mode():
        policy = OfficialPolicy(device=args.device)
        planner = ReplayPlanner(policy, args.instruction)
        result = planner.plan(observations, pose)
    previous = pose[:2]
    route_checks = []
    for point in result.get('path_xy_m', []):
        ok = memory.path_observed(previous, point, stamp, 0., .30,
                                  max_age_s=audit['source_span_s'] + 1,
                                  visibility_z=audit['virtual_optical_height_m'])
        delta = np.asarray(point) - np.asarray(previous)
        route_checks.append({'from_xy_m': list(previous), 'to_xy_m': point,
                             'distance_m': float(np.linalg.norm(delta)),
                             'relative_left_deg': math.degrees(math.atan2(delta[1], delta[0]) - pose[2]) % 360,
                             'measured_snapshot_corridor_clear': bool(ok),
                             'fresh_execution_clearance': False})
        previous = point
    report = {'instruction': args.instruction, 'model': 'official ETP-R1 R2R iteration 270',
              'official_parameters': policy.loaded_keys, 'observation': audit,
              'model_output': result, 'route_checks': route_checks,
              'load_and_inference_s': time.monotonic() - started,
              'motion_dispatch': False, 'output_directory': str(output)}
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
