"""One supervised circle, two fixed calibrated cameras; no forward motion."""
import argparse
import os
import fcntl
import hashlib
import json
import math
from pathlib import Path
from storage import results_dir
import signal
import subprocess
import sys
import time
import scan

ROOT = Path(__file__).resolve().parent
CALIBRATION = Path(os.environ.get('NAV_CALIBRATION', str(ROOT / 'dual-camera-static-calibration.json'))).expanduser()


def camera_worker(reset=False):
    import numpy as np
    import pyrealsense2 as rs
    calibration = json.loads(CALIBRATION.read_text())
    if reset:
        # Recovery is used only while stopped, after the previous worker closes.
        # Reopening a pipeline cannot recover a camera whose USB stream is stuck.
        serials = {item['serial'] for item in calibration['cameras']}
        for serial in sorted(serials):
            devices = [d for d in rs.context().query_devices()
                       if d.get_info(rs.camera_info.serial_number) == serial]
            if len(devices) != 1:
                raise RuntimeError('Configured camera unavailable for recovery: ' + serial)
            devices[0].hardware_reset()
            del devices
        time.sleep(2.)
        deadline = time.monotonic() + 10.
        while time.monotonic() < deadline:
            present = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
            if serials <= present:
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Calibrated cameras did not reconnect after reset')
    cameras = []
    try:
        for item in calibration['cameras']:
            pipeline, setup = rs.pipeline(), rs.config()
            setup.enable_device(item['serial'])
            for stream, fmt in ((rs.stream.color, rs.format.rgb8), (rs.stream.depth, rs.format.z16)):
                setup.enable_stream(stream, 1280, 720, fmt, 6)
            profile = pipeline.start(setup)
            device = profile.get_device()
            cameras.append((pipeline, rs.align(rs.stream.color), device.first_depth_sensor().get_depth_scale(), item))
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            actual = [intr.fx, intr.fy, intr.ppx, intr.ppy]
            if (intr.width, intr.height) != (1280, 720) or not np.allclose(actual, item['intrinsics'], atol=.01):
                raise RuntimeError('Capture profile differs from calibrated profile')
            if any(intr.coeffs):
                raise RuntimeError('Nonzero lens distortion requires explicit treatment')
        # Drain both devices continuously, allowing automatic exposure and global clocks to settle.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            for pipeline, *_ in cameras:
                pipeline.poll_for_frames()
            time.sleep(.01)
        print(json.dumps({'camera': {'serials': [c[3]['serial'] for c in cameras],
                                    'width': 1280, 'height': 720, 'fps': 6}}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('close'):
                break
            anchor = Path(request['file'])
            copied = []
            for pipeline, align, scale, item in cameras:
                pipeline.poll_for_frames()  # discard any buffered sample
                for attempt in range(12):
                    aligned = align.process(pipeline.wait_for_frames(3000))
                    color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                    if not color or not depth:
                        continue
                    domain = color.get_frame_timestamp_domain()
                    skew = abs(color.get_timestamp() - depth.get_timestamp()) / 1000
                    stamp = time.monotonic() + color.get_timestamp() / 1000 - time.time()
                    age = time.monotonic() - stamp
                    if (domain == rs.timestamp_domain.global_time
                            and depth.get_frame_timestamp_domain() == domain
                            and skew <= .1 and -.02 <= age <= .25):
                        break
                else:
                    raise RuntimeError('No fresh globally timed aligned RGBD sample')
                rgb = np.asanyarray(color.get_data()).copy()
                depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * scale
                valid = np.isfinite(depth_m) & (depth_m > .1) & (depth_m <= 10)
                if rgb.shape != (720, 1280, 3) or depth_m.shape != (720, 1280):
                    raise RuntimeError('Unexpected frame shape')
                if valid.mean() < .25 or rgb.std() < 3:
                    raise RuntimeError('Blank RGB or insufficient depth')
                copied.append((rgb, depth_m, {'serial': item['serial'],
                    'file': anchor.stem + '_' + item['serial'] + '.npz',
                    'capture_monotonic_s': stamp, 'host_frame_age_s': age,
                    'rgb_depth_skew_s': skew, 'timestamp_domain': str(domain),
                    'valid_depth_fraction': float(valid.mean())}))
            # Both exposures and timestamps are fixed before compression starts.
            for rgb, depth, metadata in copied:
                save = np.savez_compressed if request.get('compress', True) else np.savez
                save(anchor.parent / metadata['file'], rgb=rgb, depth_m=depth)
            print(json.dumps({'file': anchor.name, 'capture_monotonic_s': copied[-1][2]['capture_monotonic_s'],
                              'frames': [x[2] for x in copied]}), flush=True)
    finally:
        for pipeline, *_ in cameras:
            pipeline.stop()


def world_camera(pose, item, calibration):
    import numpy as np
    reference = np.median([s['waist_q_rad'] for s in calibration['body_reference']['capture_states']], axis=0)
    if pose['waist_mode_pr'] != 0 or np.max(np.abs(np.asarray(pose['waist_q_rad']) - reference)) > math.radians(2):
        raise RuntimeError('Waist posture differs from static installation calibration')
    r, p, y = pose['rpy_rad']
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
    height = pose['body_height_m']
    if height is None or not .5 < height < 1:
        raise RuntimeError('Valid measured pelvis height required')
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [*pose['position_m'][:2], height]
    return (transform @ np.asarray(item['T_pelvis_from_color_at_capture_waist'])).tolist()


class PairCamera(scan.Camera):
    def open(self, tick, reset=False):
        self.process = subprocess.Popen([os.environ.get('CAMERA_PYTHON', sys.executable), '-u', '-B', str(Path(__file__).resolve()),
            '_camera_reset' if reset else '_camera'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        metadata = self.receive(tick, timeout=35 if reset else 20)['camera']
        if self.metadata is not None and metadata != self.metadata:
            raise ValueError('Camera profile changed during recovery')
        self.metadata = metadata


def capture_pair(camera, directory, index, turner, calibration, prefix='view', *, compress=True):
    items = {c['serial']: c for c in calibration['cameras']}
    last_alignment_error = None
    for attempt in range(5):
        # A frame can predate its receipt by 0.25 s. Refill pose history after
        # blocking read-only queries before accepting either camera exposure.
        ready = time.monotonic() + .3
        while time.monotonic() < ready:
            turner.read()
            time.sleep(.02)
        begin = len(turner.history)
        pair = camera.capture(directory / ('%s_%03d.npz' % (prefix,index*30)), turner.read,
                              compress=compress)
        turner.read()
        captures = []
        for frame in pair['frames']:
            stamp = frame['capture_monotonic_s']
            poses = turner.history[max(0,begin-10):]
            nearest = min(poses,key=lambda p:max(abs(p[k]-stamp) for k in ('pose_monotonic_s','position_monotonic_s')))
            age = max(abs(nearest[k]-stamp) for k in ('pose_monotonic_s','position_monotonic_s'))
            if age > .1:
                last_alignment_error = age
                break
            captures.append({**frame,'pose':nearest,'pose_time_error_s':age,
                'world_camera':world_camera(nearest,items[frame['serial']],calibration),
                'target_relative_yaw_deg':index*30,'index':index})
        if len(captures) != len(items):
            print(json.dumps({'stationary_retry': attempt + 1,
                'pose_time_error_s': last_alignment_error, 'reason': 'recapture_with_fresh_pose'}), flush=True)
            continue
        quality = scan.capture_quality(turner.history[begin:],index*30)
        if quality['capture_stable']:
            return [{**f,**quality,'capture_attempts':attempt+1} for f in captures]
        print(json.dumps({'stationary_retry':attempt+1,'quality':quality}),flush=True)
    raise RuntimeError('No stable, pose-aligned camera pair after five stationary captures')


def main():
    import robot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    guard = open('/tmp/etpr1-walk.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calibration = json.loads(CALIBRATION.read_text())
    if calibration['status'] != 'STATIC_LOCAL_CHECKS_PASSED':
        raise RuntimeError('Checked static calibration required')
    config = json.loads((ROOT / 'config.json').read_text())
    state = robot.State(config['interface'])
    camera = stream = turner = loco = None
    directory = results_dir() / 'dual-scans' / str(time.time_ns())
    directory.mkdir(parents=True)
    record = {'kind': 'dual_camera_measured_circle', 'complete': False, 'execute': args.execute,
              'views': [], 'motion_segments': [], 'translation_command_m_s': 0,
              'calibration_sha256': hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
              'calibration': calibration, 'clock_alignment': 'global camera time to host monotonic; nearest received robot pose',
              'depth_bias_correction_applied': False}
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        state.wait()
        loco = robot.client()
        record['fsm'] = scan.walking_mode(loco, state)
        scan.wait_for_position(state)
        turner = scan.Turner(state, loco, None, .3, .50)
        record['start_pose'] = turner.read()
        # Read-only branch exercises both cameras, timestamps and transforms; no velocity stream.
        if args.execute:
            state.arm()
        for index in range(13 if args.execute else 1):
            if args.execute:
                scan.walking_mode(loco, state)
                scan.wait_for_position(state)
                stream = turner.stream = robot.VelocityStream(loco)
                pose = turner.turn(index * math.pi / 6) if index else turner.settle(0)
                record['motion_segments'].append(scan.stop_segment(loco, stream))
                stream = turner.stream = None
                if index == 12:
                    record['return_pose'] = pose
                    break
            camera = PairCamera(tick=turner.read)
            captures = capture_pair(camera,directory,index,turner,calibration)
            camera.close()
            camera = None
            record['views'].extend(captures)
            print(json.dumps({'captured_heading_deg': index*30, 'rgbd_count': len(record['views']),
                              'actual_yaw_deg': captures[0]['pose']['relative_yaw_deg'],
                              'depth_valid': [v['valid_depth_fraction'] for v in captures]}), flush=True)
        record['complete'] = args.execute
        record['read_only_check_passed'] = not args.execute
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        if camera is not None:
            camera.close()
        try:
            if stream is not None:
                record['motion_segments'].append(scan.stop_segment(loco, stream))
        except BaseException as error:
            record['stop_error'] = repr(error)
            print('STOP FAILED: use physical remote immediately', file=sys.stderr, flush=True)
        if turner is not None:
            record['motion_samples'] = turner.history
            record['elapsed_s'] = time.monotonic()-turner.started
            record['max_base_drift_m'] = turner.max_drift
            try:
                record['final_displacement'] = scan.wait_after_stop(state, record['start_pose'])
            except Exception as error:
                record['final_measurement_error'] = repr(error)
        if (record.get('stop_error') or record.get('final_measurement_error')
                or not record.get('final_displacement',{}).get('stationary_by_sensors')):
            record['complete'] = False
        (directory / 'scan.json').write_text(json.dumps(record, indent=2))
        state.disarm()
        robot.close_client(loco)
        state.close()
        print(json.dumps({'directory': str(directory), 'complete': record['complete'],
                          'capture_complete':record['complete'] and len(record['views'])==24,
                          'views': len(record['views']), 'final_displacement': record.get('final_displacement')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] in (['_camera'], ['_camera_reset']):
        try:
            camera_worker(reset=sys.argv[1] == '_camera_reset')
        except Exception as error:
            print(json.dumps({'camera_error': str(error)}), flush=True)
            raise
    else:
        main()
