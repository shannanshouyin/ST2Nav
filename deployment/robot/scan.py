"""Measured in-place turns and twelve raw RGBD views; never runs navigation."""
import argparse
import os
import fcntl
import hashlib
import json
import math
from pathlib import Path
from storage import results_dir
import select
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def load_calibration(config):
    import numpy as np
    path = config.get('camera', {}).get('camera_to_base_calibration')
    if not path or path == 'UNCONFIRMED':
        return None
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if (value.get('schema') != 'g1_head_color_static_calibration_v1'
            or value.get('status') != 'STATIC_VERIFIED_REQUIRES_MOTION_VALIDATION'
            or value.get('camera_serial') != config['camera']['serial_number']):
        raise ValueError('Camera installation calibration does not match this robot setup')
    matrix = np.asarray(value['T_head_color'], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(matrix[:3, :3]) - 1) > 1e-6):
        raise ValueError('Invalid camera installation transform')
    model = Path(value['kinematic_model_file'])
    if hashlib.sha256(model.read_bytes()).hexdigest() != value['kinematic_model_sha256']:
        raise ValueError('Robot geometry differs from installation calibration')
    joints = {j.attrib['name']: j for j in ET.fromstring(model.read_text()).findall('joint')}
    chain = []
    for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'head_joint'):
        joint = joints[name]
        origin = joint.find('origin')
        if any(float(x) != 0 for x in origin.attrib.get('rpy', '0 0 0').split()):
            raise ValueError('Unsupported rotated robot joint origin')
        xyz = [float(x) for x in origin.attrib.get('xyz', '0 0 0').split()]
        axis = None if joint.attrib['type'] == 'fixed' else [float(x) for x in joint.find('axis').attrib['xyz'].split()]
        chain.append((xyz, axis))
    value['_chain'], value['_file'], value['_sha256'] = chain, str(path), hashlib.sha256(raw).hexdigest()
    return value


def camera_pose(pose, calibration):
    import numpy as np
    if calibration is None:
        return {}
    if pose.get('waist_mode_pr') != 0:
        raise ValueError('Calibrated camera pose requires measured waist roll/pitch representation')
    q = pose.get('waist_q_rad')
    if q is None or len(q) != 3 or not np.isfinite(q).all():
        raise ValueError('Calibrated camera pose requires fresh waist angles')
    body_head = np.eye(4)
    for (xyz, axis), angle in zip(calibration['_chain'], [*q, 0.]):
        origin = np.eye(4)
        origin[:3, 3] = xyz
        body_head = body_head @ origin
        if axis is not None:
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            rotation = np.eye(4)
            rotation[:3, :3] = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross
            body_head = body_head @ rotation
    body_camera = body_head @ np.asarray(calibration['T_head_color'])
    roll, pitch, yaw = pose['rpy_rad']
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    world_body = np.eye(4)
    height = pose.get('body_height_m')
    if height is None or not math.isfinite(height) or not .4 <= height <= 1.2:
        raise ValueError('Fresh body height above the local floor is required')
    world_body[:3, :3] = rz @ ry @ rx
    # Odometry z drifts and is not height above the floor. Keep its raw value
    # in position_m, but use the controller's separate measured body_height.
    world_body[:3, 3] = [*pose['position_m'][:2], height]
    if 'odometry_rpy_rad' in pose and abs(wrap(pose['odometry_rpy_rad'][2] - yaw)) > math.radians(3):
        raise ValueError('Odometry and body heading disagree; camera world pose is unconfirmed')
    return {'T_base_color': body_camera.tolist(), 'world_camera': (world_body @ body_camera).tolist(),
            'world_frame': 'odometry xy and heading; measured body height above the locally flat floor; z up',
            'camera_pose_source': 'measured body attitude, odometry xy, body height, waist angles, and static head calibration'}


def check_camera_calibration(metadata, calibration):
    if calibration is None:
        return
    import numpy as np
    expected = calibration['camera_verified_profile']
    if (metadata['serial'] != calibration['camera_serial']
            or metadata['width'] != expected['width'] or metadata['height'] != expected['height']
            or not np.allclose(metadata['intrinsics'], expected['intrinsics'], atol=.05, rtol=0)):
        raise ValueError('Camera profile differs from installation calibration')


def capture_profile(config):
    """Explicit stream sizes; mounting rotation is recorded by calibration, never by image rotation."""
    profile = config.get('camera', {}).get('capture_profile', {})
    color = tuple(profile.get('color_size', [640, 480]))
    depth = tuple(profile.get('depth_size', [640, 480]))
    fps = profile.get('fps', 15)
    if (color, depth, fps) not in (
            ((640, 480), (640, 480), 15),
            ((1280, 720), (1280, 720), 15)):
        raise ValueError('Unsupported capture profile; verify the camera mode before adding it')
    return color, depth, fps


def camera_worker(reset=False):
    # RealSense is already installed in the board's system Python.
    import numpy as np
    import pyrealsense2 as rs
    config = json.loads((ROOT / 'config.json').read_text())
    if reset:
        serial = config['camera']['serial_number']
        devices = [d for d in rs.context().query_devices()
                   if d.get_info(rs.camera_info.serial_number) == serial]
        if len(devices) != 1:
            raise RuntimeError('Configured camera unavailable for recovery: ' + serial)
        devices[0].hardware_reset()
        del devices
        time.sleep(2.)
        deadline = time.monotonic() + 8.
        while time.monotonic() < deadline:
            if any(d.get_info(rs.camera_info.serial_number) == serial
                   for d in rs.context().query_devices()):
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Camera did not reconnect after its reset')
    pipeline, setup = rs.pipeline(), rs.config()
    setup.enable_device(config['camera']['serial_number'])
    color_size, depth_size, fps = capture_profile(config)
    setup.enable_stream(rs.stream.color, *color_size, rs.format.rgb8, fps)
    setup.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
    profile = pipeline.start(setup)
    try:
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        metadata = {'name': device.get_info(rs.camera_info.name),
                    'serial': device.get_info(rs.camera_info.serial_number),
                    'usb': device.get_info(rs.camera_info.usb_type_descriptor),
                    'width': intr.width, 'height': intr.height, 'fps': fps,
                    'intrinsics': [intr.fx, intr.fy, intr.ppx, intr.ppy],
                    'distortion_model': str(intr.model), 'distortion_coefficients': list(intr.coeffs),
                    'fov_deg': list(rs.rs2_fov(intr)), 'depth_unit_m': scale,
                    'depth_aligned_to_color': True}
        for _ in range(20):
            pipeline.wait_for_frames(5000)
        print(json.dumps({'camera': metadata}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('close'):
                break
            # Discard the buffered frame and collect fresh frames after stopping.
            for _ in range(7):
                frames = pipeline.wait_for_frames(5000)
            # USB startup can briefly yield mismatched clock domains or frames.
            # Wait for a fresh matched pair while stationary; never relax the skew.
            for pair_attempt in range(12):
                aligned = align.process(frames)
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if color and depth:
                    same_clock = color.get_frame_timestamp_domain() == depth.get_frame_timestamp_domain()
                    skew = abs(color.get_timestamp() - depth.get_timestamp()) / 1000.
                    if same_clock and skew <= .1:
                        break
                if pair_attempt < 11:
                    frames = pipeline.wait_for_frames(500)
            else:
                raise RuntimeError('No synchronized RGB/depth pair after twelve fresh frames')
            stamp = time.monotonic()
            rgb = np.asanyarray(color.get_data()).copy()
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * scale
            valid = np.isfinite(depth_m) & (depth_m > .1) & (depth_m <= 10.)
            if rgb.shape != (color_size[1], color_size[0], 3) or depth_m.shape != (color_size[1], color_size[0]):
                raise RuntimeError('Unexpected RGBD shape')
            if valid.mean() < .3 or rgb.std() < 3:
                raise RuntimeError('Insufficient measured depth or blank color image')
            destination = Path(request['file'])
            np.savez_compressed(str(destination), rgb=rgb, depth_m=depth_m)
            print(json.dumps({'file': destination.name, 'capture_monotonic_s': stamp,
                              'color_timestamp_ms': color.get_timestamp(),
                              'depth_timestamp_ms': depth.get_timestamp(),
                              'timestamp_domain': str(color.get_frame_timestamp_domain()),
                              'depth_timestamp_domain': str(depth.get_frame_timestamp_domain()),
                              'pair_attempts': pair_attempt + 1,
                              'rgb_depth_skew_s': skew, 'valid_depth_fraction': float(valid.mean()),
                              'rgb_std': float(rgb.std())}), flush=True)
    finally:
        pipeline.stop()


class CameraError(RuntimeError):
    pass


def settling_quality(poses, target_degrees, heading_tolerance_deg=3.):
    """Judge a fresh stationary window, including normal small balance motion."""
    quality = capture_quality(poses, target_degrees)
    stamps = [p['pose_monotonic_s'] for p in poses]
    duration = max(stamps) - min(stamps)
    ordered = sorted(set(stamps))
    gap = max((b - a for a, b in zip(ordered, ordered[1:])), default=0.)
    yaw_rate = max(abs(p['gyroscope_rad_s'][2]) for p in poses)
    stable = (len(ordered) >= 8 and duration >= .6 and gap <= .15
              and quality['max_heading_error_during_capture_deg'] < heading_tolerance_deg
              and quality['angular_span_during_capture_deg'] <= 2.
              and quality['position_span_during_capture_m'] <= .02
              and quality['max_angular_rate_during_capture_rad_s'] < .15
              and yaw_rate < .08)
    return {**quality, 'settled': stable, 'window_duration_s': duration,
            'largest_state_gap_s': gap, 'unique_state_samples': len(ordered),
            'max_yaw_rate_rad_s': yaw_rate}


class Camera:
    def __init__(self, tick=None):
        self.process = None
        self.metadata = None
        self.restarts = 0
        try:
            try:
                self.open(tick)
            except CameraError as error:
                self.recover(tick, error)
        except BaseException:
            self.close()
            raise

    def open(self, tick, reset=False):
        self.process = subprocess.Popen([os.environ.get('CAMERA_PYTHON', sys.executable), '-u', '-B', str(ROOT / 'scan.py'),
                                         '_camera_reset' if reset else '_camera'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        metadata = self.receive(tick, timeout=20. if reset else 15.).get('camera')
        if not isinstance(metadata, dict):
            raise CameraError('Missing camera profile')
        if self.metadata is not None and metadata != self.metadata:
            raise ValueError('Camera profile changed during recovery')
        self.metadata = metadata

    def receive(self, tick=None, timeout=12.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if tick:
                tick()
            if select.select([self.process.stdout], [], [], .0)[0]:
                line = self.process.stdout.readline()
                if not line:
                    raise CameraError('Camera process exited before replying')
                try:
                    response = json.loads(line)
                except ValueError as error:
                    raise CameraError('Invalid camera reply') from error
                if not isinstance(response, dict):
                    raise CameraError('Invalid camera reply')
                if 'camera_error' in response:
                    raise CameraError(response['camera_error'])
                return response
            time.sleep(.05)
        raise CameraError('Camera response timed out after %.0f seconds' % timeout)

    def recover(self, tick, error):
        # Only callers that have stopped may use Camera. Never retry a turn,
        # a state/remote interruption, or the entire scan here.
        while self.restarts < 2:
            self.restarts += 1
            self.close()
            print('相机暂时未能取图，保持静止，重启相机后重新取图（%d/2）：%s'
                  % (self.restarts, error), file=sys.stderr, flush=True)
            for _ in range(10):
                if tick:
                    tick()
                time.sleep(.05)
            try:
                self.open(tick, reset=True)
                return
            except CameraError as retry_error:
                error = retry_error
        raise CameraError('Camera failed after two stationary restarts: ' + str(error)) from error

    def capture(self, path, tick, *, compress=True):
        while True:
            try:
                if tick:
                    tick()
                try:
                    self.process.stdin.write(json.dumps({'file': str(path), 'compress': compress}) + '\n')
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError) as error:
                    raise CameraError('Camera request pipe closed') from error
                response = self.receive(tick)
                if response.get('file') != Path(path).name or 'capture_monotonic_s' not in response:
                    raise CameraError('Missing fresh camera capture')
                return {**response, 'camera_restarts': self.restarts}
            except CameraError as error:
                self.recover(tick, error)

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"close":true}\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass  # Closing an already broken input pipe may flush buffered data.
        self.process = None


def wait_for_position(state, seconds=3.):
    """Wait while stopped for newly received odometry before creating a stream."""
    begun = time.monotonic()
    deadline = begun + seconds
    while time.monotonic() < deadline:
        state.permit()
        with state.lock:
            fresh = (state.sport is not None and state.sport_time >= begun
                     and time.monotonic() - state.sport_time < .1)
        if fresh:
            return
        time.sleep(.02)
    raise RuntimeError('No new measured position during stopped phase handoff; no movement sent')


class Turner:
    def __init__(self, state, loco, stream, rate, max_drift=.50):
        self.state, self.loco, self.stream, self.rate = state, loco, stream, rate
        self.previous_yaw = self.origin = None
        self.heading = self.max_drift = 0.
        self.started = time.monotonic()
        self.history = []
        self.drift_limit = max_drift

    def read(self):
        self.state.permit()
        if time.monotonic() - self.started > 480:
            raise RuntimeError('Scan exceeded 480 seconds')
        with self.state.lock:
            low, sport = self.state.low, self.state.sport
            if sport is None or time.monotonic() - self.state.sport_time > .3:
                raise RuntimeError('Fresh measured position required')
            rpy = list(low.imu_state.rpy)
            gyro = list(low.imu_state.gyroscope)
            position = list(sport.position)
            stamp, position_stamp = self.state.low_time, self.state.sport_time
            waist = [float(low.motor_state[i].q) for i in (12, 13, 14)] if hasattr(low, 'motor_state') else None
            waist_mode = int(low.mode_pr) if hasattr(low, 'mode_pr') else None
            odometry_rpy = list(sport.imu_state.rpy) if hasattr(sport, 'imu_state') else None
            body_height = float(sport.body_height) if hasattr(sport, 'body_height') else None
        if not all(math.isfinite(v) for v in (*position, *gyro)):
            raise RuntimeError('Invalid measured position or rotation rate')
        if self.previous_yaw is not None:
            delta = wrap(rpy[2] - self.previous_yaw)
            if abs(delta) > math.radians(15):
                raise RuntimeError('Measured heading jumped')
            self.heading += delta
        self.previous_yaw = rpy[2]
        if self.origin is None:
            self.origin = position[:2]
        drift = math.hypot(position[0] - self.origin[0], position[1] - self.origin[1])
        self.max_drift = max(self.max_drift, drift)
        if drift > self.drift_limit:
            raise RuntimeError('Base drift exceeded %.0f cm during in-place test' % (self.drift_limit * 100))
        value = {'pose_monotonic_s': stamp, 'position_monotonic_s': position_stamp,
                 'rpy_rad': rpy, 'gyroscope_rad_s': gyro, 'position_m': position,
                 'relative_yaw_deg': math.degrees(self.heading), 'drift_m': drift,
                 'waist_q_rad': waist, 'waist_mode_pr': waist_mode}
        value['body_height_m'] = body_height
        if odometry_rpy is not None:
            value['odometry_rpy_rad'] = odometry_rpy
        self.history.append(value)
        return value

    def tick(self, rate=0.):
        value = self.read()
        self.stream.poll()
        self.stream.send(0., rate)
        return value

    def settle(self, target, heading_tolerance_deg=3.):
        deadline, window = time.monotonic() + 10., []
        while time.monotonic() < deadline:
            value = self.tick()
            window.append(value)
            cutoff = value['pose_monotonic_s'] - .7
            window = [p for p in window if p['pose_monotonic_s'] >= cutoff]
            quality = settling_quality(window, math.degrees(target), heading_tolerance_deg)
            if quality['settled']:
                return value
            time.sleep(.05)
        raise RuntimeError('Turn settling failed over the latest stationary window: '
                           + json.dumps(quality))

    def turn(self, target, heading_tolerance_deg=3.):
        self.read()
        start, begin = time.monotonic(), self.heading
        direction = 1. if target > begin else -1.
        while time.monotonic() - start < 12.:
            self.read()
            error = target - self.heading
            if abs(error) < math.radians(1.2):
                return self.settle(target, heading_tolerance_deg)
            if direction * (self.heading - begin) < -math.radians(3):
                raise RuntimeError('Robot turned opposite to requested direction')
            if time.monotonic() - start > 3 and abs(self.heading - begin) < math.radians(1):
                raise RuntimeError('No measured turn; stopping')
            rate = math.copysign(min(self.rate, max(.18, 1.6 * abs(error))), error)
            self.tick(rate)
            time.sleep(.05)
        raise RuntimeError('Turn did not reach target within 12 seconds')


def stop_segment(loco, stream):
    results = []
    for _ in range(3):
        results.append(loco.SetVelocity(0., 0., 0., .15))
        time.sleep(.05)
    report = {'replies': stream.report(), 'stop_return_codes': results}
    stream.close()
    print(json.dumps({'stop_return_codes': results}), flush=True)
    if not any(v == 0 for v in results):
        raise RuntimeError('No stop acknowledgement; use physical remote')
    return report


def walking_mode(loco, state):
    """Recover read-only reply loss while stopped; never infer a mode or retry motion."""
    import robot
    deadline = time.monotonic() + 20.
    probe, recovered = loco, None
    failures = []
    try:
        for attempt in range(12):
            state.permit()
            if time.monotonic() >= deadline:
                break
            if attempt == 3:
                # Recreate only the query connection. The motion client is unchanged.
                recovered = probe = robot.client()
            code, mode = probe.GetFsmId()
            state.permit()  # A remote stop during a query must prevent continuation.
            if code == 0:
                if mode not in (500, 501):
                    raise RuntimeError('Regular walking mode required; current mode ' + str(mode))
                if failures:
                    print(json.dumps({'walking_mode_query_recovered': True,
                        'failed_codes': failures, 'read_only_attempts': attempt + 1,
                        'query_connection_recreated': recovered is not None, 'fsm': mode}), flush=True)
                return mode
            if code not in (3102, 3104):
                raise RuntimeError('Walking mode query rejected: ' + str(code))
            failures.append(code)
            if attempt == 0:
                print('行走模式查询暂未收到回复，保持停止并重新确认；不会重发运动。', flush=True)
            until = min(deadline, time.monotonic() + min(.3 * (attempt + 1), 1.))
            while time.monotonic() < until:
                state.permit()
                time.sleep(max(0., min(.05, until - time.monotonic())))
        raise RuntimeError('Walking mode remains unconfirmed after bounded read-only recovery; '
                           'no new movement sent: ' + json.dumps(failures))
    finally:
        if recovered is not None:
            stub = recovered._ClientBase__stub
            stub._ClientStub__sendChannel.CloseWriter()
            stub._ClientStub__recvChannel.CloseReader()


def capture_quality(poses, target_degrees):
    error = max(abs(p['relative_yaw_deg'] - target_degrees) for p in poses)
    angles = [[math.degrees(p['rpy_rad'][axis]) for p in poses] for axis in (0, 1)]
    angles.append([p['relative_yaw_deg'] for p in poses])
    span = max(max(values) - min(values) for values in angles)
    x, y = poses[0]['position_m'][:2]
    shift = max(math.hypot(p['position_m'][0] - x, p['position_m'][1] - y) for p in poses)
    rate = max(abs(v) for p in poses for v in p['gyroscope_rad_s'])
    heading_ok = error <= 5
    body_still = span <= 2 and shift <= .02 and rate < .15
    return {'capture_stable': heading_ok and body_still,
            'heading_within_tolerance': heading_ok, 'body_stationary': body_still,
            'max_heading_error_during_capture_deg': error, 'angular_span_during_capture_deg': span,
            'position_span_during_capture_m': shift, 'max_angular_rate_during_capture_rad_s': rate}


def capture_view(camera, directory, name, turner, target_degrees, calibration):
    for attempt in range(3):
        begin = len(turner.history)
        frame = camera.capture(directory / name, turner.read)
        turner.read()
        # Include useful samples arriving shortly after a fast camera response.
        while time.monotonic() < frame['capture_monotonic_s'] + .1:
            time.sleep(.02)
            turner.read()
        poses = turner.history[begin:]
        quality = capture_quality(poses, target_degrees)
        stamp = frame['capture_monotonic_s']
        nearest = min(poses, key=lambda p: max(abs(p[key] - stamp)
                      for key in ('pose_monotonic_s', 'position_monotonic_s')))
        ages = {key: abs(nearest[key] - stamp)
                for key in ('pose_monotonic_s', 'position_monotonic_s')}
        aligned = max(ages.values()) <= .1
        if quality['capture_stable'] and aligned:
            break
        print(json.dumps({'retry_stationary_capture': name, 'attempt': attempt + 1,
                          'quality': quality, 'pose_time_errors_s': ages}), flush=True)
    if not quality['heading_within_tolerance']:
        raise RuntimeError('Capture heading error %.2f deg exceeded 5 deg after three attempts'
                           % quality['max_heading_error_during_capture_deg'])
    if not quality['body_stationary']:
        raise RuntimeError('Body moved during three capture attempts: ' + json.dumps(quality))
    if not aligned:
        raise RuntimeError('No sufficiently recent pose after three stationary captures: ' + json.dumps(ages))
    return {**frame, 'target_relative_yaw_deg': target_degrees, 'pose': nearest, **quality,
            'capture_attempts': attempt + 1,
            'nearest_pose_age_s': abs(nearest['pose_monotonic_s'] - frame['capture_monotonic_s']),
            'nearest_position_age_s': ages['position_monotonic_s'],
            **camera_pose(nearest, calibration)}


def measure_after_stop(state, start_pose):
    """Read only for one second after stopping; do not reuse a mid-turn pose."""
    samples = []
    deadline = time.monotonic() + 1.
    while time.monotonic() < deadline:
        with state.lock:
            if (state.sport is None or time.monotonic() - state.sport_time > .3
                    or time.monotonic() - state.low_time > .3):
                raise RuntimeError('Fresh state required for final displacement measurement')
            xy = list(state.sport.position[:2])
            yaw = float(state.low.imu_state.rpy[2])
            gyro = list(state.low.imu_state.gyroscope)
        if not all(math.isfinite(v) for v in (*xy, yaw, *gyro)):
            raise RuntimeError('Invalid final displacement measurement')
        samples.append((xy, yaw, max(abs(v) for v in gyro)))
        time.sleep(.05)
    import numpy as np
    window = samples[len(samples) // 2:]
    final = np.median([v[0] for v in window], axis=0)
    dx, dy = final - np.asarray(start_pose['position_m'][:2])
    yaw0 = start_pose['rpy_rad'][2]
    final_yaw = math.atan2(sum(math.sin(v[1]) for v in window), sum(math.cos(v[1]) for v in window))
    span = max(math.hypot(v[0][0] - final[0], v[0][1] - final[1]) for v in window)
    peak_gyro = max(v[2] for v in window)
    return {'source': 'G1 odometry; not independently tape-measured', 'sample_count': len(samples),
            'final_position_xy_m': final.tolist(), 'delta_world_xy_m': [float(dx), float(dy)],
            'delta_start_body_forward_left_m': [float(math.cos(yaw0) * dx + math.sin(yaw0) * dy),
                                               float(-math.sin(yaw0) * dx + math.cos(yaw0) * dy)],
            'net_distance_m': float(math.hypot(dx, dy)),
            'heading_change_deg': math.degrees(wrap(final_yaw - yaw0)),
            'position_span_last_half_second_m': span, 'peak_gyro_last_half_second_rad_s': peak_gyro,
            'stationary_by_sensors': span <= .02 and peak_gyro < .15}


def wait_after_stop(state, start_pose):
    """Wait only while stopped; a brief settling delay need not spoil a scan."""
    for attempt in range(5):
        state.permit()
        result = measure_after_stop(state, start_pose)
        result['stationary_check_attempts'] = attempt + 1
        if result['stationary_by_sensors'] or attempt == 4:
            return result
        time.sleep(.2)


def main():
    import robot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['turn', 'scan', 'capture'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--degrees', type=float, default=30.)
    parser.add_argument('--rate', type=float, default=.3, help='maximum yaw rate in rad/s')
    parser.add_argument('--max-drift', type=float, default=.50, help='raw capture displacement limit in meters')
    args = parser.parse_args()
    if not math.isfinite(args.degrees) or not 0 < abs(args.degrees) <= 60:
        parser.error('turn degrees must be finite, nonzero and within -60..60')
    if not math.isfinite(args.rate) or not .18 <= args.rate <= .5:
        parser.error('yaw rate must be within 0.18..0.5 rad/s')
    if not math.isfinite(args.max_drift) or not .15 <= args.max_drift <= .5:
        parser.error('raw capture max drift must be within 0.15..0.5 m')
    guard = open('/tmp/etpr1-walk.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((ROOT / 'config.json').read_text())
    calibration = load_calibration(config) if args.command != 'turn' else None
    state = robot.State(config['interface'])
    camera = stream = turner = loco = None
    directory = None
    record = {'kind': 'raw_measured_12_rgbd_views', 'complete': False, 'views': [],
              'motion_segments': [], 'camera_runs_only_after_confirmed_stop': True,
              'navigation_ready': False, 'camera_to_base': None,
              'raw_capture_drift_limit_m': args.max_drift,
              'unresolved': ['camera field of view differs from model input',
                             'camera mount has not been verified after robot rotation' if calibration else 'camera-to-base calibration is unconfirmed'],
              'yaw_source': 'body IMU; positive counterclockwise',
              'pose_alignment': 'nearest host receive time; not hardware synchronized'}
    if calibration:
        record['installation_calibration'] = {'file': calibration['_file'], 'sha256': calibration['_sha256'],
                                             'mount_link': 'head_link', 'per_view_body_and_waist_compensation': True}
    if args.command == 'capture':
        record['kind'] = 'stationary_measured_rgbd_with_camera_pose'

    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        state.wait()
        loco = robot.client()
        mode = walking_mode(loco, state)
        print(json.dumps({'command': args.command, 'execute': args.execute, 'fsm': mode,
                          'degrees': args.degrees if args.command == 'turn' else 360 if args.command == 'scan' else 0,
                          'max_yaw_rate_rad_s': args.rate, 'translation_command_m_s': 0,
                          'max_drift_m': args.max_drift}), flush=True)
        if args.command in ('scan', 'capture'):
            camera = Camera(tick=state.permit if args.execute else None)
            check_camera_calibration(camera.metadata, calibration)
            record['camera'] = camera.metadata
            print(json.dumps({'camera': camera.metadata}), flush=True)
            camera.close()
            camera = None
        if not args.execute:
            print('Read-only checks complete. No velocity commands sent.', flush=True)
            return
        directory = results_dir() / 'scans' / (str(time.time_ns()) + '-' + args.command)
        directory.mkdir(parents=True)
        state.arm()
        if args.command != 'capture':
            stream = robot.VelocityStream(loco)
        turner = Turner(state, loco, stream, args.rate, args.max_drift)
        record['start_pose'] = turner.read()
        if args.command == 'capture':
            camera = Camera(tick=turner.read)
            check_camera_calibration(camera.metadata, calibration)
            record['camera'] = camera.metadata
            for index in range(3):
                entry = capture_view(camera, directory, 'frame_%02d.npz' % index, turner, 0., calibration)
                record['views'].append({**entry, 'index': index})
            camera.close()
            camera = None
        elif args.command == 'turn':
            # Route turns use their measured endpoint for the next decision.
            # Keep scan headings at the stricter default required by capture.
            result = turner.turn(math.radians(args.degrees), heading_tolerance_deg=8.)
            record['turn_end'] = result
        else:
            for index in range(13):
                if stream is None:
                    walking_mode(loco, state)
                    stream = robot.VelocityStream(loco)
                    turner.stream = stream
                target = index * math.pi / 6
                pose = turner.turn(target) if index else turner.settle(0.)
                if index == 12:
                    record['return_pose'] = pose
                    break
                # This board loses motion replies while USB capture is active.
                # Finish movement before opening the camera; no velocity during capture.
                segment = stop_segment(loco, stream)
                record['motion_segments'].append(segment)
                stream = turner.stream = None
                camera = Camera(tick=turner.read)
                check_camera_calibration(camera.metadata, calibration)
                entry = capture_view(camera, directory, 'view_%03d.npz' % (index * 30), turner, index * 30, calibration)
                camera.close()
                camera = None
                record['views'].append({**entry, 'index': index})
                (directory / 'scan.json').write_text(json.dumps(record, indent=2))
                print(json.dumps({'view': index + 1, 'yaw_deg': round(entry['pose']['relative_yaw_deg'], 2),
                                  'depth_valid_fraction': round(entry['valid_depth_fraction'], 3)}), flush=True)
        record.update(complete=True, elapsed_s=time.monotonic() - turner.started,
                      max_base_drift_m=turner.max_drift)
    except BaseException as error:
        record['error'] = str(error) or type(error).__name__
        record['complete'] = False
        raise
    finally:
        state.disarm()
        if camera is not None:
            camera.close()
        if stream is not None:
            try:
                segment = stop_segment(loco, stream)
                record['motion_segments'].append(segment)
                record['stop_return_codes'] = segment['stop_return_codes']
            except RuntimeError as error:
                record['complete'] = False
                record['stop_error'] = str(error)
        if directory is not None:
            if turner is not None:
                record['motion_samples'] = turner.history
                record['max_base_drift_m'] = turner.max_drift
                if not record.get('stop_error'):
                    try:
                        record['final_displacement'] = measure_after_stop(state, record['start_pose'])
                        record['max_base_drift_m'] = max(turner.max_drift, record['final_displacement']['net_distance_m'])
                    except RuntimeError as error:
                        record['final_measurement_error'] = str(error)
                        record['complete'] = False
            (directory / 'scan.json').write_text(json.dumps(record, indent=2))
        robot.close_client(loco)
        state.close()
        guard.close()
    if record.get('stop_error'):
        raise RuntimeError(record['stop_error'])
    if record.get('final_measurement_error'):
        raise RuntimeError(record['final_measurement_error'])
    if directory is not None:
        print(json.dumps({'capture_complete': record['complete'], 'directory': str(directory),
                          'views': len(record['views']), 'max_base_drift_m': record['max_base_drift_m'],
                          'elapsed_s': record['elapsed_s'], 'final_displacement': record['final_displacement']}), flush=True)


if __name__ == '__main__':
    try:
        if sys.argv[1:] in (['_camera'], ['_camera_reset']):
            camera_worker(reset=sys.argv[1] == '_camera_reset')
        else:
            main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as error:
        if sys.argv[1:] in (['_camera'], ['_camera_reset']) and not isinstance(error, KeyboardInterrupt):
            print(json.dumps({'camera_error': str(error)}), flush=True)
        raise SystemExit(str(error) or 'Interrupted; stopped')
