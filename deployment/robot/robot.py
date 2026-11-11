"""G1 checks and a bounded straight-walk trial. Default never sends motion."""
import argparse
import os
import sys
import fcntl
import json
import math
from pathlib import Path
from storage import results_dir
import signal
import ssl
import struct
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
STOP = Path('/tmp/etpr1-stop')
WALKING_MODES = frozenset((500, 501))
RPC_TIMEOUT_S = 1.0
MAX_SPEED_M_S = .3
MAX_DURATION_S = 10.
MAX_PATH_DEVIATION_M = .60
MAX_HEADING_DEVIATION_DEG = 15.
BODY_CLEARANCE_RADIUS_M = .30
RPC_ERRORS = {
    3102: 'request could not be sent',
    3103: 'API not registered in the client',
    3104: 'no reply before timeout; execution is unknown',
    3105: 'reply API mismatch',
    3202: 'controller internal error',
    3203: 'API not implemented by the controller',
    3204: 'controller rejected the parameters',
    3205: 'control lease denied',
    7301: 'controller locomotion state is unavailable',
    7302: 'controller reports an invalid locomotion mode',
}


def rpc_error(code):
    return str(code) + ' (' + RPC_ERRORS.get(code, 'unclassified return code') + ')'


class ReplyChannel:
    """Drain batched replies; the SDK reader takes only one per notification."""
    def __init__(self, channel, stub):
        from cyclonedds.core import Listener
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        self.queue = stub._ClientStub__futureQueue
        self.error = None
        self.received = self.matched = self.max_batch = 0
        self.reader = DataReader(channel._Channel__participant, channel._Channel__topic,
                                 Qos(Policy.History.KeepLast(128)),
                                 Listener(on_data_available=self.receive))
        channel.CloseReader()

    def receive(self, reader):
        from cyclonedds.internal import InvalidSample
        try:
            while True:
                samples = reader.take(64)
                if not samples:
                    return
                self.max_batch = max(self.max_batch, len(samples))
                for sample in samples:
                    if isinstance(sample, InvalidSample):
                        continue
                    self.received += 1
                    future = self.queue.Get(sample.header.identity.id)
                    if future is not None:
                        if not future.Ready(sample):
                            raise RuntimeError('Reply future could not be completed')
                        self.matched += 1
        except Exception as error:
            self.error = str(error) or type(error).__name__

    def CloseReader(self):
        self.reader.set_listener(None)
        del self.reader

    def report(self):
        return {'received': self.received, 'matched_requests': self.matched,
                'max_batch': self.max_batch, 'error': self.error}


class VelocityStream:
    """Send short commands at a steady rate; require recent successful replies."""
    def __init__(self, loco):
        self.stub = loco._ClientBase__stub
        self.pending = {}
        self.last_accepted_sent = time.monotonic()
        self.sent = self.accepted = self.missing = 0
        self.max_reply_ms = 0.
        self.error_code = None

    def send(self, speed, yaw_rate=0.):
        from unitree_sdk2py.idl.unitree_api.msg.dds_ import (
            Request_, RequestHeader_, RequestIdentity_, RequestLease_, RequestPolicy_)
        identity = RequestIdentity_(time.monotonic_ns(), 7105)
        request = Request_(RequestHeader_(identity, RequestLease_(0), RequestPolicy_(0, False)),
                           json.dumps({'velocity': [speed, 0., yaw_rate], 'duration': .15}), [])
        sent = time.monotonic()
        future = self.stub.SendRequest(request, .05)
        if future is None:
            self.error_code = 3102
            raise RuntimeError('Velocity update failed: ' + rpc_error(3102))
        self.pending[identity.id] = (sent, future)
        self.sent += 1

    def poll(self):
        from unitree_sdk2py.utils.future import FutureResult
        if getattr(self.stub._ClientStub__recvChannel, 'error', None):
            self.error_code = 3001
            raise RuntimeError('Velocity reply receiver failed; stopping')
        now = time.monotonic()
        for identity, (sent, future) in list(self.pending.items()):
            result = future.GetResult(0.)
            if result.code == FutureResult.FUTURE_SUCC:
                del self.pending[identity]
                response = result.value
                code = response.header.status.code if response.header.identity.api_id == 7105 else 3105
                if code != 0:
                    self.error_code = code
                    raise RuntimeError('Velocity update failed: ' + rpc_error(code))
                self.accepted += 1
                self.last_accepted_sent = max(self.last_accepted_sent, sent)
                self.max_reply_ms = max(self.max_reply_ms, (now - sent) * 1000)
            elif result.code != FutureResult.FUTUTE_ERR_TIMEOUT:
                self.error_code = 3001
                raise RuntimeError('Velocity reply processing failed; stopping')
            elif now - sent >= 1.:
                # Get removes by request ID; this also avoids the old SDK Remove typo.
                self.stub._ClientStub__futureQueue.Get(identity)
                del self.pending[identity]
                self.missing += 1
        if now - self.last_accepted_sent > .5:
            self.error_code = 3104
            raise RuntimeError('No recent successful velocity reply for 0.5 seconds; stopping')

    def close(self):
        for identity in self.pending:
            self.stub._ClientStub__futureQueue.Get(identity)
        self.pending.clear()

    def report(self):
        report = {'sent': self.sent, 'accepted': self.accepted,
                'expired_without_reply': self.missing, 'pending': len(self.pending),
                'max_reply_observed_ms': round(self.max_reply_ms, 2)}
        receiver = self.stub._ClientStub__recvChannel
        if isinstance(receiver, ReplyChannel):
            report['reply_receiver'] = receiver.report()
        return report


def remote_input(message):
    data = bytes(message.wireless_remote)
    buttons = data[2] | (data[3] << 8)
    axes = [struct.unpack_from('<f', data, offset)[0] for offset in (4, 8, 12, 20)]
    return buttons, axes


def remote_active(message):
    buttons, axes = remote_input(message)
    return buttons != 0 or not all(math.isfinite(v) and abs(v) < .15 for v in axes)


class State:
    def __init__(self, interface, position=True):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, interface)
        self.low = self.sport = None
        self.low_time = self.sport_time = 0
        self.armed = self.manual_stop = False
        self.lock = threading.Lock()
        self.subscribers = []
        self.position_subscriber = None
        sub = ChannelSubscriber('rt/lowstate', LowState_)
        sub.Init(self.on_low, 1)
        self.subscribers.append(sub)
        self.subscribe_position(position)

    def subscribe_position(self, enabled):
        """Only the active phase reads odometry; keep remote/IMU monitoring live."""
        if enabled and self.position_subscriber is None:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            sub = ChannelSubscriber('rt/odommodestate', SportModeState_)
            sub.Init(self.on_sport, 1)
            self.position_subscriber = sub
            self.subscribers.append(sub)
        elif not enabled and self.position_subscriber is not None:
            sub = self.position_subscriber
            sub.Close()
            self.subscribers.remove(sub)
            self.position_subscriber = None
            with self.lock:
                self.sport, self.sport_time = None, 0

    def on_low(self, msg):
        with self.lock:
            self.low, self.low_time = msg, time.monotonic()
            if self.armed and remote_active(msg):
                self.manual_stop = True

    def on_sport(self, msg):
        with self.lock:
            self.sport, self.sport_time = msg, time.monotonic()

    def wait(self, seconds=3):
        deadline = time.monotonic() + seconds
        while self.low is None and time.monotonic() < deadline:
            time.sleep(.02)
        if self.low is None:
            raise RuntimeError('No G1 state received on ' + 'rt/lowstate')

    def report(self):
        with self.lock:
            return {'low_state_age_s': round(time.monotonic() - self.low_time, 3),
                    'body_rpy_deg': [round(math.degrees(v), 1) for v in self.low.imu_state.rpy],
                    'machine_mode': self.low.mode_machine,
                    'remote_buttons': self.low.wireless_remote[2] | (self.low.wireless_remote[3] << 8),
                    'navigation_position_available': self.sport is not None and time.monotonic() - self.sport_time < .3}

    def permit(self):
        with self.lock:
            if self.low is None or time.monotonic() - self.low_time > .3:
                raise RuntimeError('State stale; stopping')
            if self.manual_stop:
                raise RuntimeError('Remote input detected during trial; stopping, no automatic resume')
            rpy = self.low.imu_state.rpy
            if not all(math.isfinite(v) for v in rpy) or max(abs(rpy[0]), abs(rpy[1])) > math.radians(20):
                raise RuntimeError('Robot is not upright; no walking permitted')
            if remote_active(self.low):
                raise RuntimeError('Release all remote buttons and center the sticks; remote input stops the trial')
        if STOP.exists():
            raise RuntimeError('Stop requested')

    def arm(self):
        with self.lock:
            if not self.armed:
                self.manual_stop = remote_active(self.low)
            self.armed = True
        self.permit()

    def disarm(self):
        with self.lock:
            self.armed = False

    def monitor_remote(self):
        print('Read only for 10 seconds. Briefly press R1 alone, then release it. No motion sent.', flush=True)
        deadline, previous = time.monotonic() + 10, None
        while time.monotonic() < deadline:
            with self.lock:
                if time.monotonic() - self.low_time > .3:
                    raise RuntimeError('State stale; remote check stopped')
                buttons, axes = remote_input(self.low)
            sample = (buttons, tuple(round(v, 3) for v in axes))
            if sample != previous:
                print(json.dumps({'buttons': buttons, 'sticks': list(sample[1])}), flush=True)
                previous = sample
            time.sleep(.05)

    def close(self):
        for sub in self.subscribers:
            sub.Close()


def camera_report():
    code = ('import json,pyrealsense2 as rs; '
            'print(json.dumps([{"name":d.get_info(rs.camera_info.name),'
            '"serial":d.get_info(rs.camera_info.serial_number),'
            '"usb":d.get_info(rs.camera_info.usb_type_descriptor)} '
            'for d in rs.context().query_devices()]))')
    result = subprocess.run([os.environ.get('CAMERA_PYTHON', sys.executable), '-B', '-c', code], capture_output=True, text=True, timeout=12)
    return {'realsense': json.loads(result.stdout) if result.returncode == 0 else [],
            'camera_query_error': result.stderr.strip() if result.returncode else None,
            'video_nodes': sorted(str(p) for p in Path('/dev').glob('video*'))}


def client():
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    obj = LocoClient()
    obj.SetTimeout(RPC_TIMEOUT_S)
    obj.Init()
    stub = obj._ClientBase__stub
    stub._ClientStub__recvChannel = ReplyChannel(stub._ClientStub__recvChannel, stub)
    return obj


def close_client(obj):
    """Release DDS endpoints before Python tears down their callback state."""
    if obj is None:
        return
    stub = obj._ClientBase__stub
    stub._ClientStub__recvChannel.CloseReader()
    stub._ClientStub__sendChannel.CloseWriter()


def read_query(call):
    """Retry only lost read-only requests; never use this for movement."""
    for _ in range(3):
        result = call()
        if result[0] not in (3102, 3104):
            return result
        time.sleep(.2)
    return result


def controller_report():
    """Query service metadata only; never start, stop, or select a service."""
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
    report = {}
    for label, kind, method in [('active_controller', MotionSwitcherClient, 'CheckMode'),
                                ('motion_services', RobotStateClient, 'ServiceList')]:
        try:
            obj = kind()
            obj.SetTimeout(1.)
            obj.Init()
            code, data = getattr(obj, method)()
            if method == 'ServiceList' and code == 0:
                data = [vars(item) for item in data if item.name in
                        ('ai_sport', 'sport_mode', 'advanced_sport', 'motion_switcher')]
            report[label] = {'query_code': code, 'data': data}
        except Exception as error:
            report[label] = {'error': str(error)}
    return report


def locomotion_report(loco):
    """Read only: version, FSM submode, and controller locomotion state."""
    report = {}
    for name, call in [('api_version', loco.GetServerApiVersion),
                       ('fsm_mode', lambda: loco._Call(7002, '{}')),
                       ('balance_mode', lambda: loco._Call(7003, '{}'))]:
        started = time.monotonic()
        code, data = read_query(call) if name != 'balance_mode' else call()
        value = data if name == 'api_version' else json.loads(data).get('data') if code == 0 else None
        report[name] = {'code': code, 'value': value,
                        'elapsed_ms': round((time.monotonic() - started) * 1000, 2)}
        if code != 0:
            report[name]['error'] = rpc_error(code)
    return report


def walking_blockers(state, code, mode, locomotion=None):
    reasons = []
    if code != 0:
        reasons.append('Locomotion query failed (' + str(code) +
                       '); walking service unavailable or not connected. Mode is unknown.')
    elif mode not in WALKING_MODES:
        reasons.append('Regular walking mode 500 or 501 required; current mode is ' + str(mode))
    # Supplementary version/submode queries are diagnostic, not prerequisites.
    # The fresh FSM, state and actual motion replies still gate execution.
    try:
        state.permit()
    except RuntimeError as error:
        reasons.append(str(error))
    return reasons


def forward_progress(origin, current, heading, yaw, distance):
    if not all(math.isfinite(v) for v in (*origin, *current, heading, yaw, distance)):
        raise RuntimeError('Invalid position during distance-limited walking')
    dx, dy = current[0] - origin[0], current[1] - origin[1]
    forward = math.cos(heading) * dx + math.sin(heading) * dy
    lateral = -math.sin(heading) * dx + math.cos(heading) * dy
    error = math.atan2(math.sin(yaw - heading), math.cos(yaw - heading))
    if abs(lateral) > MAX_PATH_DEVIATION_M or abs(error) > math.radians(MAX_HEADING_DEVIATION_DEG) or forward < -.10 or forward > distance + .30:
        raise RuntimeError('Walking departed from the checked straight corridor: ' + json.dumps(
            {'forward_m': forward, 'lateral_m': lateral, 'heading_error_deg': math.degrees(error),
             'target_distance_m': distance, 'max_path_deviation_m': MAX_PATH_DEVIATION_M,
             'max_heading_deviation_deg': MAX_HEADING_DEVIATION_DEG}))
    return forward, forward >= distance - .10


def walk(state, seconds, speed, execute, distance=None, *, loco=None, prewalk=None):
    import scan
    # Reuse the phase's connection and recover only read-only reply loss.
    # Capture the checked corridor after recovery, never before a query delay.
    loco = client() if loco is None else loco
    mode = scan.walking_mode(loco, state)
    if prewalk is not None:
        distance = prewalk()
    if distance is not None and (not math.isfinite(distance) or not .15 <= distance <= 1.5):
        raise RuntimeError('Fresh walking distance must be within 0.15..1.5 m')
    print(json.dumps({'execute': execute, 'fsm': mode, 'speed_m_s': speed,
                      'duration_s': seconds, 'nominal_distance_m': speed * seconds,
                      'measured_target_distance_m': distance,
                      'max_path_deviation_m': MAX_PATH_DEVIATION_M if distance is not None else None,
                      'max_heading_deviation_deg': MAX_HEADING_DEVIATION_DEG if distance is not None else None}), flush=True)
    if mode not in WALKING_MODES:
        raise RuntimeError('Use the physical remote to enter regular walking mode first (FSM 500 or 501); no automatic mode switch')
    # Version and balance details belong to `check`; they do not gate a walk.
    # Fresh walking mode and successful velocity replies remain mandatory.
    blockers = walking_blockers(state, 0, mode)
    if distance is not None and (state.sport is None or time.monotonic() - state.sport_time > .3):
        blockers.append('Fresh position required for distance-limited walking')
    if blockers:
        raise RuntimeError('; '.join(blockers))
    if not execute:
        print('READY. Read-only checks passed; velocity acceptance and actual walking remain unverified. No motion sent. R1 need not be held.')
        return
    command_error = None
    stop_results = []
    reached, progress = False, 0.
    with state.lock:
        origin = list(state.sport.position[:2]) if distance is not None else None
        heading = float(state.low.imu_state.rpy[2])
    previous_position = origin
    stream = VelocityStream(loco)
    try:
        state.arm()
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            state.permit()
            if distance is not None:
                with state.lock:
                    if state.sport is None or time.monotonic() - state.sport_time > .3:
                        raise RuntimeError('Position stale during distance-limited walking')
                    current = list(state.sport.position[:2])
                    yaw = float(state.low.imu_state.rpy[2])
                if math.hypot(current[0] - previous_position[0], current[1] - previous_position[1]) > .10:
                    raise RuntimeError('Position jumped during distance-limited walking')
                previous_position = current
                progress, reached = forward_progress(origin, current, heading, yaw, distance)
                if reached:
                    break
                if time.monotonic() - started > 3 and progress < .03:
                    raise RuntimeError('No measured forward progress; stopping')
            stream.poll()
            stream.send(speed)
            time.sleep(.05)
        stream.poll()
    except RuntimeError as error:
        command_error = {'operation': 'forward', 'code': stream.error_code,
                         'error': str(error), 'execution_confirmed': False}
        raise
    finally:
        state.disarm()
        for _ in range(3):
            result = loco.SetVelocity(0., 0., 0., .15)
            stop_results.append(result)
            time.sleep(.05)
        print(json.dumps({'velocity_replies': stream.report(), 'stop_return_codes': stop_results,
                          'stop_acknowledged': any(code == 0 for code in stop_results)}), flush=True)
        stream.close()
        if command_error is not None or any(stop_results):
            record = {'time_unix': time.time(), 'command_error': command_error,
                      'stop_return_codes': stop_results}
            print(json.dumps(record), flush=True)
            try:
                record_dir = results_dir()
                record_dir.mkdir(parents=True, exist_ok=True)
                (record_dir / 'last_motion_error.json').write_text(json.dumps(record, indent=2))
            except OSError as error:
                print('Could not save error outside project: ' + str(error), flush=True)
    if not any(code == 0 for code in stop_results):
        raise RuntimeError('Stop was not acknowledged. Use the verified physical remote stop procedure.')
    if distance is not None:
        print(json.dumps({'distance_target_reached_within_0_10_m': reached,
                          'measured_forward_progress_before_stop_m': progress}), flush=True)
        if not reached:
            raise RuntimeError('Distance target not reached before time limit; stopped, no automatic retry')
    print('Trial ended. Stop commands sent. Distance above is commanded distance, not measured travel.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'camera', 'remote', 'walk', 'stop'], nargs='?', default='check')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--speed', type=float, default=.1)
    parser.add_argument('--distance', type=float, help='optional measured forward distance, in meters')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= MAX_DURATION_S or not math.isfinite(args.speed) or not 0 < args.speed <= MAX_SPEED_M_S:
        parser.error('walking limits: 0 < seconds <= 10; 0 < speed <= 0.3 m/s')
    if args.distance is not None and (not math.isfinite(args.distance) or not .15 <= args.distance <= 1.5):
        parser.error('distance must be within 0.15..1.5 m')
    if args.command == 'camera':
        print(json.dumps(camera_report(), indent=2))
        return
    config = json.loads((ROOT / 'config.json').read_text())
    if args.command == 'stop':
        STOP.touch()
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, config['interface'])
        loco = client()
        results = []
        for _ in range(3):
            results.append(loco.SetVelocity(0., 0., 0., .15))
            time.sleep(.05)
        print(json.dumps({'stop_return_codes': results}), flush=True)
        close_client(loco)
        if not any(code == 0 for code in results):
            raise RuntimeError('Stop was not acknowledged. Use the verified physical remote stop procedure.')
        print('Stop requested and zero velocity sent. Use the physical remote if communication is lost.')
        return
    guard = open('/tmp/etpr1-walk.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = State(config['interface'])
    loco = None
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        state.wait()
        if args.command == 'remote':
            state.monitor_remote()
        elif args.command == 'walk':
            if args.execute and STOP.exists():
                raise RuntimeError('Stop latch active. After checking the robot, remove /tmp/etpr1-stop to re-arm.')
            loco = client()
            walk(state, args.seconds, args.speed, args.execute, args.distance, loco=loco)
        else:
            loco = client()
            code, mode = read_query(loco.GetFsmId)
            report = {**state.report(), 'fsm_query_code': code, 'fsm': mode,
                      **camera_report(), 'motion_sent': False}
            status = locomotion_report(loco) if code == 0 else None
            report['locomotion_queries'] = status
            blockers = walking_blockers(state, code, mode, status)
            report['walk_ready'] = not blockers
            report['walk_blockers'] = blockers
            if code != 0:
                report['controller_diagnostics'] = controller_report()
            try:
                from model_options import service_health
                report['model_server'] = service_health(ROOT, config.get('default_model', 'st2nav'))
            except Exception as error:
                report['model_server'] = str(error)
            print(json.dumps(report, indent=2))
            if blockers:
                raise SystemExit(1)
    finally:
        close_client(loco)
        state.close()
        guard.close()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or 'Interrupted; stop requested')
