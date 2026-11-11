"""Supervised continuous navigation. Child processes own bounded motion."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
from storage import results_dir, configure_results
import signal
import subprocess
import time
from model_options import MODELS, model_id, service_health

ROOT = Path(__file__).resolve().parent
STOP = Path('/tmp/etpr1-stop')


def json_messages(text):
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        start = text.find('{', offset)
        if start < 0:
            return
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            offset = start + 1
            continue
        yield value
        offset = start + length


def completed_before_native_exit(code, key, messages):
    """Only a recorded stopped completion may survive a native shutdown fault."""
    if code not in (-6, -7, -11, 134, 135, 139):
        return False
    values = [v for v in messages if key in v]
    if not values:
        return False
    final = values[-1]
    result = final.get('step_result') if key == 'step_result' else final
    if not isinstance(result, dict) or not result.get('complete'):
        return False
    if key == 'step_result' and not result.get('movement_started'):
        return False
    if key != 'step_result' and (key != 'capture_complete' or not result.get(key)):
        return False
    if not result.get('final_displacement', {}).get('stationary_by_sensors'):
        return False
    stops = [v for v in messages if 'stop_return_codes' in v]
    return bool(stops and 0 in stops[-1]['stop_return_codes'])


class StageFailure(RuntimeError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class Runner:
    def __init__(self, directory, state):
        self.directory, self.state, self.count = directory, state, 0

    def pose(self):
        import scan
        self.state.subscribe_position(True)
        deadline = time.monotonic() + 3.
        while True:
            self.state.permit()
            with self.state.lock:
                if self.state.sport is not None and time.monotonic() - self.state.sport_time < .1:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError('No fresh position after phase handoff')
            time.sleep(.02)
        pose = scan.Turner(self.state, None, None, .3).read()
        return [*pose['position_m'][:2], pose['rpy_rad'][2]]

    def run(self, arguments, key, timeout):
        self.state.permit()
        self.state.subscribe_position(False)
        self.count += 1
        log = self.directory / ('%03d.log' % self.count)
        with log.open('w') as output:
            child = subprocess.Popen(arguments, cwd=ROOT, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + timeout
                while child.poll() is None:
                    self.state.permit()
                    if time.monotonic() > deadline:
                        raise RuntimeError('Stage timed out: ' + arguments[0])
                    time.sleep(.1)
                self.state.permit()
            except BaseException:
                STOP.touch()
                if child.poll() is None:
                    import os
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=3)
                raise
        content = log.read_text(errors='replace')
        messages = list(json_messages(content))
        with log.open('a') as output:
            output.write('\n' + json.dumps({'stage_exit_code': child.returncode}) + '\n')
        if child.returncode:
            if completed_before_native_exit(child.returncode, key, messages):
                self.state.permit()
                self.stable()
                print('上一段已完成并重新确认停稳；已记录退出异常，不重复运动。', flush=True)
            else:
                values = [v for v in messages if key in v]
                raise StageFailure('Stage failed (exit=%s); no automatic retry. %s\n%s' %
                                   (child.returncode, log, content[-1800:]),
                                   values[-1] if values else {})
        values = [v for v in messages if key in v]
        if not values:
            raise RuntimeError('Missing stage result: ' + str(log))
        return values[-1]

    def stable(self):
        import scan
        current = self.pose()
        pose = {'position_m': current[:2], 'rpy_rad': [0, 0, current[2]]}
        result = scan.wait_after_stop(self.state, pose)
        if not result['stationary_by_sensors']:
            raise RuntimeError('Robot did not settle; no further motion')
        return result


def validate_plan(plan):
    output = plan['model_output']
    if plan.get('motion_dispatch') is not False:
        raise ValueError('Expected planning-only response')
    prior = plan.get('observation', {}).get('reference_pose_xy_yaw')
    if (not isinstance(prior, list) or len(prior) != 3
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in prior)):
        raise ValueError('Invalid observation pose')
    if output['status'] not in ('stop', 'return_then_stop', 'path_proposal'):
        raise RuntimeError('Model did not provide an executable route: ' + output['status'])
    path = output.get('path_xy_m', [])
    if (not isinstance(path, list) or len(path) > 30
            or any(not isinstance(p, list) or len(p) != 2
                   or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not math.isfinite(v) for v in p) for p in path)):
        raise ValueError('Invalid model route')
    if output['status'] == 'stop' and path:
        raise ValueError('Stop response must not contain a route')
    if output['status'] != 'stop' and not path:
        raise ValueError('Empty movement route')
    return output, path


def face_waypoint(runner, waypoint):
    for turns in range(6):
        pose = runner.pose()
        dx, dy = waypoint[0] - pose[0], waypoint[1] - pose[1]
        angle = (math.degrees(math.atan2(dy, dx) - pose[2]) + 180) % 360 - 180
        # Avoid chasing centimetre-scale turn drift; step independently enforces
        # 10 degrees after its fresh image. Keep 2 degrees of margin here.
        if math.hypot(dx, dy) < .15 or abs(angle) <= 8:
            return pose
        if turns == 5:
            raise RuntimeError('Could not face model waypoint after 5 turns: '
                               'remaining %.1f deg, distance %.2f m' % (angle, math.hypot(dx, dy)))
        result = runner.run(['./run.sh', 'turn', '--execute', '--degrees',
                             str(max(-60., min(60., angle))), '--rate', '.3'],
                            'capture_complete', 90)
        if not result.get('capture_complete'):
            raise RuntimeError('Incomplete turn; no walking attempted')
        runner.stable()


def relative_leg(pose, waypoint):
    dx, dy = waypoint[0] - pose[0], waypoint[1] - pose[1]
    distance = math.hypot(dx, dy)
    angle = (math.degrees(math.atan2(dy, dx) - pose[2]) + 180) % 360 - 180 if distance else 0.
    return {'turn_left_deg': angle, 'distance_m': distance}


def leg_text(leg):
    angle, distance = leg['turn_left_deg'], leg['distance_m']
    if distance < .15:
        return '目标已在脚边（约 %.0f 厘米），跳过这段' % (distance * 100)
    if abs(angle) <= 8:
        direction = '朝前' if abs(angle) < .5 else '朝前略偏%s约 %.0f°' % ('左' if angle > 0 else '右', abs(angle))
        return '%s，前进约 %.2f 米' % (direction, distance)
    return '向%s转约 %.0f°，再前进约 %.2f 米' % ('左' if angle > 0 else '右', abs(angle), distance)


def explain_route(pose, path):
    current = list(pose)
    legs = []
    for waypoint in path:
        leg = relative_leg(current, waypoint)
        legs.append(leg)
        # A skipped nearby waypoint does not rotate or relocate the robot.
        if leg['distance_m'] >= .15:
            current = [*waypoint, math.atan2(waypoint[1] - current[1], waypoint[0] - current[0])]
    return legs


def execute_route(runner, source, plan, directory, event, progress=None):
    progress = progress if progress is not None else {}
    progress['completed_waypoints'] = 0
    output, path = validate_plan(plan)
    initial = runner.pose()
    prior = plan['observation']['reference_pose_xy_yaw']
    if math.hypot(initial[0] - prior[0], initial[1] - prior[1]) > .30:
        settled = runner.stable()
        event({'phase': 'observation_position_changed', 'observation_pose': prior,
               'current_pose': initial, 'settled': settled,
               'next_action': 'fresh_scan_and_model_decision'})
        print('推理等待期间位置读数变化超过30厘米；保持停止，重新观察并让模型规划。', flush=True)
        return False
    legs = explain_route(initial, path)
    print('本轮路线（从现在的位置和朝向算起）：', flush=True)
    for index, leg in enumerate(legs):
        print('  第 %d 段：%s%s。' % (index + 1, leg_text(leg),
              '（按上一段到达后的朝向估算）' if index else ''), flush=True)
    if output['status'] == 'return_then_stop':
        print('模型要求走完这条返回路线后结束。', flush=True)
    event({'phase': 'route_explanation', 'measured_pose_xy_yaw': initial, 'legs': legs,
           'later_legs_assume_exact_arrival': True})
    segments = 0
    for index, waypoint in enumerate(path):
        # Intermediate execution files never replace the raw model result.
        while True:
            pose = runner.pose()
            distance = math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1])
            if distance < .15:
                break
            if segments >= 40:
                raise RuntimeError('Execution segment limit reached')
            print('准备执行第 %d 段：%s（按此刻实测位置和朝向重算）。' %
                  (index + 1, leg_text(relative_leg(pose, waypoint))), flush=True)
            pose = face_waypoint(runner, waypoint)
            distance = math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1])
            if distance < .15:
                break
            fraction = min(1., 1.25 / distance)
            print('转向完成，这一小段准备前进约 %.2f 米。' % (distance * fraction), flush=True)
            target = [pose[i] + fraction * (waypoint[i] - pose[i]) for i in range(2)]
            execution = copy.deepcopy(plan)
            execution['model_output'] = {**output, 'status': 'path_proposal', 'path_xy_m': [target]}
            execution['observation']['reference_pose_xy_yaw'] = pose
            execution['execution_adapter'] = {
                'source_model_result': str(source),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'original_model_output': output, 'waypoint_index': index,
                'reference_reason': 'fresh measured start of this bounded route segment',
                'is_raw_model_output': False}
            segments += 1
            file = directory / ('decision-%02d-leg-%02d.json' % (output['step'], segments))
            file.write_text(json.dumps(execution, indent=2, allow_nan=False))
            try:
                result = runner.run(['./run.sh', 'step', str(file), '--human-clear-path', '--execute'],
                                    'step_result', 70)
            except StageFailure as error:
                stopped = error.result.get('step_result', {})
                if (stopped.get('complete') is False and stopped.get('movement_started') is False
                        and stopped.get('error') == 'A fresh measured obstacle blocks the proposed short leg'):
                    progress['blocked_by_measured_obstacle'] = True
                    settled = runner.stable()
                    event({'phase': 'route_blocked_before_motion', 'result': error.result,
                           'settled': settled, 'next_action': 'fresh_scan_and_model_decision'})
                    print('前方实测障碍挡住本段，尚未前进；保持停止，重新观察并让模型规划。', flush=True)
                    return False
                raise
            if not result['step_result'].get('complete'):
                raise RuntimeError('Incomplete walking segment')
            settled = runner.stable()
            event({'phase': 'walk_complete', 'source': str(source), 'waypoint_index': index,
                   'execution_file': str(file), 'result': result, 'settled': settled})
            if fraction == 1.:
                # Next observation uses actual endpoint, not the planned position.
                break
            after = runner.pose()
            remaining = math.hypot(waypoint[0] - after[0], waypoint[1] - after[1])
            if remaining > distance - .1:
                raise RuntimeError('Insufficient progress along model route')
        progress['completed_waypoints'] = index + 1
    return True


def navigate(runner, instruction, directory, event, max_decisions=30, model='etpr1', start_decision=1):
    for decision in range(start_decision, max_decisions + 1):
        print('第 %d 轮：转圈采集双相机 24 张图。' % decision, flush=True)
        scan = runner.run(['./run.sh', 'dual-scan', '--execute'], 'capture_complete', 540)
        if not scan.get('capture_complete') or scan.get('views') != 24:
            raise RuntimeError('Incomplete scan; not sent to model')
        runner.stable()
        print('在推理电脑拼接成 12 张 90°画面，然后由模型决定下一步。', flush=True)
        args = ['./plan.sh', scan['directory'], '--instruction', instruction, '--dual-camera', '--model', model]
        if decision == 1:
            args.append('--new-task')
        result = runner.run(args, 'saved_result', 540)
        source = Path(result['saved_result'])
        plan = json.loads(source.read_text())
        if plan.get('model_id') != model:
            raise RuntimeError('Received a plan from a different model')
        output, _ = validate_plan(plan)
        event({'phase': 'model_decision', 'scan': scan['directory'], 'file': str(source),
               'model_output': output})
        print('模型决定：' + json.dumps(output, ensure_ascii=False), flush=True)
        if output['status'] in ('stop', 'return_then_stop') and output.get('selected_action_index') != 0:
            print('本轮没有可执行的继续路线，任务中止；这不表示已经到达目标。', flush=True)
            return 'policy_limit_or_no_candidates'
        if output['status'] != 'stop':
            progress = {}
            complete = execute_route(runner, source, plan, directory, event, progress)
            feedback = {'step': output['step'], 'route_complete': complete,
                        'completed_waypoints': progress['completed_waypoints'],
                        'blocked_by_measured_obstacle': progress.get('blocked_by_measured_obstacle', False),
                        'actual_pose_xy_yaw': runner.pose()}
            feedback_file = directory / ('feedback-%02d.json' % output['step'])
            feedback_file.write_text(json.dumps(feedback, allow_nan=False))
            runner.run(['./plan.sh', scan['directory'], '--instruction', instruction,
                        '--model', model, '--execution-feedback', str(feedback_file)],
                       'execution_feedback', 40)
            event({'phase': 'execution_feedback', **feedback})
            if complete is False:
                continue
        if output['status'] in ('stop', 'return_then_stop'):
            print('模型决定结束，保持停止。', flush=True)
            return 'model_stop'
    return 'decision_limit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--human-clear-path', action='store_true')
    parser.add_argument('--clear-stop', action='store_true', help='explicitly re-arm a NEW task')
    parser.add_argument('--instruction')
    parser.add_argument('--results-dir', type=Path, help='Persistent experiment root (overrides NAV_RESULTS_DIR); robot records go in its robot/ subdirectory')
    parser.add_argument('--resume-task', type=Path,
                        help='explicitly continue a stopped task with a NEW observation; never replay its route')
    parser.add_argument('--model', type=model_id, choices=MODELS, default='st2nav')
    args = parser.parse_args()
    configure_results(args.results_dir)
    if not args.execute or not args.human_clear_path:
        parser.error('Requires --execute --human-clear-path; no movement sent')
    instruction = args.instruction or input('输入导航指令（建议英文）：').strip()
    if not instruction.strip() or len(instruction) > 2000:
        parser.error('Instruction must contain 1..2000 characters')
    start_decision = 1
    if args.resume_task:
        previous = json.loads((args.resume_task / 'episode.json').read_text())
        if (previous.get('instruction') != instruction or previous.get('model_id') != args.model
                or previous.get('status') != 'interrupted_or_failed'
                or previous.get('final_stop', {}).get('returncode') != 0):
            raise ValueError('Resume requires the same interrupted task and an acknowledged final stop')
        steps = [v['model_output']['step'] for v in previous['events'] if v['phase'] == 'model_decision']
        if not steps or not 1 <= max(steps) < 30:
            raise ValueError('No resumable model decision before the task limit')
        session_file = Path.home() / ('.local/share/etpr1/session-' + args.model + '.json')
        session = json.loads(session_file.read_text())
        if session.get('instruction') != instruction or session.get('model_id') != args.model:
            raise ValueError('The matching model session is no longer available')
        start_decision = max(steps) + 1
    health = service_health(ROOT, args.model)
    if health.get('observation') != 'dual_24_rgbd_to_12_90deg':
        raise RuntimeError('Selected model has not loaded the dual-camera stitching pipeline')
    if health.get('execution_feedback') is not True:
        raise RuntimeError('Selected model service must reload execution-feedback support')
    print('本次使用模型：' + MODELS[args.model]['name'], flush=True)
    lock = open('/tmp/etpr1-navigation.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STOP.exists() and not args.clear_stop:
        raise RuntimeError('Stop latch active; use --clear-stop only after checking the robot')
    directory = results_dir() / 'episodes' / str(time.time_ns())
    directory.mkdir(parents=True)
    state = None
    report = {'instruction': instruction, 'model_id': args.model, 'status': 'starting', 'events': []}
    if args.resume_task:
        report.update(resumed_from=str(args.resume_task.resolve()), start_decision=start_decision)
    def event(value):
        report['events'].append(value)
        (directory / 'episode.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    def interrupted(*_):
        raise KeyboardInterrupt
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        if args.clear_stop:
            STOP.unlink(missing_ok=True)
        import robot
        state = robot.State(json.loads((ROOT / 'config.json').read_text())['interface'], position=False)
        state.wait()
        time.sleep(1.)
        state.arm()  # Keep manual-input interruption latched even while planning.
        runner = Runner(directory, state)
        check = runner.run(['./run.sh', 'check'], 'walk_ready', 60)
        if not check['walk_ready']:
            raise RuntimeError('Robot or model is not ready')
        runner.stable()
        report['status'] = navigate(runner, instruction, directory, event, model=args.model,
                                    start_decision=start_decision)
    except BaseException as error:
        report.update(status='interrupted_or_failed', error=str(error) or type(error).__name__)
        raise
    finally:
        STOP.touch()
        # All exits remain stopped and require explicit re-arming for another task.
        try:
            stop = subprocess.run(['./run.sh', 'stop'], cwd=ROOT, capture_output=True,
                                  text=True, timeout=8)
            report['final_stop'] = {'returncode': stop.returncode, 'output': stop.stdout + stop.stderr}
        except Exception as error:
            report['final_stop'] = {'error': str(error)}
        if state is not None:
            state.disarm()
            state.close()
        event({'phase': 'finished', 'status': report['status']})
        lock.close()
        print('运行结束：%s。记录：%s' % (report['status'], directory), flush=True)
        if report.get('final_stop', {}).get('returncode') != 0:
            raise RuntimeError('Final stop unconfirmed; use the verified physical remote')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt, BlockingIOError) as error:
        raise SystemExit(str(error) or 'Interrupted; stopped')
