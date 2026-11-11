"""Send one complete recorded real scan to the inference server. Prints a proposal; never walks."""
import argparse
import io
import json
from pathlib import Path
from storage import results_dir
import ssl
import sys
import urllib.error
import urllib.request
import time
from model_options import MODELS, model_id, model_url, service_health

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'runtime'))


def packet(directory, allow_synthetic=False):
    import numpy as np
    from g1prep.full_scan import Survey30
    from g1prep.sequence_io import read_sequence
    metadata = json.loads((directory / 'sequence.json').read_text())
    if metadata.get('kind') != 'actual_12_view_scan_sequence':
        raise ValueError('requires a measured 12-view scan recording')
    if metadata.get('synthetic') and not allow_synthetic:
        raise ValueError('artificial input refused; use --allow-synthetic only for software checks')
    survey, frames = Survey30(), []
    for value, _ in read_sequence(directory):
        prior, index = survey.stable_frame, survey.index
        survey.observe(value)
        if survey.index != index:
            frames.extend([prior, value])
    survey.observations(value.stamp)
    arrays = {'count': np.array(len(frames))}
    for i, frame in enumerate(frames):
        arrays.update({f'rgb{i}': frame.rgb, f'depth{i}': frame.depth_m,
                       f'intrinsics{i}': np.array(frame.intrinsics), f'camera{i}': frame.world_camera,
                       f'pose{i}': np.array(frame.base_pose),
                       f'times{i}': np.array([frame.stamp, frame.depth_stamp, frame.pose_stamp])})
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def native_packet(directory):
    import numpy as np
    metadata = json.loads((directory / 'scan.json').read_text())
    if (metadata.get('kind') != 'raw_measured_12_rgbd_views' or not metadata.get('complete')
            or len(metadata.get('views', [])) != 12 or metadata.get('synthetic')):
        raise ValueError('Requires twelve completed real captures')
    arrays = {'count': np.array(12), 'metadata': np.array(json.dumps(metadata))}
    for i, view in enumerate(metadata['views']):
        if view['file'] != 'view_%03d.npz' % (i * 30):
            raise ValueError('Invalid capture filename')
        with np.load(directory / view['file'], allow_pickle=False) as data:
            arrays[f'rgb{i}'], arrays[f'depth{i}'] = data['rgb'], data['depth_m']
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def dual_packet(directory):
    import numpy as np
    metadata=json.loads((directory/'scan.json').read_text())
    views=metadata.get('views',[])
    if (metadata.get('kind')!='dual_camera_measured_circle' or not metadata.get('complete')
            or len(views)!=24 or metadata.get('synthetic') or metadata.get('stop_error')
            or not metadata.get('final_displacement',{}).get('stationary_by_sensors')):
        raise ValueError('Requires a completed, stopped, real 24-frame dual-camera scan')
    arrays={'count':np.array(24),'metadata':np.array(json.dumps(metadata,allow_nan=False))}
    for i,view in enumerate(views):
        expected='view_%03d_%s.npz'%(view['index']*30,view['serial'])
        if view['file']!=expected or Path(expected).name!=expected:
            raise ValueError('Invalid dual-camera capture filename')
        with np.load(directory/expected,allow_pickle=False) as data:
            rgb,depth=data['rgb'],data['depth_m']
            if rgb.shape!=(720,1280,3) or rgb.dtype!=np.uint8 or depth.shape!=(720,1280):
                raise ValueError('Dual-camera stream profile mismatch')
            arrays['rgb'+str(i)],arrays['depth'+str(i)]=rgb,depth.astype(np.float32)
    output=io.BytesIO();np.savez_compressed(output,**arrays)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--new-task', action='store_true')
    parser.add_argument('--execution-feedback', type=Path)
    parser.add_argument('--model', type=model_id, choices=MODELS, default='st2nav')
    parser.add_argument('--allow-synthetic', action='store_true')
    parser.add_argument('--native-camera', action='store_true', help='explicit supervised native-lens experiment')
    parser.add_argument('--dual-camera',action='store_true',help='24 real frames, shared panorama, twelve 90-degree inputs')
    args = parser.parse_args()
    service_health(ROOT, args.model)
    if args.native_camera and args.allow_synthetic:
        parser.error('Native camera experiment requires real input')
    if args.dual_camera and (args.native_camera or args.allow_synthetic):
        parser.error('Dual-camera mode requires real dual-camera input only')
    body = None if args.execution_feedback else (dual_packet(args.directory) if args.dual_camera else native_packet(args.directory)
            if args.native_camera else packet(args.directory,args.allow_synthetic))
    base = Path.home() / '.local/share/etpr1'
    token = (base / 'token').read_text().strip()
    url = model_url(json.loads((ROOT / 'config.json').read_text())['model_url'], args.model)
    context = ssl.create_default_context(cafile=str(base / 'server.crt'))
    def post(path, payload, session=None):
        headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/octet-stream',
                   'X-ETP-Model': args.model}
        if session:
            headers['X-ETP-Session'] = session
        request = urllib.request.Request(url + path, payload, headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=480 if args.dual_camera else 90, context=context) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error
    progress = base / ('session-' + args.model + '.json')
    task = json.loads(progress.read_text()) if progress.exists() else {}
    if args.execution_feedback:
        if args.new_task or task.get('instruction') != args.instruction or task.get('model_id') != args.model:
            raise ValueError('Feedback requires the matching existing model session')
        result = post('/execution-feedback', args.execution_feedback.read_bytes(), task['session'])
        if result.get('model_id') != args.model or result.get('accepted') is not True:
            raise RuntimeError('Execution feedback was not acknowledged')
        print(json.dumps({'execution_feedback': result}))
        return
    if args.new_task or task.get('instruction') != args.instruction:
        task = {**post('/reset', json.dumps({'instruction': args.instruction}).encode()),
                'instruction': args.instruction}
        if task.get('model_id') != args.model:
            raise RuntimeError('Model identity mismatch while starting task')
        progress.write_text(json.dumps(task))
    result = post('/plan-dual' if args.dual_camera else '/plan-measured' if args.native_camera else '/plan',body,task['session'])
    if result.get('model_id') != args.model:
        raise RuntimeError('Model identity mismatch in returned plan')
    if not args.native_camera and not args.dual_camera:
        metadata = json.loads((args.directory / 'sequence.json').read_text())
        result['synthetic_input'] = metadata['synthetic']
    result['scope'] = 'planning only; no robot motion'
    output = results_dir() / 'plans' / str(time.time_ns())
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({**result, 'saved_result': str(output / 'result.json')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))
