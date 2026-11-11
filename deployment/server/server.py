"""Single-session ST2Nav planning service. No robot motion interface."""
import argparse
import hmac
import io
import json
import os
from pathlib import Path
from storage import results_dir, configure_results
import sys
import ssl
import time
import uuid
import zipfile
from model_options import MODELS, model_id
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'runtime'))
os.environ.setdefault('HF_HOME', str(Path.home() / '.cache/etpr1/huggingface'))
os.environ.setdefault('TRANSFORMERS_CACHE', str(Path.home() / '.cache/etpr1/huggingface/hub'))
os.environ.setdefault('TORCH_HOME', str(Path.home() / '.cache/etpr1/torch'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', type=model_id, choices=MODELS, default='st2nav')
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.local/share/etpr1')
    parser.add_argument('--results-dir', type=Path, help='Persistent experiment root (overrides NAV_RESULTS_DIR); server records go in its server/ subdirectory')
    args = parser.parse_args()
    configure_results(args.results_dir)
    args.port = args.port or MODELS[args.model]['port']
    os.environ['ETPR1_MODEL'] = args.model
    import numpy as np
    import cv2
    import torch
    from g1prep.model import OfficialPolicy
    from g1prep.planner import ReplayPlanner
    from g1prep.memory import RGBDFrame
    from g1prep.full_scan import Survey30
    from measured import prepare
    from stitch_panorama import reconstruct,validate_record,PIPELINE_REVISION
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    token = (args.state_dir / 'token').read_text().strip()
    if len(token) < 32:
        raise RuntimeError('Service token missing or too short')
    policy = OfficialPolicy(device=args.device)
    model_name = ('official ETP-R1 R2R iteration 270' if args.model == 'etpr1' else
                  MODELS[args.model]['name'] + ' R2R iteration ' + str(policy.checkpoint_iteration))
    current = {'session': None, 'planner': None, 'last_stamp': None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                self.reply(404, {'error': 'not found'})
                return
            self.reply(200, {'ready': True, 'model': model_name, 'model_id': args.model, 'device': str(policy.device),
                             'observation':'dual_24_rgbd_to_12_90deg', 'pipeline_revision':PIPELINE_REVISION,
                             'source_frames':24,'model_views':12,'motion_dispatch':False,
                             'execution_feedback':True,'language_padding':'tokenizer_pad_id',
                             'waypoint_postprocess_revision':policy.waypoint_postprocess_revision,
                             'graph_coordinate_revision':ReplayPlanner.coordinate_revision})

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.reply(401, {'error': 'unauthorized'})
                return
            if self.headers.get('X-ETP-Model', args.model) != args.model:
                self.reply(409, {'error': 'request sent to the wrong model service'})
                return
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError('body limit 64 MiB')
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('incomplete request')
                if self.path == '/reset':
                    instruction = json.loads(body)['instruction']
                    if not isinstance(instruction, str) or not 0 < len(instruction) <= 2000:
                        raise ValueError('instruction required')
                    with torch.inference_mode():
                        current['planner'] = ReplayPlanner(policy, instruction)
                    current['instruction'] = instruction
                    current['session'], current['last_stamp'] = str(uuid.uuid4()), None
                    self.reply(200, {'session': current['session'], 'model_id': args.model})
                    return
                if self.path not in ('/plan', '/plan-measured','/plan-dual'):
                    if self.path == '/execution-feedback':
                        if current['planner'] is None or self.headers.get('X-ETP-Session') != current['session']:
                            raise ValueError('reset this session first')
                        result = current['planner'].execution_feedback(json.loads(body))
                        self.reply(200, {**result, 'model_id': args.model})
                        return
                    self.reply(404, {'error': 'not found'})
                    return
                if current['planner'] is None or self.headers.get('X-ETP-Session') != current['session']:
                    self.reply(409, {'error': 'reset this session first'})
                    return
                with zipfile.ZipFile(io.BytesIO(body)) as archive:
                    if sum(f.file_size for f in archive.infolist()) > 192 * 1024 * 1024 or len(archive.infolist()) > 240:
                        raise ValueError('expanded scan exceeds limit')
                if self.path=='/plan-dual':
                    directory=results_dir()/'incoming-dual'/str(time.time_ns())
                    with np.load(io.BytesIO(body),allow_pickle=False) as data:
                        if int(data['count'])!=24:raise ValueError('24 actual RGBD frames required')
                        metadata=json.loads(str(data['metadata']));validate_record(metadata)
                        stamp=max(v['capture_monotonic_s'] for v in metadata['views'])
                        if current['last_stamp'] is not None and stamp<=current['last_stamp']:
                            raise ValueError('Previous scan cannot be reused in this session')
                        current['last_stamp']=stamp
                        directory.mkdir(parents=True)
                        for i,view in enumerate(metadata['views']):
                            rgb,depth=data['rgb'+str(i)],data['depth'+str(i)]
                            if (rgb.shape!=(720,1280,3) or rgb.dtype!=np.uint8
                                    or depth.shape!=(720,1280) or not np.isfinite(depth).all()):
                                raise ValueError('Invalid dual-camera RGBD profile')
                            np.savez_compressed(directory/view['file'],rgb=rgb,depth_m=depth)
                        (directory/'scan.json').write_text(json.dumps(metadata,allow_nan=False))
                    started=time.monotonic()
                    observations,pose,stamp,audit=reconstruct(directory,directory/'stitched',reference='end')
                    stitched_s=time.monotonic()-started
                    started=time.monotonic()
                    with torch.inference_mode():result=current['planner'].plan(observations,pose)
                    report={'model':model_name,'model_id':args.model,'model_output':result,
                        'observation':audit,'instruction':current['instruction'],'motion_dispatch':False,
                        'synthetic_input':False,'stitching_s':round(stitched_s,3),
                        'inference_s':round(time.monotonic()-started,3)}
                    (directory/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))
                    self.reply(200,report)
                    return
                if self.path == '/plan-measured':
                    directory = results_dir() / 'incoming' / str(time.time_ns())
                    directory.mkdir(parents=True)
                    with np.load(io.BytesIO(body), allow_pickle=False) as data:
                        if int(data['count']) != 12:
                            raise ValueError('Twelve actual RGBD captures required')
                        metadata = json.loads(str(data['metadata']))
                        if len(metadata.get('views', [])) != 12:
                            raise ValueError('Twelve capture records required')
                        for i, view in enumerate(metadata['views']):
                            if view['file'] != 'view_%03d.npz' % (i * 30):
                                raise ValueError('Invalid capture filename')
                            rgb, depth = data[f'rgb{i}'], data[f'depth{i}']
                            if rgb.ndim != 3 or max(rgb.shape[:2]) > 2048:
                                raise ValueError('Invalid image dimensions')
                            np.savez_compressed(directory / view['file'], rgb=rgb, depth_m=depth)
                        (directory / 'scan.json').write_text(json.dumps(metadata))
                    observations, memory, pose, stamp, audit = prepare(directory)
                    if current['last_stamp'] is not None and stamp <= current['last_stamp']:
                        raise ValueError('Previous scan cannot be reused in this session')
                    current['last_stamp'] = stamp
                    started = time.monotonic()
                    with torch.inference_mode():
                        result = current['planner'].plan(observations, pose)
                    report = {'model': model_name, 'model_id': args.model, 'model_output': result,
                              'observation': audit, 'instruction': current['instruction'],
                              'motion_dispatch': False, 'synthetic_input': False,
                              'inference_s': round(time.monotonic() - started, 3)}
                    (directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                    self.reply(200, report)
                    return
                survey = Survey30()
                with np.load(io.BytesIO(body), allow_pickle=False) as data:
                    count = int(data['count'])
                    if not 26 <= count <= 30:
                        raise ValueError('send 13 measured stationary headings, with two confirmations each')
                    for i in range(count):
                        frame = RGBDFrame(data[f'rgb{i}'], data[f'depth{i}'], tuple(data[f'intrinsics{i}']),
                                          data[f'camera{i}'], tuple(data[f'pose{i}']), *data[f'times{i}'])
                        if max(frame.rgb.shape[:2]) > 2048:
                            raise ValueError('image too large')
                        if current['last_stamp'] is not None and frame.stamp <= current['last_stamp']:
                            raise ValueError('previous scan cannot be reused')
                        survey.observe(frame)
                    observations = survey.observations(frame.stamp)
                    pose = frame.base_pose
                # Consume the scan even if inference fails. Never retry it as a new observation.
                current['last_stamp'] = frame.stamp
                started = time.monotonic()
                with torch.inference_mode():
                    result = current['planner'].plan(observations, pose)
                self.reply(200, {**result, 'actual_views': 12, 'history_reprojection': False,
                                 'model_id': args.model, 'model': model_name,
                                 'motion_dispatch': False, 'inference_s': round(time.monotonic() - started, 3)})
            except (ValueError, KeyError, RuntimeError, TypeError, OverflowError, zipfile.BadZipFile,cv2.error) as error:
                self.reply(400, {'error': str(error)})

    print(json.dumps({'ready': True, 'listen': [args.host, args.port], 'device': str(policy.device),
                      'loaded_parameters': policy.loaded_keys}), flush=True)
    server = HTTPServer((args.host, args.port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.state_dir/'server.crt', args.state_dir/'server.key')
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
