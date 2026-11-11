"""One saved observation -> one model decision. No robot imports or network."""
import argparse, json, math, os, sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--instruction',required=True)
    p.add_argument('--pose',type=float,nargs=3,default=[0,0,0],metavar=('X','Y','YAW_RAD'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import numpy as np
    import torch
    from g1prep.model import OfficialPolicy
    from g1prep.planner import ReplayPlanner
    from g1prep.waypoint_input import ordered_views
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    with np.load(a.input,allow_pickle=False) as data:
        obs=ordered_views({k:data[k] for k in data.files})
    with torch.inference_mode():
        policy=OfficialPolicy(device=a.device)
        result=ReplayPlanner(policy,a.instruction).plan(obs,a.pose)
    relative=[]
    for x,y in result.get('path_xy_m',[]):
        dx,dy=x-a.pose[0],y-a.pose[1];yaw=a.pose[2]
        forward=dx*math.cos(yaw)+dy*math.sin(yaw)
        left=-dx*math.sin(yaw)+dy*math.cos(yaw)
        relative.append({'forward_m':forward,'left_m':left,'left_deg':math.degrees(math.atan2(left,forward)),'distance_m':math.hypot(dx,dy)})
    report={'model':policy.model_id,'instruction':a.instruction,'pose':a.pose,'model_output':result,'relative_to_input_pose':relative,
            'offline_replay':True,'robot_connected':False,'weights_modified':False,'loaded_parameters':policy.loaded_keys,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'model':policy.model_id,'status':result['status'],'relative_to_input_pose':relative,'robot_connected':False}))
if __name__=='__main__':main()
