"""Offline package checks, without weights, downloads, or robot access."""
import ast, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runtime'))
def main():
    import numpy as np
    import torch
    from complete_ranges import complete_ranges
    from g1prep.waypoint_input import ordered_views, circular_waypoint_nms
    import stitch_panorama
    from g1prep.planner import habitat_pose, robot_point
    for p in ROOT.parent.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    for rel,h in json.loads((ROOT/'source_manifest.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    d=np.full((32,64),3.,dtype=np.float32);d[:,5:12]=0;d[:,50:]=0;d[0,0]=12.
    measured=d>0;filled,estimated,_=complete_ranges(d,measured)
    assert np.array_equal(filled[measured],d[measured])
    assert np.isfinite(filled).all() and (filled>0).all() and (filled!=100).all()
    assert not estimated[measured].any()
    try:complete_ranges(np.zeros((4,4),np.float32),np.zeros((4,4),bool))
    except ValueError:pass
    else:raise AssertionError('No-support input must fail')
    obs={}
    for i in reversed(range(12)):
        suffix='' if i==0 else '_'+str(i*30)
        obs['depth'+suffix]=np.full((1,256,256,1),i/12,np.float32)
        obs['rgb'+suffix]=np.full((1,224,224,3),i,np.uint8)
    ordered=ordered_views(obs)
    assert list(ordered)[:4]==['rgb','depth','rgb_30','depth_30']
    heat=torch.zeros(1,1,122,12);heat[0,0,1,4]=.9;heat[0,0,120,4]=.8;heat[0,0,61,4]=.7
    peaks=circular_waypoint_nms(heat)
    assert peaks[0,0,1,4]==.9 and peaks[0,0,120,4]==0 and peaks[0,0,61,4]==.7
    position,_=habitat_pose([2,3,.5]);assert np.allclose(robot_point(position),[2,3])
    print('PASS: source integrity, syntax, depth preservation/completion, view ordering, circular peaks, coordinate round-trip; no robot access.')
if __name__=='__main__':main()
