"""Create positional RGB-D observations without exposing physical labels."""
from pathlib import Path
import sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,protocol
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.v06_camera import render,track

def make(split,i):
    path=BASE/f'observations/{split}-{i}.npz'
    if path.exists():return path
    ref=BASE/f'fields/{split}-{i}.npz'
    if not ref.exists():raise FileNotFoundError(ref)
    ids=np.flatnonzero(np.isclose(points()[:,1],-.04));seed={'development':9100,'calibration':9200,'test_id':9300,'test_material':9400,'test_environment':9500,'transfer_development':9100,'transfer_test':9300,'transfer_calibration':9200}[split]+i;rng=np.random.default_rng(seed)
    with np.load(ref) as z:
        observations=[]
        for field in z['field']:
            rgb,depth,camera=render(points()+field,0,ids);tracked=track(rgb,depth,camera,ids,rng);observations.append(tracked-points()[ids])
        poses=z['poses'].copy()
    if split.startswith('transfer_'):
        original={'transfer_development':'development','transfer_calibration':'calibration','transfer_test':'test_id'}[split]
        with np.load(make(original,i)) as previous:observations[:3]=list(previous['measurements'][:3])
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,measurements=np.array(observations),visible_ids=ids,poses=poses,seed=seed);temp.replace(path);return path
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');a=p.parse_args()
    for i in range(len(protocol()[a.split])):
        if (BASE/f'fields/{a.split}-{i}.npz').exists():print(make(a.split,i))
