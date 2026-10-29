"""Resumable inverse comparisons; fit only the acquired prefix, score separately."""
from pathlib import Path
import argparse,fcntl,json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,protocol,within_deadline
from scripts.position_ood.observations import make
from scripts.position_ood.inverse import fit,physical
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write

def run(split,i,method,mesh=2,iterations=100,prefix=(0,1,2)):
    tag=f'p{len(prefix)}'+('' if tuple(prefix)==(0,1,2) else '-'+('-'.join(map(str,prefix))))
    destination=BASE/f'inverse/{split}-{i}/{method}-n{mesh}-{tag}.json'
    destination.parent.mkdir(parents=True,exist_ok=True);lock=destination.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if destination.exists() and json.loads(destination.read_text()).get('state')=='complete':return
    within_deadline()
    if split.startswith('test'):
        from scripts.position_ood.seal import verify
        verify()
    obs=np.load(make(split,i));ids=obs['visible_ids'];visible=np.zeros(81,bool);visible[ids]=True
    acquired=np.array(prefix);observed=np.zeros((len(prefix),81,3));observed[:,ids]=obs['measurements'][acquired];poses=obs['poses'][acquired]
    env=method!='fixed_spatial_field';flex=env;frame='material' if method=='material_frame' else 'world'
    model=DensePositionFEM(mesh,field_frame=frame);results=[]
    for layout in range(3):
        cache=destination.with_name(destination.stem+f'-layout{layout}.json')
        if cache.exists():row=json.loads(cache.read_text())
        else:
            initial=None
            if len(prefix)>3:
                previous=destination.with_name(f'{method}-n{mesh}-p3-layout{layout}.json')
                if previous.exists():initial=json.loads(previous.read_text())['z']
            row=fit(model,poses,observed,visible,layout,environment=env,iterations=iterations,material_field=True,flexible_field=flex,initial=initial);write(cache,row)
        results.append(row);print('INVERSE',split,i,method,layout,row['loss'],flush=True)
    best=min(results,key=lambda row:row['loss']);sample_ids=np.argmin(np.linalg.norm(model.geo.rest[:,None]-points()[None],axis=-1),axis=0)
    with torch.no_grad():predicted=np.array([model.predict(model.tensor(best['theta']),model.tensor(pose),best['layout'])[sample_ids].cpu().numpy() for pose in obs['poses']])
    # The fit above never opens the held-out reference target array. Predictions
    # and immutable fit metadata are stored before the separate scoring phase.
    np.savez_compressed(destination.with_suffix('.npz'),predicted=predicted,poses=obs['poses'])
    write(destination,dict(state='complete',split=split,object=i,method=method,mesh=mesh,prefix=list(prefix),rows=results,best=best,input_scope='Rendered visible-point positions and prescribed tool poses only. No material, gravity, field or reaction-force labels.',time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--method',choices=['joint','fixed_spatial_field','material_frame'],default='joint');p.add_argument('--iterations',type=int,default=100);p.add_argument('--mesh',type=int,default=2);a=p.parse_args();s=protocol()
    if a.split.startswith('test') and not (BASE/'method-seal.json').exists():raise RuntimeError('Freeze methods before test inference')
    status=BASE/f'inverse-{a.split}-{a.method}-{a.worker}-status.json';write(status,dict(state='running',time=time.time()))
    try:
        for i in range(a.worker,len(s[a.split]),a.workers):
            while not (BASE/f'fields/{a.split}-{i}.npz').exists():
                within_deadline();q=BASE/f'qualification/{a.split}-{i}.json'
                if q.exists() and not json.loads(q.read_text())['qualified']:raise RuntimeError(f'Unqualified reference {a.split}-{i}')
                time.sleep(10)
            run(a.split,i,a.method,a.mesh,a.iterations)
        write(status,dict(state='complete',time=time.time()))
    except BaseException as e:write(status,dict(state='failed',time=time.time(),error=repr(e)));raise
if __name__=='__main__':main()
