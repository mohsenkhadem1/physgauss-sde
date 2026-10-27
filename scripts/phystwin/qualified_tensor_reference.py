"""Candidate edge-refined Q2 reference; kept separate until development validation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.tensor_fixture import TensorFixture
from scripts.phystwin.consistent_fixture import query
from scripts.phystwin.qualified_reference import BASE,ACTIONS,protocol,write_json,release,solve,checked_compare,sample81,nonrigid
from scripts.phystwin.qualify_fixture import metrics

LEVELS=[6,10,14,18,22]


def build(n,r,device,order=3):
    import fcntl
    lock=None
    if n>=20:
        lock=open(BASE/'large-mesh.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:m=TensorFixture(n,r['E'],r['K'],r['layout'],order=order,device=device,x_power=2.,y_grading=True,z_refinement=4,memory_limit_bytes=80*1024**3)
    except BaseException:
        if lock is not None:lock.close()
        raise
    m._large_mesh_lock=lock
    return m


def run(name,r,device='cuda:0'):
    spec=protocol();directory=BASE/'tensor-refinements'/name;directory.mkdir(parents=True,exist_ok=True)
    audit=BASE/'tensor-qualification'/f'{name}.json';final=BASE/'tensor-fields'/f'{name}-qualified.npz'
    if final.exists() and audit.exists():
        assert json.loads(audit.read_text())['qualified'];return
    old=[];previous=None;old_n=None;streak=0;checks=[]
    for n in LEVELS:
        release(previous);print('TENSOR_MESH',name,n,flush=True);m=build(n,r,device);rows=[]
        for j,action in enumerate(ACTIONS):
            path=directory/f'n{n}-q3-{action}.npz'
            if path.exists():
                with np.load(path) as z:row={k:z[k] for k in z.files}
            else:
                start=time.time();initial=query(previous,old[j]['u'],m.rest) if old else None
                state=solve(m,action,initial);u=state['displacement_local_m'];row=dict(u=u,elapsed_seconds=time.time()-start,**metrics(m,u,state))
                tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(path)
                print('TENSOR_LOAD',name,n,action,round(time.time()-start,2),flush=True)
            rows.append(row)
        if old:
            paired=[checked_compare(a,b,spec,r,action) for a,b,action in zip(old,rows,ACTIONS)]
            passed=all(c['pass'] for c in paired);streak=streak+1 if passed else 0
            checks.append(dict(coarse=old_n,fine=n,actions=paired,passed=passed));print('TENSOR_SPATIAL',name,old_n,n,passed,streak,flush=True)
        old,previous,old_n=rows,m,n
        write_json(audit,dict(case=r,space='Q2_edge',qualified=False,latest_mesh=n,spatial=checks))
        if streak>=2:break
    if streak<2:release(m);raise RuntimeError(f'{name}: tensor spatial checks failed through {n}')
    release(previous);previous=None;m=build(n,r,device,4);quad=[]
    for j,action in enumerate(ACTIONS):
        path=directory/f'n{n}-q4-{action}.npz'
        if path.exists():
            with np.load(path) as z:row={k:z[k] for k in z.files}
        else:
            start=time.time();state=solve(m,action,old[j]['u']);u=state['displacement_local_m'];row=dict(u=u,elapsed_seconds=time.time()-start,**metrics(m,u,state))
            tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(path)
        quad.append(checked_compare(old[j],row,spec,r,action,True));print('TENSOR_QUADRATURE',name,n,action,quad[-1]['pass'],flush=True)
    qualified=all(c['pass'] for c in quad)
    write_json(audit,dict(case=r,space='Q2_edge',qualified=qualified,latest_mesh=n,spatial=checks,quadrature=quad))
    if not qualified:release(m);raise RuntimeError(f'{name}: tensor quadrature checks failed')
    final.parent.mkdir(parents=True,exist_ok=True);tmp=final.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,fields=np.stack([sample81(a['sampled_u']) for a in old]),nonrigid=np.array([nonrigid(a['sampled_u']) for a in old]),parameters=[r['E'],r['K']],layout=r['layout'],mesh_n=n,reference_space='Q2_edge');tmp.replace(final)
    release(m);print('TENSOR_QUALIFIED',name,n,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration','test'],default='development');p.add_argument('--index',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(f'{a.split}-{a.index}',protocol()[a.split][a.index],a.device)

if __name__=='__main__':main()
