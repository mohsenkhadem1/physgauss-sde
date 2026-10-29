"""Development-only position-control reference qualification, checkpointed by action."""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.fixture import PositionFixture,controls
from scripts.phystwin.sequential_reference import write
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/position-ood-v09';OUT=ROOT/'reports/position-ood-v09'
ACTIONS=[('rest',.006),('translate_y',.006),('translate_z',.006),('translate_y',.003),('stretch',.006),('opposed_y',.006),('twist_x',.006),('bend_z',.006),('diagonal',.006),('translate_y',.012)]
CASES=[dict(E=8000,Kleft=1500,Kright=12000,layout='crossed',gravity=[0,0,-9.81],field_amplitude_J=.01),dict(E=20000,Kleft=5000,Kright=8000,layout='parallel_lower',gravity=[0,-3.355,-9.218],field_amplitude_J=.02),dict(E=40000,Kleft=15000,Kright=15000,layout='opposite_y',gravity=[0,0,-9.81],field_amplitude_J=-.02)]
def level(i,n,order=3):
    source=hashlib.sha256((ROOT/'scripts/position_ood/fixture.py').read_bytes()).hexdigest()
    spec=dict(case=CASES[i],actions=ACTIONS,n=n,order=order,source=source);signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    dest=BASE/f'pilot-{i}/n{n}-q{order}.npz'
    if dest.exists():
        with np.load(dest) as z:
            assert str(z['signature'])==signature,'Stale physics cache';return {k:z[k] for k in z.files}
    model=PositionFixture(n,**CASES[i],order=order);rows=[];started=time.time()
    try:
        for j,(action,amp) in enumerate(ACTIONS):
            path=BASE/f'pilot-{i}/checkpoints/n{n}-q{order}-{j}.npz'
            if path.exists():
                with np.load(path) as z:
                    assert str(z['signature'])==signature,'Stale physics checkpoint';row={k:z[k] for k in z.files if k!='signature'}
            else:
                model.set_controls(controls(action,amp));state=model.solve(force_tolerance_N=1e-9);row=model.sampled(state)
                assert row['residual_N']<=1e-9 and row['reaction_balance_N']<=1e-7 and row['minimum_J']>.1
                path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,signature=signature,**row);temp.replace(path)
            rows.append(row);print('ACTION',i,n,order,j,round(float(np.sqrt(np.mean(row['field']**2)))*1000,3),flush=True)
        data={k:np.stack([r[k] for r in rows]) for k in rows[0]};data.update(signature=signature,elapsed_seconds=time.time()-started,poses=np.array([controls(a,b) for a,b in ACTIONS]))
        temp=dest.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(dest);return data
    finally:
        model.clear_linear_cache();model.engine=None;del model;gc.collect()
        import torch;torch.cuda.empty_cache()
def comparison(a,b):
    delta=a['field']-b['field'];rms=np.sqrt(np.mean(delta*delta,axis=(1,2)));maximum=abs(delta).max((1,2))
    return dict(rms_m=rms.tolist(),maximum_m=maximum.tolist(),passed=bool(rms.max()<2e-5 and maximum.max()<1e-4))
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);a=p.parse_args();status=BASE/f'pilot-{a.case}-status.json';write(status,dict(state='running',time=time.time()))
    previous=None;checks=[]
    try:
        for n in [2,4,6,8,10,12,14,16,18,20]:
            current=level(a.case,n)
            if previous is not None:
                check=comparison(previous,current);checks.append(dict(n=n,**check));print('REFINE',a.case,n,check,flush=True)
                if check['passed']:
                    quad=level(a.case,n,4);qc=comparison(current,quad)
                    if qc['passed']:
                        write(OUT/f'pilot-{a.case}-qualification.json',dict(case=CASES[a.case],actions=ACTIONS,mesh=n,spatial=checks,quadrature=qc,qualified=True));write(status,dict(state='complete',time=time.time(),mesh=n));return
            previous=current
        raise RuntimeError('Pilot reference refinement unresolved')
    except BaseException as e:
        write(status,dict(state='failed',time=time.time(),error=repr(e)));raise
if __name__=='__main__':main()
