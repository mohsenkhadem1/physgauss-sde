"""Fresh position-controlled cohorts and resumable nonlinear references."""
from pathlib import Path
import argparse,datetime,fcntl,gc,hashlib,json,os,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.material_field import MaterialPositionFixture
from scripts.position_ood.fixture import controls
from scripts.position_ood.pilot import ACTIONS,comparison
from scripts.phystwin.sequential_fixture import points,LAYOUTS
from scripts.phystwin.tensor_fixture import basis
from scripts.phystwin.sequential_reference import write
from scripts.position_ood.runtime import ROOT,BASE,OUT,CONFIG,BOUNDARY

def within_deadline():
    s=json.loads(CONFIG.read_text());end=datetime.datetime.fromisoformat(s['campaign_deadline_utc'].replace('Z','+00:00')).timestamp()
    if time.time()>=end:raise RuntimeError('Campaign deadline reached')
def protocol():
    path=BASE/'cohorts.json'
    if path.exists():return json.loads(path.read_text())
    s=dict(version=10 if BOUNDARY=='smooth' else 9,boundary=BOUNDARY,status='development; freeze methods before test scoring',actions=ACTIONS,prefix_indices=[0,1,2],id_indices=[3],ood_indices=[4,5,6,7,8,9],reference_levels=[4,6,8,10,12,14,16,18,20,22,24],noise_sigma_m=.0003)
    for split,count,seed in [('development',9,290901),('calibration',12,290902),('test_id',18,290903),('test_material',12,290904),('test_environment',12,290905)]:
        rng=np.random.default_rng(seed);rows=[]
        for i in range(count):
            E=float(np.exp(rng.uniform(np.log(8000),np.log(40000))));K=np.exp(rng.uniform(np.log(1500),np.log(15000),2));h=float(rng.uniform(-.5,.5));tilt=rng.uniform(-.3,.3,2);g=np.r_[tilt,-1.];g*=9.81/np.linalg.norm(g)
            center=[.12,0,-.09];width=.08;A=float(rng.uniform(-.15,.15))
            if split=='test_material':E=float(rng.uniform(5500,7800) if i%2==0 else rng.uniform(42000,65000));h=float(rng.choice([-1,1])*rng.uniform(.6,.9))
            if split=='test_environment':
                center=[float(rng.uniform(.06,.18)),float(rng.uniform(-.03,.03)),float(rng.uniform(-.11,-.06))];width=float(rng.uniform(.045,.065));A=float(rng.choice([-1,1])*rng.uniform(.18,.35));tilt=rng.uniform(-.5,.5,2);g=np.r_[tilt,-1.];g*=9.81/np.linalg.norm(g)
            rows.append(dict(E=E,Kleft=float(K[0]),Kright=float(K[1]),layout=LAYOUTS[i%3],gravity=g.tolist(),field_amplitude_J=A,field_center=center,field_width=width,material_gradient=h))
        s[split]=rows
    write(path,s);return s

def interpolate_samples(field,query,n=1):
    s=np.linspace(-1,1,n+1);z=np.sign(s)*.04*(1-np.cos(np.pi*abs(s)))/2
    axes=[np.linspace(0,.24,4*n+1),np.linspace(-.04,.04,n+1),z]
    cell=np.stack([np.clip(np.searchsorted(a,query[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(axes)],axis=1)
    lo=np.stack([a[cell[:,i]] for i,a in enumerate(axes)],axis=1);width=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(axes)],axis=1)
    N,_=basis(2*(query-lo)/width-1);offset=np.stack(np.meshgrid(*[np.arange(3)]*3,indexing='ij'),-1).reshape(-1,3)
    ids=np.ravel_multi_index((2*cell[:,None]+offset).transpose(2,0,1),(8*n+1,2*n+1,2*n+1));return np.einsum('qa,qai->qi',N,field[ids])

def level(split,i,n,order=3,warm=None):
    r=protocol()[split][i];source={name:hashlib.sha256((ROOT/'scripts/position_ood'/name).read_bytes()).hexdigest() for name in ['fixture.py','material_field.py']}
    if BOUNDARY=='smooth':source['smooth_fixture.py']=hashlib.sha256((ROOT/'scripts/position_ood/smooth_fixture.py').read_bytes()).hexdigest()
    signature=hashlib.sha256(json.dumps(dict(case=r,n=n,order=order,actions=ACTIONS,source=source),sort_keys=True).encode()).hexdigest();dest=BASE/f'references/{split}-{i}/n{n}-q{order}.npz'
    if dest.exists():
        with np.load(dest) as z:
            assert str(z['signature'])==signature,'Stale reference';return {k:z[k] for k in z.files}
    within_deadline();device=os.environ.get('POSITION_REFERENCE_DEVICE','cuda:0');fixture=MaterialPositionFixture
    if BOUNDARY=='smooth':
        from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
        fixture=SmoothMaterialFixture
    model=fixture(n,**r,order=order,device=device)
    if device=='cpu':model.linear_solver='cpu_multigrid'
    rows=[];started=time.time();previous_full=None
    try:
        for j,(action,amp) in enumerate(ACTIONS):
            within_deadline();checkpoint=dest.parent/f'checkpoints/n{n}-q{order}-{j}.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as z:
                    assert str(z['signature'])==signature;row={k:z[k] for k in z.files if k!='signature'}
            else:
                model.set_controls(controls(action,amp))
                initial=previous_full
                if initial is None and warm is not None:
                    initial=interpolate_samples(warm['full_displacement'][j],model.rest,int(warm['mesh_n'])) if 'full_displacement' in warm else interpolate_samples(warm['field'][j],model.rest)
                if initial is not None:
                    try:model.evaluate(initial)
                    except ValueError:
                        print('DISCARD_INVALID_WARM_START',split,i,n,order,j,flush=True);initial=None
                state=model.solve(initial=initial,force_tolerance_N=1e-9);row=model.sampled(state);row['full_displacement']=state['displacement_local_m']
                assert row['residual_N']<1e-9 and row['reaction_balance_N']<1e-7 and row['minimum_J']>.1
                checkpoint.parent.mkdir(parents=True,exist_ok=True);tmp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(tmp,signature=signature,**row);tmp.replace(checkpoint)
            previous_full=row.get('full_displacement');rows.append(row);print('REFERENCE_ACTION',split,i,n,order,j,flush=True)
        keys=set.intersection(*(set(row) for row in rows));data={k:np.stack([r[k] for r in rows]) for k in keys};data.update(mesh_n=n,reference_device=device,signature=signature,elapsed_seconds=time.time()-started,poses=np.array([controls(a,b) for a,b in ACTIONS]))
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(dest);return data
    finally:
        model.clear_linear_cache();model.engine=None;del model;gc.collect()
        import torch;torch.cuda.empty_cache()

def reference(split,i):
    directory=BASE/f'references/{split}-{i}';directory.mkdir(parents=True,exist_ok=True);lock=(directory/'reference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    final=BASE/f'fields/{split}-{i}.npz';audit=BASE/f'qualification/{split}-{i}.json'
    if final.exists() and audit.exists():return
    prev=None;checks=[]
    for n in protocol()['reference_levels']:
        current=level(split,i,n,warm=prev)
        if prev is not None:
            check=comparison(prev,current);check['rms_passed']=bool(max(check['rms_m'])<2e-5);checks.append(dict(n=n,**check));print('REFINEMENT',split,i,n,'rms_passed',check['rms_passed'],'pointwise_passed',check['passed'],max(check['rms_m']),max(check['maximum_m']),flush=True)
            if max(check['rms_m'])<2e-5:
                q4=level(split,i,n,4,warm=current);qcheck=comparison(current,q4);qcheck['rms_passed']=bool(max(qcheck['rms_m'])<2e-5)
                if max(qcheck['rms_m'])<2e-5:
                    write(audit,dict(case=protocol()[split][i],mesh=n,spatial=checks,quadrature=qcheck,qualified=True,boundary_model=BOUNDARY,qualification_scope='RMS displacement refinement below 20 micrometres; maximum-coordinate refinement is a separate reported diagnostic',strict_pointwise_passed=bool(check['passed'] and qcheck['passed'])));final.parent.mkdir(exist_ok=True);tmp=final.with_suffix('.tmp.npz');np.savez_compressed(tmp,**current);tmp.replace(final);return
        prev=current
    write(audit,dict(mesh=n,spatial=checks,qualified=False));raise RuntimeError(f'{split}-{i}: reference not qualified')

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args();s=protocol();status=BASE/f'reference-{a.split}-{a.worker}-status.json';write(status,dict(state='running',time=time.time()))
    failures=[]
    for i in range(a.worker,len(s[a.split]),a.workers):
        try:reference(a.split,i)
        except Exception as e:
            failures.append(dict(case=i,error=repr(e)));write(status,dict(state='running',time=time.time(),failures=failures));print('CASE_FAILED',a.split,i,repr(e),flush=True)
    write(status,dict(state='complete' if not failures else 'completed_with_failures',time=time.time(),failures=failures))
if __name__=='__main__':main()
