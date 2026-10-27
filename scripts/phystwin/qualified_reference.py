"""v05: consistent quadratic reference bank and per-object adaptive qualification."""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.active_support_reference import ROOT,BOUNDS,LAYOUTS,ACTIONS,cohort
from scripts.phystwin.consistent_fixture import P2Fixture,query
from scripts.phystwin.qualify_fixture import metrics
from scripts.phystwin.summarize_fixture import compare
BASE=ROOT/'runs/phystwin-qualified-support-v05'


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def protocol():
    gates=json.loads((ROOT/'reports/phystwin-fixture-convergence-v04/protocol.json').read_text())
    p=dict(version=5,layouts=LAYOUTS,actions=ACTIONS,development=cohort(12,401),calibration=cohort(20,602),test=cohort(24,603),
        bank_grid=9,bank_mesh_n=4,truth_levels=[4,6,8,12,16,20],noise_sigma_m=.0003,noise_repeats=4,
        qualification=gates,required_consecutive_spatial_passes=2,
        model='Consistent surface Robin integration, graded P2 tetrahedra, log-J Neo-Hookean; all four loads unchanged from v03',
        split_policy='Development reused for diagnosis; calibration and test parameter draws newly seeded. Test outcomes do not select models, calibration scales or policies.',
        prior='Uniform fixture category; independent log-uniform E[1500,10000] Pa and K[400,16000] N/m',
        bank_interpolation='Cubic tensor interpolation in log E,K; development direct-solve check <=30 micrometres RMS',
        interpolation_gate_rms_m=.00003,probe_nonrigid_rms_gate_m=.00005,probe_nonrigid_fraction_gate=.02,
        quadrature_sizes=[33,65,129,257],quadrature_gate_mean_sd_m=.00001,quadrature_gate_log_quantile=.02,
        utility_samples=[256,1024,4096],utility_seed=731,
        discrepancy='Development-only fine-minus-bank mean and joint covariance, 10% diagonal shrinkage and 1 micrometre SD floor',
        calibration_rule='Object-wise max over all three observation budgets, 243 target coordinates and four repeats; 19th/20 score for 90% simultaneous bands',
        baselines=['calibration-selected fixed','uniform random','inferred fixture-side heuristic'],
        inference_success='Posterior predictive moments and marginal physical quantiles converge; report raw E/K coverage separately from calibrated displacement bands',
        probing_success='Positive object-bootstrap 95% improvement versus fixed AND fixture-side heuristic, coverage >=90%, stable numerical decisions',
        scope='Known geometry and force, unknown E/K and fixture category; static synthetic numerical reference, not real video, gravity/contact or learned PhysTwin')
    path=BASE/'protocol.json';BASE.mkdir(parents=True,exist_ok=True)
    if path.exists():assert json.loads(path.read_text())==p,'Frozen v05 protocol mismatch'
    else:write_json(path,p)
    return p


def build(n,r,device,order=3):
    # Serialize the largest meshes across workers to bound host-memory use.
    lock=None
    if n>=32:
        import fcntl
        lock=open(BASE/'large-mesh.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:m=P2Fixture(n,r['E'],r['K'],r['layout'],order=order,device=device,grading=True,memory_limit_bytes=80*1024**3,iterative=n>=8)
    except BaseException:
        if lock is not None:lock.close()
        raise
    m._large_mesh_lock=lock
    # The available GPUs have 11 GiB. Keep the largest fine operator on CPU;
    # constitutive assembly can still use the GPU in bounded element batches.
    m.linear_solver='cpu_multigrid' if n>=40 or (n>=8 and not device) else ('gpu_multigrid' if device and n>=8 else 'pardiso');return m


def release(m):
    if m is not None:
        m.clear_linear_cache();m.engine=None;m.rows=None;m.cols=None
        if getattr(m,'_large_mesh_lock',None) is not None:
            m._large_mesh_lock.close();m._large_mesh_lock=None
    gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()


def sample81(u):return u.reshape(17,5,5,3)[::2,::2,::2].reshape(81,3)


def nonrigid(u):
    x=np.stack(np.meshgrid(np.linspace(0,.24,17),np.linspace(-.04,.04,5),np.linspace(-.04,.04,5),indexing='ij'),-1).reshape(-1,3)
    y=x+u;X=x-x.mean(0);Y=y-y.mean(0);v,_,w=np.linalg.svd(X.T@Y);D=np.eye(3);D[-1,-1]=np.linalg.det(v@w)
    return float(np.sqrt(np.mean((X@v@D@w-Y)**2))),float(np.sqrt(np.mean(u*u)))


def checked_compare(a,b,spec,r,action,quadrature=False):
    result=compare(a,b,spec['qualification'],quadrature)
    if r['layout'].endswith('full') and action=='push_'+r['layout'].split('_')[0]:
        # Exact continuum equilibrium: uniform translation F/K, zero strain.
        # Relative errors of roundoff-sized zero quantities are undefined.
        exact=np.array([0.,.03/r['K'],0.]);traction=-np.array([0.,.03,0.])/(.08*.08)
        result['analytic_uniform_translation']=True
        result['analytic_displacement_error_m']=float(max(abs(v['sampled_u']-exact).max() for v in [a,b]))
        result['analytic_strain_error']=float(max(abs(v['strain']).max() for v in [a,b]))
        result['analytic_elastic_energy_J']=float(max(abs(v['elastic_J']) for v in [a,b]))
        result['analytic_traction_error_Pa']=float(max(abs(v['traction']-traction).max() for v in [a,b]))
        result['checks']['analytic_translation']=result['analytic_displacement_error_m']<=1e-12
        result['checks']['strain']=result['analytic_strain_error']<=1e-12
        result['checks']['analytic_traction']=result['analytic_traction_error_Pa']<=1e-9
        result['checks']['energy']=result['analytic_elastic_energy_J']<=1e-20 and result['spring_J_relative']<=spec['qualification']['quadrature_energy_relative_tolerance' if quadrature else 'energy_relative_tolerance']
        result['pass']=all(result['checks'].values())
    return result


def solve(m,action,initial=None):
    m.set_action(action)
    try:return m.solve(initial,force_tolerance_N=1e-10)
    except Exception:
        if initial is not None:raise
        u=np.zeros_like(m.rest)
        for a in [.25,.5,.75,1.]:
            m.set_action(action,a);state=m.solve(u,force_tolerance_N=1e-10);u=state['displacement_local_m']
        return state


def bank_job(name,r,device,n=4):
    import fcntl
    directory=BASE/'job-locks';directory.mkdir(parents=True,exist_ok=True)
    with open(directory/f'bank-n{n}-{name}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _bank_job(name,r,device,n)


def _bank_job(name,r,device,n=4):
    path=(BASE/'fields' if n==4 else BASE/'banks'/f'n{n}')/f'{name}.npz'
    if path.exists():return
    start=time.time();m=build(n,r,device);fields=[]
    coarse=BASE/'fields'/f'{name}.npz'
    warm=np.load(coarse)['fields'] if n>4 and coarse.exists() else None
    for j,action in enumerate(ACTIONS):
        initial=None
        if warm is not None:
            from scipy.interpolate import RegularGridInterpolator
            initial=RegularGridInterpolator((np.linspace(0,.24,9),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3)),warm[j].reshape(9,3,3,3))(m.rest)
        state=solve(m,action,initial);u=state['displacement_local_m'];fields.append(query(m,u,observation_points()))
    path.parent.mkdir(exist_ok=True,parents=True);temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,fields=fields,parameters=[r['E'],r['K']],layout=r['layout']);temp.replace(path)
    release(m);print('BANK',name,'seconds',round(time.time()-start,2),flush=True)


def observation_points():
    return np.stack(np.meshgrid(np.linspace(0,.24,9),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3)


def truth_job(name,r,device):
    import fcntl
    directory=BASE/'job-locks';directory.mkdir(parents=True,exist_ok=True)
    with open(directory/f'truth-{name}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _truth_job(name,r,device)


def _truth_job(name,r,device):
    spec=protocol();final=BASE/'fields'/f'{name}-qualified.npz';audit=BASE/'qualification'/f'{name}.json'
    if final.exists() and audit.exists():
        assert json.loads(audit.read_text())['qualified'];return
    directory=BASE/'refinements'/name;directory.mkdir(parents=True,exist_ok=True)
    previous=None;old=[];old_n=None;checks=[];streak=0
    extension=BASE/'refinement-extension.json'
    levels=spec['truth_levels']+(json.loads(extension.read_text())['additional_levels'] if extension.exists() else [])
    extension2=BASE/'refinement-extension-v2.json'
    if extension2.exists():levels+=json.loads(extension2.read_text())['additional_levels']
    for n in levels:
        release(previous);m=build(n,r,device);current=[]
        for j,action in enumerate(ACTIONS):
            path=directory/f'n{n}-q3-{action}.npz'
            if path.exists():
                with np.load(path) as z:row={k:z[k] for k in z.files}
            else:
                start=time.time();initial=query(previous,old[j]['u'],m.rest) if previous is not None else None
                state=solve(m,action,initial);u=state['displacement_local_m'];row=dict(u=u,elapsed_seconds=time.time()-start,**metrics(m,u,state))
                tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(path)
                print('TRUTH',name,n,action,'seconds',round(time.time()-start,2),flush=True)
            current.append(row)
        if old:
            paired=[checked_compare(a,b,spec,r,action) for a,b,action in zip(old,current,ACTIONS)]
            passed=all(v['pass'] for v in paired);streak=streak+1 if passed else 0
            checks.append(dict(coarse=old_n,fine=n,actions=paired,passed=passed))
            print('SPATIAL',name,old_n,n,passed,'streak',streak,flush=True)
        previous,old,old_n=m,current,n
        write_json(audit,dict(case=r,qualified=False,spatial=checks,latest_mesh=n))
        if streak>=spec['required_consecutive_spatial_passes']:break
    if streak<spec['required_consecutive_spatial_passes']:
        release(m);raise RuntimeError(f'{name}: spatial qualification failed through n={n}')
    release(previous);previous=None;m=build(n,r,device,4);quad=[]
    for j,action in enumerate(ACTIONS):
        path=directory/f'n{n}-q4-{action}.npz'
        if path.exists():
            with np.load(path) as z:row={k:z[k] for k in z.files}
        else:
            start=time.time();state=solve(m,action,old[j]['u']);u=state['displacement_local_m'];row=dict(u=u,elapsed_seconds=time.time()-start,**metrics(m,u,state))
            tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(path)
        quad.append(checked_compare(old[j],row,spec,r,action,True))
    qualified=all(q['pass'] for q in quad)
    write_json(audit,dict(case=r,qualified=qualified,spatial=checks,quadrature=quad,latest_mesh=n))
    if not qualified:release(m);raise RuntimeError(f'{name}: quadrature qualification failed')
    fields=np.stack([sample81(row['sampled_u']) for row in old]);deformation=np.array([nonrigid(row['sampled_u']) for row in old])
    final.parent.mkdir(exist_ok=True);tmp=final.with_suffix('.tmp.npz');np.savez_compressed(tmp,fields=fields,nonrigid=deformation,parameters=[r['E'],r['K']],layout=r['layout'],mesh_n=n);tmp.replace(final)
    release(m);print('QUALIFIED',name,n,flush=True)


def jobs(spec,split):
    if split=='bank':
        return [(f'bank9-{l}-{i}-{j}',dict(layout=layout,E=float(np.exp(e)),K=float(np.exp(k)))) for l,layout in enumerate(LAYOUTS) for i,e in enumerate(np.linspace(*BOUNDS[0],9)) for j,k in enumerate(np.linspace(*BOUNDS[1],9))]
    return [(f'{split}-{i}',r) for i,r in enumerate(spec[split])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['bank','development','calibration','test'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--device',default='cuda:0');p.add_argument('--bank-mesh',type=int,choices=[4,8,12],default=4);p.add_argument('--start-index',type=int,default=0);p.add_argument('--keep-going',action='store_true');a=p.parse_args()
    spec=protocol()
    pending=jobs(spec,a.split)
    if a.split=='bank' and a.bank_mesh>4:pending+=[(f'development-{i}',r) for i,r in enumerate(spec['development'])]
    failures=[]
    for i,(name,r) in enumerate(pending):
        if i<a.start_index or i%a.workers!=a.worker:continue
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        try:
            if a.split=='bank':bank_job(name,r,a.device,a.bank_mesh)
            else:truth_job(name,r,a.device)
        except Exception as e:
            write_json(BASE/'failures'/f'{name}.json',dict(error=str(e),case=r));failures.append(name)
            if not a.keep_going:raise
            print('FAILED',name,str(e),flush=True);gc.collect()
            import torch
            torch.cuda.empty_cache()
    if failures:raise RuntimeError(f'Unqualified cases remain: {failures}')

if __name__=='__main__':main()
