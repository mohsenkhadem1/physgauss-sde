"""Bounded deterministic Fisher screen; no fresh64 data, PDE, RNG or GPU.

Root must release the hash-bound recipe. Run under an independent owned RSS
watchdog as well as this process's120s alarm/high-water guard.
"""
import argparse,ast,json,os,resource,shutil,signal,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_nonlinear_posterior_cost_v03 as cost
from experiments import run_surface_p2_study_v04 as model_tools
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
from experiments.campaign_action_fisher_screen_v01 import total_jacobian,screen,SIGMA,TIE_RELATIVE,TIE_ABSOLUTE_M2
ROOT=cost.ROOT;LOCAL=ROOT/'runs/operator-dev/action-fisher-screen-v01'
SOURCE='experiments/run_action_fisher_screen_v01.py';TEST='tests/test_run_action_fisher_screen_v01.py'
CPU_COST_SHA='3a98be21beb6e11fccd06224fe9d2809a69cb293be3ad074076f435ac583543b'
PREFIX='normal_x180'
ACTIONS=tuple(family+'_x'+x for family in('normal','paired_axial')for x in('090','180','270'))
TARGETS=tuple(family+'_x'+x for family in('normal','paired_axial')for x in('135','225'))
RESOURCES=dict(CPU=17,GPU=False,threads=1,wall_s=120,RSS_bytes=4*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,independent_owned_RSS_watchdog_required=True)
MATH=dict(points='Existing exact Cheb9 tensor grid,81 deterministic nodes; counts are not prior-weighted population probabilities',prior_covariance='Unit-square uniform covariance I/12 approximated locally by an unconstrained Gaussian, including boundary rows',prior_precision=12.,sigma_m=SIGMA,prefix=PREFIX,actions=list(ACTIONS),targets=list(TARGETS),risk='Unweighted trace(Jtarget*C*Jtarget.T)/96',ties=dict(relative_to_prefix_risk=TIE_RELATIVE,absolute_m2=TIE_ABSOLUTE_M2),refinement='Record ALL81 n5-vs-n9 rank comparisons; disagreements never filtered or used as an admission gate',FD_steps=[1e-4,5e-5],FD_agreement_absolute_m_per_unit=1e-10,FD_agreement_relative=1e-6,all_linear_modes_retained_unclipped=True,force_budgets='Original catalog unchanged; normal and axial are not equal force or cost',scope='Local-Gaussian/Fisher screening only; no observations/posteriors/closed-loop policy or advantage inference')
sha,record,write=cost.sha,cost.record,cost.write


def sources():
    todo=[SOURCE,TEST,'tests/test_action_fisher_screen_v01.py'];seen=set()
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse((ROOT/p).read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:names=[node.module or '']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for name in names:
                path=name.replace('.','/')+'.py'
                if(ROOT/path).is_file()and path not in seen:todo.append(path)
    return {p:sha(ROOT/p)for p in sorted(seen)}


def guard():
    if set(os.sched_getaffinity(0))!={17}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('ExactCPU17/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['RSS_bytes']:raise MemoryError('Own4GiB RSS highwater')
    if model_tools.available_memory()<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('RAM/disk reserves')


def prepare():
    guard();receipt=cost.completed_cost_gate(CPU_COST_SHA);p,_=cost.validate(CPU_COST_SHA)
    origin=dict(CPU_cost=receipt,full_atlas=p['origin_v01']['full_atlas'],full_protocol_SHA256=p['full_protocol_SHA256'],atlas_fingerprint=p['full_atlas_fingerprint'],labels=p['all_target_labels'])
    recipe=dict(schema='action-fisher-screen-v01',source_SHA256=sources(),origin=origin,resources=RESOURCES,math=MATH,no_fresh64_access=True,no_RNG_PDE_GPU=True)
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',recipe);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['math']!=MATH or p['origin']['CPU_cost']!=cost.completed_cost_gate(CPU_COST_SHA):raise ValueError('Frozen screen source/math/resources/CPUcost admission')
    return p


def release_gate(digest):
    p=validate(digest);r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('independent_RSS_watchdog')is not True:raise ValueError('Concrete root release and independent RSS watchdog required')
    return p


def finite_difference(predict,unit,h):
    """Independent fourth-order central or one-sided prior-contained stencil."""
    q=np.asarray(unit,float);out=[]
    for d in range(2):
        result=None
        for mask,offsets,weights,step in [(q[:,d]<2*h,[0,1,2,3,4],[-25,48,-36,16,-3],h),(q[:,d]>1-2*h,[0,1,2,3,4],[-25,48,-36,16,-3],-h),((q[:,d]>=2*h)&(q[:,d]<=1-2*h),[-2,-1,0,1,2],[1,-8,0,8,-1],h)]:
            if not np.any(mask):continue
            value=None
            for offset,w in zip(offsets,weights):
                if w==0:continue
                points=q[mask].copy();points[:,d]+=offset*step;term=w*np.asarray(predict(points),float)
                value=term if value is None else value+term
            value/=12*step
            if result is None:result=np.empty((len(q),*value.shape[1:]))
            result[mask]=value
        out.append(result)
    return np.stack(out,axis=-1)


def aggregate(results):
    public={}
    for n,row in results.items():
        winners=row['winner_mask'];unique=winners.sum(axis=1)==1
        public[str(n)]=dict(points=81,winner_membership_counts=dict(zip(ACTIONS,winners.sum(axis=0).astype(int).tolist())),unique_winner_counts=dict(zip(ACTIONS,winners[unique].sum(axis=0).astype(int).tolist())),tied_points=int((~unique).sum()),top_margin_m2_range=[float(row['top_margin_m2'].min()),float(row['top_margin_m2'].max())])
    public['n5_n9_rank_disagreement_points']=int(np.any(results[5]['winner_mask']!=results[9]['winner_mask'],axis=1).sum());public['all81_refinement_comparisons_retained']=True
    return public


def run(digest):
    guard();p=release_gate(digest)
    if(LOCAL/'result-LOCAL.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();result=dict(completed=False,passed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),no_fresh64_access=True,new_draws=0,new_PDE=0,GPU=False)
    def save():result.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result-LOCAL.json',result)
    def stopped(*_):raise TimeoutError('Prospective120s screen cap')
    old=signal.signal(signal.SIGALRM,stopped);signal.alarm(120)
    try:
        origin=p['origin'];path=ROOT/origin['full_atlas']['archive']['path']
        if sha(path)!=origin['full_atlas']['archive']['sha256']:raise ValueError('Admitted atlas bytes')
        with np.load(path,allow_pickle=False)as z:atlas=NonlinearResidualAtlas(z['delta'],tuple(str(v)for v in z['labels']),provenance_fingerprint=origin['full_protocol_SHA256'])
        if atlas.fingerprint!=origin['atlas_fingerprint']or list(atlas.labels)!=origin['labels']:raise ValueError('Original atlas fingerprint')
        fwd,model=model_tools.make_model(device_mode='cpu');guard()
        if model_tools.model_fingerprint(model)!=origin['CPU_cost']['memo']['model_fingerprint']:raise ValueError('Exact admitted linear model fingerprint')
        if len(model.modal.eigenvalues)!=2976:raise ValueError('All2976 boundary modes retained, no clipping/truncation')
        unit=np.stack(np.meshgrid(atlas.axis,atlas.axis,indexing='ij'),axis=-1).reshape(-1,2);labels=atlas.labels;results={};Jacobians={}
        for n in(5,9):
            chosen=atlas if n==9 else atlas.coarsen(n);J=total_jacobian(model,chosen,unit,labels);Jacobians[n]=J
            prefix=J[:,labels.index(PREFIX)];actions=J[:,[labels.index(v)for v in ACTIONS]];targets=J[:,[labels.index(v)for v in TARGETS]].reshape(81,96,2)
            results[n]=screen(prefix,actions,targets);guard()
        predict=lambda q:model.select(labels)(q)+atlas.predict(q,labels)
        checks=[]
        for h in MATH['FD_steps']:
            FD=finite_difference(predict,unit,h);error=float(np.max(abs(FD-Jacobians[9])));limit=MATH['FD_agreement_absolute_m_per_unit']+MATH['FD_agreement_relative']*float(np.max(abs(Jacobians[9])))
            checks.append(dict(step=h,maximum_absolute_error_m_per_unit=error,tolerance_m_per_unit=limit,passed=error<=limit));guard()
        arrays=dict(unit=unit,**{f'n{n}_{key}':value for n,row in results.items()for key,value in row.items()});arrays.update(n5_Jacobian=Jacobians[5],n9_Jacobian=Jacobians[9]);np.savez_compressed(LOCAL/'pointwise-screen-LOCAL.npz',**arrays)
        result.update(completed=True,passed=all(row['passed']for row in checks),derivative_checks=checks,aggregates=aggregate(results),maximum_n5_n9_Jacobian_difference_m_per_unit=float(np.max(abs(Jacobians[5]-Jacobians[9]))),raw=record(LOCAL/'pointwise-screen-LOCAL.npz'),minimum_generalized_eigenvalue=float(model.modal.eigenvalues.min()),all_modes_unclipped=True,scope=MATH['scope']);guard();save()
    except BaseException as error:result['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);save()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','guard']);parser.add_argument('--sha');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='guard':guard();print('guard passed')
    else:r=run(args.sha);print(json.dumps(dict(completed=r['completed'],passed=r['passed'])));raise SystemExit(0 if r['completed']and r['passed']else 1)
