"""Bounded CPU-only joint algorithm-cost control on explicitly synthetic observations.

Not a cohort, force qualification, policy comparison or physical proof. Recipe
and source hashes are sealed before the separately owned worker starts.
"""
import argparse,ast,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.campaign_surface_conditional_amplitude_v01 import ConditionalAmplitudeReference,AmplitudePlan,RatioModal,ExactRatioMemo
from experiments.campaign_surface_condensed_reference_v01 import CondensedStaticReference
from experiments import campaign_surface_p2_static_v02 as physics
from experiments import campaign_surface_p2_force_v03 as forces
from experiments.campaign_surface_force_study_v01 import observation_points
from experiments.campaign_surface_p2_inference_v01 import LOG_BOUNDS,LabelledModal
from experiments.campaign_surface_response_cache_v01 import model_fingerprint

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/operator-dev/surface-amplitude-cpu-cost-v01'
WORKER_CPU=17
CONTROLLER_CPU=16
TARGETS=('normal_x135','normal_x225','paired_axial_x135','paired_axial_x225')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,p)


def sources():
    pending=['experiments/benchmark_surface_amplitude_cpu_v01.py','tests/test_surface_conditional_amplitude_v01.py','tests/test_amplitude_adaptive_oracle_v01.py','tests/test_surface_amplitude_cpu_v01.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('Missing source '+name)
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or'']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for module in modules:
                relative=module.replace('.','/')+'.py'
                if(ROOT/relative).is_file()and relative not in seen:pending.append(relative)
    return {name:sha(ROOT/name)for name in sorted(seen)}


def prepare():
    if(OUT/'recipe.json').exists():raise ValueError('No refreeze/retry')
    recipe=dict(schema='surface-amplitude-cpu-cost-v01',scope='Synthetic finite-linear algorithm cost only; no physics/catalog admission or actual cohort',
        mesh_n=4,all_modes=True,device="cpu",CPU=[WORKER_CPU],controller_CPU=CONTROLLER_CPU,threads=1,worker_RSS_cap_bytes=4*1024**3,
        worker_wall_s=600,global_wall_s=600,GPU=False,
        plan=dict(inner=[32,64,128],outer=[128,256,512],moment_chunk=64),
        synthetic_parameter=[4000.,12.],noise='Deterministic20um*sin(channel+1) prefix;20um*cos(channel+.5) reset; not a random cohort draw',
        cases=['joint_independent_cold'],prefix='normal_x180',probe='normal_x090',
        ratio_memo_bytes=256*1024**2,source_SHA256=sources(),actual_cohort_draws=0,policy_evaluations=0)
    write(OUT/'recipe.json',recipe);return sha(OUT/'recipe.json')


def validate(digest):
    if sha(OUT/'recipe.json')!=digest:raise ValueError('Recipe hash mismatch')
    p=json.loads((OUT/'recipe.json').read_text())
    if p['source_SHA256']!=sources():raise ValueError('Source changed after recipe seal')
    return p


def worker(digest):
    p=validate(digest)
    guard()
    started=time.monotonic();fwd=physics.SparseStaticForward(4);loads=forces.load_catalog(4)
    f=np.column_stack([x.ravel()[fwd.free]for x in loads.values()]);P=physics.point_query_matrix(fwd,observation_points())
    modal=CondensedStaticReference(fwd.Kunit_free,fwd.W_free,f,P);model=LabelledModal(modal,tuple(loads))
    build_s=time.monotonic()-started;CPU=RatioModal(modal,model.labels)
    def ratio(z,labels):return CPU(np.exp(z),labels)
    memo=ExactRatioMemo(ratio,model_fingerprint=model_fingerprint(model),maximum_bytes=p['ratio_memo_bytes'])
    unit=(np.log([4000.,12.])-LOG_BOUNDS[:,0])/np.diff(LOG_BOUNDS,axis=1).ravel()
    y0=model.select(['normal_x180'])(unit[None])[0,0]+20e-6*np.sin(np.arange(24)+1)
    y1=model.select(['normal_x090'])(unit[None])[0,0]+20e-6*np.cos(np.arange(24)+.5)
    cases=[[('normal_x180',y0),('normal_x090',y1)]]
    rows=[]
    for name,observations in zip(p['cases'],cases):
        begin=time.monotonic();engine=ConditionalAmplitudeReference(None,observations,TARGETS,log_ratio_predict=memo)
        answer=engine.qualify(plan=AmplitudePlan());elapsed=time.monotonic()-begin
        rows.append(dict(name=name,wall_s=elapsed,qualified=answer['qualified'],numerical=answer['diagnostics'],memo=memo.diagnostics()))
        write(OUT/'progress.json',dict(completed_cases=len(rows),rows=rows,recipe_SHA256=digest,pid=os.getpid()))
        del answer,engine
    result=dict(schema='surface-amplitude-cpu-cost-v01',completed=True,recipe_SHA256=digest,build_s=build_s,rows=rows,
        all_numerical_controls_qualified=all(r['qualified']for r in rows),wall_s=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        GPU=False,modal_diagnostics=modal.diagnostics,
        actual_cohort_draws=0,actual_policy_outcomes=0,scope=p['scope'])
    write(OUT/'result.json',result)


def guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.sched_getaffinity(0)!={WORKER_CPU}or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('Exact CPU17/one-thread/noGPU guard required')


def command(digest,*,guard_only=False):
    return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','taskset','-c',str(WORKER_CPU),sys.executable,'-m','experiments.benchmark_surface_amplitude_cpu_v01','guard'if guard_only else'worker','--digest',digest]


def run(digest):
    p=validate(digest)
    if os.sched_getaffinity(0)!={CONTROLLER_CPU}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('ControllerCPU16/noGPU required')
    from experiments.run_campaign_surrogate_sbc_v03 import owned_jobs
    from experiments.campaign_owned_proc import proc_rss
    def check():
        path=OUT/'journal.json';owned=[] if not path.exists()else json.loads(path.read_text()).get('live',[])
        if proc_rss(os.getpid())+sum(proc_rss(r['pid'])for r in owned)>4*1024**3:raise MemoryError('Aggregate4GiB RSS cap')
    records=owned_jobs(lambda i,slot:command(digest),[0],1,OUT,worker_wall_s=600,global_wall_s=600,RSS_bytes=4*1024**3,controller_check=check)
    write(OUT/'owned-receipt.json',dict(completed=True,all_reaped=True,controller_pid=os.getpid(),recipe_SHA256=digest,records=records))
    return dict(completed=True,records=records)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','guard']);a.add_argument('--digest');x=a.parse_args()
    if x.mode=='prepare':print(prepare())
    elif x.mode=='run':print(json.dumps(run(x.digest)))
    elif x.mode=='guard':guard();print(json.dumps(dict(guard_passed=True)))
    else:worker(x.digest)
