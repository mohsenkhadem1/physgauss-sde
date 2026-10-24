"""Three independent synthetic joint posteriors sharing GPU1; no cohort.

Separate source/recipe; completed single-worker control remains immutable.
"""
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import benchmark_surface_amplitude_v01 as base

ROOT=base.ROOT;OUT=ROOT/'runs/operator-dev/surface-amplitude-concurrency-v01';UUID=base.UUID
sha=base.sha;write=base.write


def sources():
    return dict(base.sources(),**{p:sha(ROOT/p)for p in('experiments/benchmark_surface_amplitude_concurrency_v01.py','tests/test_surface_amplitude_concurrency_v01.py')})


def prepare():
    if(OUT/'recipe.json').exists():raise ValueError('No refreeze/retry')
    recipe=dict(schema='surface-amplitude-concurrency-v01',workers=3,worker_CPU=[0,1,2],controller_CPU=3,
        GPU_UUID=UUID,individual_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=12*1024**3,
        owned_physical_GPU_VRAM_cap_bytes=4*1024**3,Torch_allocated_cap_bytes_per_worker=512*1024**2,
        worker_wall_s=500,global_wall_s=600,source_SHA256=sources(),
        observations='Identical fixed deterministic joint synthetic data as singleworker control; independent model/memo perworker',
        cases_per_worker=1,all_modes=True,default_plan=dict(inner=[32,64,128],outer=[128,256,512]),
        no_cohort_or_policy_outcomes=True,scope='Algorithm throughput only; not physical qualification or calibration')
    write(OUT/'recipe.json',recipe);return sha(OUT/'recipe.json')


def validate(digest):
    if sha(OUT/'recipe.json')!=digest:raise ValueError('Recipe SHA mismatch')
    p=json.loads((OUT/'recipe.json').read_text())
    if p['source_SHA256']!=sources():raise ValueError('Frozen source mismatch')
    return p


def guard(slot):
    if slot not in range(3)or os.sched_getaffinity(0)!={slot}or os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU/one-thread/GPU1 environment required')


def worker(slot,digest):
    p=validate(digest);guard(slot);start=time.monotonic();fwd=base.physics.SparseStaticForward(4);loads=base.forces.load_catalog(4)
    f=np.column_stack([value.ravel()[fwd.free]for value in loads.values()]);P=base.physics.point_query_matrix(fwd,base.observation_points())
    modal=base.CondensedStaticReference(fwd.Kunit_free,fwd.W_free,f,P)
    model=base.Float64GPUModal(modal,tuple(loads),physical_UUID=UUID,allowed_UUIDs=[UUID],maximum_batch=4096,allocated_cap_bytes=512*1024**2)
    build_s=time.monotonic()-start;CPU=base.RatioModal(modal,model.labels)
    def ratio(z,labels):return CPU(np.exp(z),labels)if len(z)<16 else model.ratio_select(labels)(z)
    memo=base.ExactRatioMemo(ratio,model_fingerprint=base.model_fingerprint(model),maximum_bytes=256*1024**2)
    unit=(np.log([4000.,12.])-base.LOG_BOUNDS[:,0])/np.diff(base.LOG_BOUNDS,axis=1).ravel()
    y0=model.select(['normal_x180'])(unit[None])[0,0]+20e-6*np.sin(np.arange(24)+1)
    y1=model.select(['normal_x090'])(unit[None])[0,0]+20e-6*np.cos(np.arange(24)+.5)
    begin=time.monotonic();result=base.ConditionalAmplitudeReference(None,[('normal_x180',y0),('normal_x090',y1)],base.TARGETS,log_ratio_predict=memo).qualify(plan=base.AmplitudePlan())
    write(OUT/f'worker-{slot}-result.json',dict(slot=slot,completed=True,pid=os.getpid(),recipe_SHA256=digest,
        build_s=build_s,posterior_s=time.monotonic()-begin,total_s=time.monotonic()-start,qualified=result['qualified'],
        numerical=result['diagnostics'],memo=memo.diagnostics(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_Torch_allocated_bytes=model.torch.cuda.max_memory_allocated(model.device),actual_cohort_draws=0,policy_outcomes=0))


def command(slot,digest,*,guard_only=False):
    return ['env','CUDA_VISIBLE_DEVICES='+UUID,'CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','taskset','-c',str(slot),sys.executable,'-m','experiments.benchmark_surface_amplitude_concurrency_v01','guard'if guard_only else'worker','--slot',str(slot),'--digest',digest]


def run(digest):
    p=validate(digest)
    from experiments.run_campaign_surrogate_sbc_v03 import owned_jobs
    from experiments.campaign_owned_proc import proc_rss
    telemetry=dict(maximum_aggregate_RSS_bytes=0,maximum_owned_GPU_VRAM_bytes=0)
    def check():
        if not(OUT/'journal.json').exists():return
        owned={r['pid']for r in json.loads((OUT/'journal.json').read_text()).get('live',[])}
        memory=proc_rss(os.getpid())+sum(proc_rss(pid)for pid in owned)
        telemetry['maximum_aggregate_RSS_bytes']=max(telemetry['maximum_aggregate_RSS_bytes'],memory)
        if memory>p['aggregate_RSS_cap_bytes']:raise MemoryError('Aggregate12GiB RSS cap')
        if owned:
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5);total=0
            for line in q.stdout.splitlines():
                pid,memory,uuid=[x.strip()for x in line.split(',')]
                if int(pid)in owned:
                    if uuid!=UUID:raise RuntimeError('Owned GPU device mismatch')
                    total+=int(memory)*1024**2
            telemetry['maximum_owned_GPU_VRAM_bytes']=max(telemetry['maximum_owned_GPU_VRAM_bytes'],total)
            if total>p['owned_physical_GPU_VRAM_cap_bytes']:raise MemoryError('Shared GPU1 VRAM cap4GiB')
    started=time.monotonic();records=owned_jobs(lambda index,slot:command(index,digest),range(3),3,OUT,worker_wall_s=500,global_wall_s=600,RSS_bytes=4*1024**3,controller_check=check)
    rows=[json.loads((OUT/f'worker-{i}-result.json').read_text())for i in range(3)]
    elapsed=time.monotonic()-started
    result=dict(schema='surface-amplitude-concurrency-v01',completed=True,recipe_SHA256=digest,workers=rows,
        all_qualified=all(r['qualified']for r in rows),concurrent_wall_s=elapsed,posteriors_per_wall_hour=3*3600/elapsed,
        records=records,telemetry=telemetry,controller_pid=os.getpid(),all_reaped=True,no_actual_cohort=True)
    write(OUT/'result.json',result);return dict(completed=True,all_qualified=result['all_qualified'],wall_s=elapsed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','worker','guard']);p.add_argument('--digest');p.add_argument('--slot',type=int);a=p.parse_args()
    if a.mode=='prepare':print(prepare())
    elif a.mode=='run':print(json.dumps(run(a.digest)))
    elif a.mode=='guard':guard(a.slot);print(json.dumps(dict(guard_passed=True,slot=a.slot)))
    else:worker(a.slot,a.digest)
