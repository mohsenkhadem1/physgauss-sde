"""Three bounded replicas of one failed prefix on shared GPUs0-2.

User authorized sharing with unrelated training. Only owned process groups and
allocations are monitored; other jobs are never changed. GPU3 is excluded.
This is retrospective numerical throughput, not three independent objects.
"""
import argparse,json,os,resource,sys,time
from pathlib import Path
import numpy as np
from experiments import benchmark_surface_positive_cdf_v01 as cpu
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02

ROOT=cpu.ROOT;OUT=ROOT/'runs/operator-dev/surface-positive-cdf-GPU-control-v01'
old=cpu.diagnostic.old;sha=cpu.sha;write=cpu.write
UUIDS=tuple(old.GPU_UUIDS);CORES=(13,14,15)


def sources():
    return dict(cpu.sources(),**{name:sha(ROOT/name)for name in(
        'experiments/benchmark_surface_positive_cdf_gpu_v01.py',
        'tests/test_surface_positive_cdf_gpu_control_v01.py')})


def prepare():
    cpu.diagnostic.old_closed()
    if(OUT/'protocol.json').exists():raise ValueError('No refreeze/retry')
    if not json.loads((cpu.OUT/'result-LOCAL.json').read_text())['qualified']:
        raise ValueError('Qualified unchanged CPU failed-prefix control required')
    paths=[cpu.INPUT/'truth-LOCAL.npz',cpu.INPUT/'truth-checkpoint-LOCAL.json',
        cpu.INPUT/'matching_linear-prefix-LOCAL.json',cpu.OUT/'result-LOCAL.json',
        cpu.OUT/'owned-receipt.json',cpu.OUT/'posterior-LOCAL.npz']
    p=dict(schema='surface-positive-cdf-GPU-control-v01',workers=3,
        CPU_workers=list(CORES),CPU_controller=16,GPU_UUIDs=list(UUIDS),
        GPU3_forbidden=True,external_jobs_unchanged=True,threads=1,
        RSS_worker_bytes=4*1024**3,RSS_aggregate_bytes=12*1024**3,
        GPU_owned_VRAM_per_device_bytes=2*1024**3,Torch_allocated_worker_bytes=512*1024**2,
        worker_wall_s=600,global_wall_s=660,
        scope='Three replicas of same failedobj02 matchingprefix; retrospective precision/cost only',
        no_heldout_truth_reads=True,no_policy_score=True,no_new_objects_or_redraw=True,
        all_original_accuracy_gates_unchanged=True,panels=[128,256,512],outer=[128,256,512],
        CPU_GPU_parity=dict(target_mean_SD_absolute_m=1e-12,intrinsic_arrays_absolute=1e-8),
        source_SHA256=sources(),input_SHA256={str(p.relative_to(ROOT)):sha(p)for p in paths})
    write(OUT/'protocol.json',p);return sha(OUT/'protocol.json')


def validate(digest):
    if sha(OUT/'protocol.json')!=digest:raise ValueError('Changed protocol')
    p=json.loads((OUT/'protocol.json').read_text())
    if p['source_SHA256']!=sources()or any(sha(ROOT/n)!=h for n,h in p['input_SHA256'].items()):
        raise ValueError('Changed sources/inputs')
    cpu.diagnostic.old_closed();return p


def guard(slot):
    if slot not in range(3)or os.sched_getaffinity(0)!={CORES[slot]}or os.environ.get('CUDA_VISIBLE_DEVICES')!=UUIDS[slot]or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('One pinned CPU/core and physicalGPU0-2 required')


def worker(slot,digest):
    validate(digest);guard(slot);start=time.monotonic()
    meta=json.loads((cpu.INPUT/'truth-checkpoint-LOCAL.json').read_text())
    row=meta['labels'].index(cpu.diagnostic.PREFIX)
    y=cpu.diagnostic.selected_npy_row(cpu.INPUT/'truth-LOCAL.npz','matching_linear.npy',row)
    y+=cpu.diagnostic.selected_npy_row(cpu.INPUT/'truth-LOCAL.npz','prefix_noise.npy')
    original=json.loads((cpu.INPUT/'matching_linear-prefix-LOCAL.json').read_text())
    if old.observation_fingerprint([(cpu.diagnostic.PREFIX,y)])!=original['observations_SHA256']:
        raise ValueError('Original acquired prefix fingerprint mismatch')
    build=time.monotonic();_,model=old.make_model(device_mode='gpu',slot=slot)
    fingerprint=cpu.diagnostic.base.model_fingerprint(model)
    if fingerprint!=original['model_fingerprint']:raise ValueError('Original finite model fingerprint mismatch')
    build_s=time.monotonic()-build;callback=old.ratio_callback(model)
    begin=time.monotonic()
    result=ConditionalAmplitudeReferenceV02(None,[(cpu.diagnostic.PREFIX,y)],old.study.TARGETS,
        log_ratio_predict=callback,retain_hypotheses=False).qualify()
    final=result.pop('local_final');cost=None if final is None else cpu.save_arrays(final,OUT/f'worker-{slot}-posterior-LOCAL.npz')
    write(OUT/f'worker-{slot}-result-LOCAL.json',dict(completed=True,slot=slot,pid=os.getpid(),
        protocol_SHA256=digest,qualified=result['qualified'],diagnostics=result['diagnostics'],
        model_build_s=build_s,inference_s=time.monotonic()-begin,total_wall_s=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_Torch_allocated_bytes=model.torch.cuda.max_memory_allocated(model.device),
        storage=cost,original_prefix_qualified=False,no_heldout_truth_reads=True,
        no_policy_score=True,one_original_object_not_three_independent_objects=True))


def command(slot,digest,guard_only=False):
    if slot not in range(3):raise ValueError('Only GPUs0-2')
    return ['env','CUDA_VISIBLE_DEVICES='+UUIDS[slot],'CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c',
        str(CORES[slot]),sys.executable,'-m','experiments.benchmark_surface_positive_cdf_gpu_v01',
        'guard'if guard_only else'worker','--slot',str(slot),'--digest',digest]


def run(digest):
    p=validate(digest)
    if os.sched_getaffinity(0)!={16}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('ControllerCPU16/noGPU')
    if(OUT/'journal.json').exists():raise ValueError('No implicit relaunch')
    from experiments.run_campaign_surrogate_sbc_v03 import owned_jobs
    from experiments.campaign_owned_proc import proc_rss
    telemetry=dict(maximum_aggregate_RSS_bytes=0,maximum_owned_VRAM_bytes={u:0 for u in UUIDS})
    def check():
        path=OUT/'journal.json';live=[]if not path.exists()else json.loads(path.read_text()).get('live',[])
        rss=proc_rss(os.getpid())+sum(proc_rss(r['pid'])for r in live)
        telemetry['maximum_aggregate_RSS_bytes']=max(telemetry['maximum_aggregate_RSS_bytes'],rss)
        if rss>p['RSS_aggregate_bytes']:raise MemoryError('Aggregate12GiB cap')
        if live:
            usage=old.GPU_usage([r['pid']for r in live],cap_bytes=p['GPU_owned_VRAM_per_device_bytes'])
            for u,n in usage.items():telemetry['maximum_owned_VRAM_bytes'][u]=max(telemetry['maximum_owned_VRAM_bytes'][u],n)
    start=time.monotonic()
    records=owned_jobs(lambda index,slot:command(index,digest),range(3),3,OUT,
        worker_wall_s=p['worker_wall_s'],global_wall_s=p['global_wall_s'],
        RSS_bytes=p['RSS_worker_bytes'],controller_check=check)
    rows=[json.loads((OUT/f'worker-{i}-result-LOCAL.json').read_text())for i in range(3)]
    parity=[]
    with np.load(cpu.OUT/'posterior-LOCAL.npz')as reference:
        for i in range(3):
            values={}
            path=OUT/f'worker-{i}-posterior-LOCAL.npz'
            if not path.exists():parity.append(dict(slot=i,passed=False,reason='Missing final'));continue
            with np.load(path)as actual:
                for key in('predictive_mean','predictive_SD','unit_mean','unit_covariance','unit_CDF','unit_quantiles'):
                    values[key]=float(np.max(abs(actual[key]-reference[key])))
            passed=all(v<=(1e-12 if k.startswith('predictive_')else 1e-8)for k,v in values.items())
            parity.append(dict(slot=i,passed=passed,maximum_absolute_differences=values))
    write(OUT/'owned-result-LOCAL.json',dict(completed=True,all_reaped=True,controller_pid=os.getpid(),
        records=records,protocol_SHA256=digest,all_qualified=all(r['qualified']for r in rows)and all(r['passed']for r in parity),CPU_GPU_parity=parity,
        concurrent_wall_s=time.monotonic()-start,telemetry=telemetry,rows=rows))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','guard'])
    a.add_argument('--digest');a.add_argument('--slot',type=int);x=a.parse_args()
    if x.mode=='prepare':print(prepare())
    elif x.mode=='run':run(x.digest)
    elif x.mode=='guard':guard(x.slot);print(json.dumps(dict(guard_passed=True,slot=x.slot)))
    else:worker(x.slot,x.digest)
