"""One retrospective failed-prefix correctness/cost control, separate namespace.

Replays only the saved initial observation of object02 matching_linear. No
held-out truth, physical parameters, probe observations or policy scores are
read. A new engine may qualify this control; the frozen original stays failed.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
import numpy as np
from experiments import diagnose_surface_cdf_v01 as diagnostic

ROOT=diagnostic.ROOT
OUT=ROOT/'runs/operator-dev/surface-positive-cdf-control-v01'
INPUT=diagnostic.INPUT
sha=diagnostic.sha
write=diagnostic.write


def sources():
    names=('experiments/campaign_positive_amplitude_pwl_v01.py',
           'experiments/campaign_surface_conditional_amplitude_v02.py',
           'experiments/benchmark_surface_positive_cdf_v01.py',
           'tests/test_positive_amplitude_pwl_v01.py',
           'tests/test_surface_conditional_amplitude_v02.py',
           'tests/test_surface_grouped_moments_v02.py',
           'tests/test_surface_positive_cdf_control_v01.py')
    return dict(diagnostic.sources(),**{name:sha(ROOT/name) for name in names})


def prepare():
    diagnostic.old_closed()
    if (OUT/'protocol.json').exists():raise ValueError('No control refreeze/retry')
    paths=[INPUT/'truth-LOCAL.npz',INPUT/'truth-checkpoint-LOCAL.json',
           INPUT/'matching_linear-prefix-LOCAL.json',
           diagnostic.old.LOCAL/'controlled-stop-ownership.json',
           diagnostic.OUT/'result-LOCAL.json',diagnostic.OUT/'owned-receipt.json']
    if json.loads(paths[2].read_text())['qualified'] is not False:
        raise ValueError('Original failed prefix must remain failed')
    p=dict(schema='surface-positive-cdf-control-v01',object_index=2,law='matching_linear',
           scope='Retrospective new-engine correctness/cost control only; original failure unchanged',
           panels=[128,256,512],outer=[128,256,512],CPU_worker=17,CPU_controller=16,
           GPU=False,threads=1,RSS_bytes=4*1024**3,worker_wall_s=900,global_wall_s=900,
           prefix_only=True,no_heldout_truth_reads=True,no_policy_or_material_score=True,
           all_original_accuracy_gates_unchanged=True,
           source_SHA256=sources(),input_SHA256={str(p.relative_to(ROOT)):sha(p)for p in paths})
    write(OUT/'protocol.json',p)
    return sha(OUT/'protocol.json')


def validate(digest):
    if sha(OUT/'protocol.json')!=digest:raise ValueError('Changed protocol')
    p=json.loads((OUT/'protocol.json').read_text())
    if p['source_SHA256']!=sources():raise ValueError('Changed source')
    if any(sha(ROOT/name)!=h for name,h in p['input_SHA256'].items()):raise ValueError('Changed input')
    diagnostic.old_closed()
    return p


def save_arrays(final,path):
    arrays={key:value for key,value in final.items() if isinstance(value,np.ndarray)}
    if not arrays:raise ValueError('Retained final arrays required for cost measurement')
    np.savez_compressed(path,**arrays)
    return dict(compressed_bytes=path.stat().st_size,
                array_bytes=sum(value.nbytes for value in arrays.values()),
                arrays=sorted(arrays))


def worker(digest):
    p=validate(digest);diagnostic.guard();started=time.monotonic()
    old=diagnostic.old
    meta=json.loads((INPUT/'truth-checkpoint-LOCAL.json').read_text())
    row=meta['labels'].index(diagnostic.PREFIX)
    observation=diagnostic.selected_npy_row(INPUT/'truth-LOCAL.npz','matching_linear.npy',row)
    observation+=diagnostic.selected_npy_row(INPUT/'truth-LOCAL.npz','prefix_noise.npy')
    original=json.loads((INPUT/'matching_linear-prefix-LOCAL.json').read_text())
    if old.observation_fingerprint([(diagnostic.PREFIX,observation)])!=original['observations_SHA256']:
        raise ValueError('Original acquired prefix fingerprint mismatch')
    build=time.monotonic();_,model=old.make_model(device_mode='cpu')
    fingerprint=diagnostic.base.model_fingerprint(model)
    if fingerprint!=original['model_fingerprint']:raise ValueError('Original finite model fingerprint mismatch')
    build_s=time.monotonic()-build
    ratio=diagnostic.RatioModal(model.modal,model.labels)
    callback=diagnostic.ExactRatioMemo(lambda z,l:ratio(np.exp(z),l),model_fingerprint=fingerprint,maximum_bytes=256*1024**2)
    from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02
    engine=ConditionalAmplitudeReferenceV02(None,[(diagnostic.PREFIX,observation)],old.study.TARGETS,
        log_ratio_predict=callback,retain_hypotheses=True)
    inference=time.monotonic();result=engine.qualify();inference_s=time.monotonic()-inference
    final=result.pop('local_final')
    cost=None if final is None else save_arrays(final,OUT/'posterior-LOCAL.npz')
    summary=dict(completed=True,qualified=result['qualified'],original_prefix_qualified=False,
        protocol_SHA256=digest,model_build_s=build_s,inference_s=inference_s,
        total_wall_s=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        diagnostics=result['diagnostics'],storage=cost,
        projected_32_prefix_archive_bytes=None if cost is None else 32*cost['compressed_bytes'],
        prefix_observation_replay_verified=True,same_original_model_verified=True,
        no_heldout_truth_reads=True,no_policy_score=True,raw_arrays_LOCAL=True)
    write(OUT/'result-LOCAL.json',summary)
    print(json.dumps(dict(qualified=result['qualified'],inference_s=inference_s,storage=cost)))


def command(digest,guard_only=False):
    return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
            'MKL_NUM_THREADS=1','taskset','-c','17',sys.executable,'-m',
            'experiments.benchmark_surface_positive_cdf_v01',
            'guard' if guard_only else 'worker','--digest',digest]


def run(digest):
    p=validate(digest)
    if os.sched_getaffinity(0)!={16}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('ControllerCPU16/noGPU')
    if (OUT/'journal.json').exists():raise ValueError('No implicit control relaunch')
    from experiments.run_campaign_surrogate_sbc_v03 import owned_jobs
    from experiments.campaign_owned_proc import proc_rss
    def check():
        path=OUT/'journal.json';live=[] if not path.exists() else json.loads(path.read_text()).get('live',[])
        if proc_rss(os.getpid())+sum(proc_rss(v['pid'])for v in live)>p['RSS_bytes']:
            raise MemoryError('Aggregate4GiB cap')
    records=owned_jobs(lambda i,s:command(digest),[0],1,OUT,
        worker_wall_s=p['worker_wall_s'],global_wall_s=p['global_wall_s'],
        RSS_bytes=p['RSS_bytes'],controller_check=check)
    write(OUT/'owned-receipt.json',dict(completed=True,all_reaped=True,
        controller_pid=os.getpid(),records=records,protocol_SHA256=digest))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','guard'])
    a.add_argument('--digest');x=a.parse_args()
    if x.mode=='prepare':print(prepare())
    elif x.mode=='run':run(x.digest)
    elif x.mode=='guard':diagnostic.guard();print(json.dumps(dict(guard_passed=True)))
    else:worker(x.digest)
