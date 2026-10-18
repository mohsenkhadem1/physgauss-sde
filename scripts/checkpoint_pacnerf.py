"""Run the original PAC worker with recoverable warmup and retained draws.

The source worker, model, adapted kernel and RNG stream are unchanged. Splitting
calls adds only deterministic gradient replays at recovery boundaries. Original
settings are retained; this separate wrapper has its own recorded SHA256.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import torch
import pacnerf_comparison as worker
from physgauss.cli import source_fingerprint

FINGERPRINT='eb703c41c125e971edae446bcad13d8f80a0ca5a34efb9f6a7b156762bf9f655'


from physgauss.checkpoint_hmc import sample_with_warmup_checkpoints,sample_in_chunks


def canonical_model_records(model,bundle):
    """Normalize JSON container types without changing any numerical tensors.

    The original worker compares in-memory view tuples to lists loaded from
    JSON, including nested bootstrap views. Its original source/hash and all
    substantive settings checks remain intact through this adapter.
    """
    model.specs=json.loads(json.dumps(model.specs,allow_nan=False))
    bundle=dict(bundle,static=json.loads(json.dumps(bundle['static'],allow_nan=False)))
    return model,bundle


class PreflightComplete(Exception):pass


def run(inputs,output,index,chunk,preflight=False,pilot=False):
    if preflight and pilot:raise ValueError('Preflight and development pilot are separate modes')
    if source_fingerprint()!=FINGERPRINT:raise ValueError('Numerical source differs from the frozen experiment')
    scene,views,noise,force=worker.cases()[index]
    checkpoint=Path(output)/f'{scene}__{views}views__{noise}__{force}'
    metadata=checkpoint/'checkpoint_worker.json'
    previous=json.loads(metadata.read_text()).get('invocations',[]) if metadata.exists() else []
    previous.append({
        'wrapper_sha256_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'base_worker_sha256_lf':hashlib.sha256(Path(worker.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'source_fingerprint':source_fingerprint(),'maximum_retained_chunk':chunk,
        'warmup_checkpoint_interval':25,
        'actual_environment':worker._environment(),
        'settings_container_normalization':'JSON container types only for observed_pairs and bootstrap; original substantive checks unchanged',
        'preflight_additional_draws':16 if preflight else 0,
        'development_pilot':pilot,
        'github_run_id':os.environ.get('GITHUB_RUN_ID'),
        'qualification':'original worker settings/target/kernel preserved; extra checkpoint boundaries add gradient replays only'})
    worker.write(metadata,{'invocations':previous})
    original_make=worker.make_model;original_sample=worker.sample_split_hmc
    def portable_model(*args,**kwargs):return canonical_model_records(*original_make(*args,**kwargs))
    def checkpointed(evaluate,dimension,config,initial=None,progress=None,resume=None):
        if preflight:
            if resume is None or config.warmup or config.draws<16:raise ValueError('Preflight needs a retained prefix and >=16 requested additional draws')
            result=sample_in_chunks(evaluate,dimension,replace(config,draws=16),checkpoint,chunk,initial,progress,resume)
            worker.write(checkpoint/'resume_preflight.json',{'additional_draws':16,
                'original_draws_per_chain':resume['white'].shape[1],
                'saved_draws_per_chain':result['white'].shape[1],
                'frozen_step_size':result['step_size'],'warmup_added':0,
                'qualification':'exact next sixteen retained transitions; preserved prefix/RNG; no reconstruction scoring or final eligibility claim'})
            raise PreflightComplete()
        return sample_in_chunks(evaluate,dimension,config,checkpoint,chunk,initial,progress,resume)
    worker.make_model=portable_model;worker.sample_split_hmc=checkpointed
    try:worker.run(inputs,output,index,pilot=pilot)
    except PreflightComplete:pass
    finally:worker.make_model=original_make;worker.sample_split_hmc=original_sample


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/pacnerf-inputs');p.add_argument('--output',default='runs/pacnerf-comparison')
    p.add_argument('--case-index',type=int,required=True);p.add_argument('--chunk-size',type=int,default=100)
    p.add_argument('--preflight',action='store_true',help='Save only the next sixteen retained draws, then stop without scoring')
    p.add_argument('--pilot',action='store_true',help='Observed-data-only 100-warmup/100-draw development run')
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(a.inputs,a.output,a.case_index,a.chunk_size,a.preflight,a.pilot)
