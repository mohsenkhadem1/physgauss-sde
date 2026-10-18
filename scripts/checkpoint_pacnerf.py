"""Run the original PAC worker with finer retained-draw checkpoints.

The source worker, model, adapted kernel and RNG stream are unchanged. Splitting
calls adds only a deterministic gradient replay at chunk boundaries. Original
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
from physgauss.split_hmc import sample_split_hmc
from physgauss.cli import source_fingerprint

FINGERPRINT='052a4cb2013105f9880912d3ab71b009dd9d65140a2ff44d2c30cab43d1ae1f4'


def sample_in_chunks(evaluate,dimension,config,checkpoint,max_chunk=100,initial=None,progress=None,resume=None):
    if max_chunk<32:raise ValueError('Checkpoint chunks must allow at least 32 retained draws')
    if config.draws<16:raise ValueError('Need at least sixteen new draws')
    checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True)
    remaining=config.draws;current=resume
    requested_total=(0 if resume is None else resume['white'].shape[1])+remaining
    while remaining:
        count=min(16 if current is None else max_chunk,remaining)
        if 0<remaining-count<16:
            count=remaining if remaining<=max_chunk else remaining-16
        call=replace(config,draws=count,warmup=config.warmup if current is None else 0,
            step_size=config.step_size if current is None else current['step_size'])
        current=sample_split_hmc(evaluate,dimension,call,
            initial=initial if current is None else None,progress=progress,resume=current)
        temporary=checkpoint/'chains.tmp.pt'
        torch.save(current,temporary);temporary.replace(checkpoint/'chains.pt')
        worker.write(checkpoint/'checkpoint_progress.json',{
            'retained_draws_per_chain':current['white'].shape[1],
            'requested_current_stage_total':requested_total,'frozen_step_size':current['step_size'],
            'gradient_batches':current['gradient_evaluations'],
            'qualification':'same frozen kernel/RNG; original diagnostic budgets and stopping gates unchanged'})
        remaining-=count
    return current


def run(inputs,output,index,chunk):
    if source_fingerprint()!=FINGERPRINT:raise ValueError('Numerical source differs from the frozen experiment')
    scene,views,noise,force=worker.cases()[index]
    checkpoint=Path(output)/f'{scene}__{views}views__{noise}__{force}'
    metadata=checkpoint/'checkpoint_worker.json'
    previous=json.loads(metadata.read_text()).get('invocations',[]) if metadata.exists() else []
    previous.append({
        'wrapper_sha256_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'base_worker_sha256_lf':hashlib.sha256(Path(worker.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'source_fingerprint':source_fingerprint(),'maximum_retained_chunk':chunk,
        'github_run_id':os.environ.get('GITHUB_RUN_ID'),
        'qualification':'original worker settings/target/kernel preserved; extra checkpoint boundaries add gradient replays only'})
    worker.write(metadata,{'invocations':previous})
    def checkpointed(evaluate,dimension,config,initial=None,progress=None,resume=None):
        return sample_in_chunks(evaluate,dimension,config,checkpoint,chunk,initial,progress,resume)
    worker.sample_split_hmc=checkpointed
    worker.run(inputs,output,index,pilot=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/pacnerf-inputs');p.add_argument('--output',default='runs/pacnerf-comparison')
    p.add_argument('--case-index',type=int,required=True);p.add_argument('--chunk-size',type=int,default=100)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(a.inputs,a.output,a.case_index,a.chunk_size)
