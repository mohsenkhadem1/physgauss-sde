"""Observed-only paired short/long state-kernel movement probes at saved states."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import torch
from bird_local import load_model,make_target,DeviceRenderer,write,sha
from pacnerf_reflected_hybrid import ReflectedStateContext
from pacnerf_reflections import state_transition


@torch.no_grad()
def run(args):
    out=Path(args.output)
    if out.exists():raise ValueError('Use a fresh probe directory')
    paths=[Path(args.shards)/f'chain-{i}'/'chains.pt' for i in range(4)]
    payloads=[p.read_bytes() for p in paths]
    states=[torch.load(io.BytesIO(b),weights_only=True) for b in payloads]
    initial=torch.cat([s['position'] for s in states])
    model=load_model(args.inputs,json.loads(Path(args.gate).read_text()))
    reference=torch.load(Path(args.shards)/'chain-0/reference.pt',weights_only=True)
    target=make_target(model,reference)
    if args.device!='cpu':
        model.renderer=DeviceRenderer(model.renderer,args.device);model._image_cache={}
    context=ReflectedStateContext(target,initial,model.core_dimension)
    normal,offset=context.geometry();before=target.decode(initial)
    rows=[]
    for seed in [1500001,1500002]:
        for duration in [2.,8.,20.]:
            rng=torch.Generator().manual_seed(seed);began=time.perf_counter()
            steps=round(duration/.06)
            candidate,info=state_transition(initial[:,2:model.core_dimension],context,normal,offset,.002,.06,steps,rng)
            white=initial.clone();white[:,2:model.core_dimension]=candidate
            after=target.decode(white)
            indices=[67,230,236,241]
            rows.append({'seed':seed,'duration':duration,'epsilon':.06,'steps':steps,
                'elapsed_s':time.perf_counter()-began,'accepted':info['accepted'].tolist(),
                'invalid':info['invalid'].tolist(),'divergent':info['divergent'].tolist(),
                'energy_error':info['energy_error'].tolist(),
                'white_squared_jump':(white-initial).square().sum(-1).tolist(),
                'physical_coordinate_changes':(after[:,indices]-before[:,indices]).tolist(),
                'physical_coordinate_indices':indices})
            print(json.dumps(rows[-1]),flush=True)
    report={'probes':rows,'source_checkpoint_sha256':[hashlib.sha256(b).hexdigest() for b in payloads],
        'source_retained_counts':[s['retained'] for s in states],'runner_sha256':sha(__file__),
        'future_geometry_used':False,'valid_for_ranking':False,
        'qualification':'Paired local movement and numerical-error probes, not stationary ESS or proof of better mixing. Probe samples are discarded.'}
    write(out/'probes.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/inputs');p.add_argument('--gate',default='runs/bird-8000/convergence.json')
    p.add_argument('--shards',default='runs/bird-local-production/shards')
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(args)
