"""Observed-only partial-chain diagnostics; never eligible as a full result."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import torch
from bird_local import load_model,make_target,source_hashes,write,label
from compare_pacnerf_discrepancy import force_moments_stream,replay
from physgauss.chain_diagnostics import diagnose_quantities
from pacnerf_elliptical import atomic_save


@torch.no_grad()
def run(args):
    out=Path(args.output)
    if out.exists():raise ValueError('Use a fresh preview directory')
    paths=[Path(args.shards)/f'chain-{i}' for i in range(4)]
    protocols=[json.loads((p/'local_protocol.json').read_text()) for p in paths]
    common=lambda p:{k:v for k,v in p.items() if k not in ['chain_index','seed','device']}
    if any(common(p)!=common(protocols[0]) for p in protocols):
        raise ValueError('Different shard protocols')
    if protocols[0]['source_sha256']!=source_hashes():
        raise ValueError('Frozen numerical source differs')
    payloads=[(p/'chains.pt').read_bytes() for p in paths]
    data=[torch.load(io.BytesIO(b),weights_only=True) for b in payloads]
    if any(d['retained']<args.draws for d in data):
        raise ValueError('Not all workers have completed the requested preview')
    white=torch.cat([d['white'][:,:args.draws] for d in data])
    ratios=torch.cat([d['log_ratio'][:,:args.draws] for d in data])
    gate=json.loads(Path(args.gate).read_text());model=load_model(args.inputs,gate)
    reference=torch.load(paths[0]/'reference.pt',weights_only=True);target=make_target(model,reference)
    proof=replay(model,target,{'white':white,'log_ratio':ratios})
    native=target.coordinates(white.flatten(0,1));physical=target.decode(white.flatten(0,1))
    fm,_,_,_,normalizers=force_moments_stream(model,physical)
    density=ratios-.5*white.square().sum(-1)-target.center
    values=torch.cat((physical.reshape(4,args.draws,-1),native.reshape(4,args.draws,-1),
        density[...,None],fm.reshape(4,args.draws,-1),normalizers.reshape(4,args.draws,2)),-1)
    names=[f'physical_{i}' for i in range(model.dimension)]+[f'native_{i}' for i in range(model.dimension)]+['log_target']+[
        f'conditional_load_{i}' for i in range(model.force_dimension)]+['logdet_Q','logdet_A']
    diagnostics=diagnose_quantities(values,names)
    rows=diagnostics['quantities']
    worst=sorted(rows,key=lambda k:rows[k]['rank_folded_split_rhat'] or 999,reverse=True)[:20]
    result={'preview_draws_per_chain':args.draws,'warmup':protocols[0]['warmup'],
        'failed_quantities':len(diagnostics['failed_quantities']),
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in rows.values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in rows.values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in rows.values()),
        'worst_quantities':[{'quantity':k,'meaning':label(k),**rows[k]} for k in worst],
        'density_replay':proof,'snapshot_retained_counts':[d['retained'] for d in data],
        'source_checkpoint_sha256':[hashlib.sha256(b).hexdigest() for b in payloads],
        'future_geometry_used':False,'valid_for_ranking':False,
        'qualification':'Observed-only prefix audit. Does not replace the full predeclared 2000/4000/8000-draw convergence gates.'}
    out.mkdir(parents=True)
    write(out/'preview.json',result);write(out/'diagnostics.json',diagnostics)
    atomic_save(out/'prefix.pt',{'white':white,'log_ratio':ratios,'source_checkpoint_sha256':result['source_checkpoint_sha256']})
    print(json.dumps({k:v for k,v in result.items() if k!='worst_quantities'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/inputs')
    p.add_argument('--gate',default='runs/bird-8000/convergence.json')
    p.add_argument('--shards',default='runs/bird-local-production/shards')
    p.add_argument('--draws',type=int,default=500)
    p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(args)
