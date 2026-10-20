"""Discarded observed-only conditional-transport pilot; no production eligibility."""
import argparse
import json
from pathlib import Path
import time
import torch
from bird_local import load_model,sha,write,gpu_check,source_hashes
from bird_conditional_bridge import conditional_target
from bird_dense_sampler import sample
from compare_pacnerf_discrepancy import replay,force_moments_stream
from physgauss.chain_diagnostics import diagnose_quantities


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path(args.gate).read_text());model=load_model(args.inputs,gate)
    reference=torch.load(args.reference,weights_only=True);target=conditional_target(model,reference)
    initial=reference['initial_white']
    if args.device!='cpu':
        renderer,proof=gpu_check(model,args.device,target.decode(initial))
        write(out/'device-validation.json',proof);print(json.dumps(proof),flush=True)
        model.renderer=renderer;model._image_cache={}
    protocol={'seed':args.seed,'warmup':args.warmup,'device':args.device,
        'source_sha256':{**source_hashes(),'experiments/bird_conditional_bridge.py':sha(Path(__file__).with_name('bird_conditional_bridge.py')),
            'experiments/trial_conditional_bird.py':sha(__file__)},
        'reference_sha256':sha(args.reference),'input_sha256':sha(Path(args.inputs)/'bird-inputs.pt'),
        'future_geometry_used':False,'valid_for_ranking':False,'four_chain_pilot_shared_warmup_adaptation':True}
    path=out/'development_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Pilot protocol changed')
    write(path,protocol)
    context=target.reflection_context(initial);normal,offset=context.geometry()
    physical=target.decode(initial)
    height=offset+torch.einsum('bkd,bd->bk',normal,initial[:,2:model.core_dimension])
    error=float((height-model.floor_heights(physical).flatten(1)).abs().max())
    if error>1e-9:raise ValueError('Actual conditional transport walls differ')
    began=time.time()
    chains=sample(target,model.dimension,model.core_dimension,out,args.draws,
        warmup=args.warmup,seed=args.seed,initial=initial,chains_count=4,
        interweave=True,progress=lambda row:print(json.dumps({'elapsed_s':time.time()-began,**row}),flush=True))
    proof=replay(model,target,chains);w=chains['white']
    native=target.coordinates(w.flatten(0,1));physical=target.decode(w.flatten(0,1))
    fm,_,_,_,norm=force_moments_stream(model,physical)
    density=chains['log_ratio']-.5*w.square().sum(-1)-target.center
    values=torch.cat((physical.reshape(4,args.draws,-1),native.reshape(4,args.draws,-1),
        density[...,None],fm.reshape(4,args.draws,-1),norm.reshape(4,args.draws,2)),-1)
    names=[f'physical_{i}' for i in range(model.dimension)]+[f'native_{i}' for i in range(model.dimension)]+['log_target']+[
        f'conditional_load_{i}' for i in range(model.force_dimension)]+['logdet_Q','logdet_A']
    diagnostics=diagnose_quantities(values,names);rows=diagnostics['quantities']
    result={'draws_per_chain':args.draws,'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in rows.values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in rows.values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in rows.values()),
        'failed_quantities':len(diagnostics['failed_quantities']),'density_replay':proof,
        'counts':chains['counts'],'maximum_height_mapping_error':error,
        'future_geometry_used':False,'valid_for_ranking':False,
        'qualification':'Short development pilot, shared warmup adaptation, no predictive scores; not a final convergence result'}
    write(out/'diagnostics.json',diagnostics);write(out/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/inputs');p.add_argument('--gate',default='runs/bird-8000/convergence.json')
    p.add_argument('--reference',default='runs/bird-conditional-reference/reference.pt')
    p.add_argument('--device',default='cuda:2');p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=128);p.add_argument('--draws',type=int,default=128)
    p.add_argument('--seed',type=int,default=1370009)
    args=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(args)
