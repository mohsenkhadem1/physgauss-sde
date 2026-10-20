"""Observed-only Bird trial with exact logistic-slice reflective state HMC."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from pacnerf_floor import make_floor_model
from pacnerf_recovery import restore_correlations
from pacnerf_conditioned_transport import make_target
from pacnerf_reflected_hybrid import sample,ReflectedStateContext
from physgauss.chain_diagnostics import diagnose_quantities


def run(args):
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path(args.settings).read_text());model,_=make_floor_model(args.inputs,8)
    restored=restore_correlations(model,original['model'])
    reference=torch.load(args.reference,weights_only=True)
    protocol={'case_index':8,'warmup':args.warmup,'seed':args.seed,
        'input_sha256':hashlib.sha256((Path(args.inputs)/'bird-inputs.pt').read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        'runner_sha256_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'source_sha256_lf':{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (ROOT/'scripts').glob('*.py')},
        'frozen_model':original['model'],'future_geometry_used':False,'valid_for_ranking':False}
    p=out/'development_protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Reflected development protocol changed')
    p.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    if not (out/'reference.pt').exists():
        from shutil import copyfile
        copyfile(args.reference,out/'reference.pt')
    target=make_target(model,reference);initial=reference['initial_white']
    context=ReflectedStateContext(target,initial,model.core_dimension)
    normal,offset=context.geometry();physical=target.decode(initial)
    error=float((offset+torch.einsum('bkd,bd->bk',normal,initial[:,2:model.core_dimension])-model.floor_heights(physical).flatten(1)).abs().max())
    value,gradient,bad=context(initial[:,2:model.core_dimension])
    marginal,invalid=target.log_ratio_no_grad(initial)
    identity=float((value-marginal-model.floor_penalty(physical)).abs().max())
    if error>1e-9 or identity>1e-7 or bool((bad|invalid).any()):raise ValueError('Actual Bird residual/constraint preflight failed')
    preflight={'maximum_height_mapping_error_m':error,'maximum_residual_density_identity_error':identity,
        'floor_penalty':model.floor_penalty(physical).tolist(),'maximum_residual_gradient':float(gradient.abs().max()),
        'constant_restoration':restored,'all_pass':True}
    (out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n',encoding='utf-8');print(preflight,flush=True)
    start=time.time()
    chains=sample(target,model.dimension,model.core_dimension,out,args.draws,warmup=args.warmup,seed=args.seed,
        initial=initial,progress=lambda row:print(row,flush=True))
    w=chains['white'];native=target.coordinates(w.flatten(0,1));physical=target.decode(w.flatten(0,1))
    log_density=chains['log_ratio']-.5*w.square().sum(-1)-target.center
    values=torch.cat((physical.reshape(4,args.draws,-1),native.reshape(4,args.draws,-1),log_density[...,None]),-1)
    names=[f'physical_{i}' for i in range(model.dimension)]+[f'native_{i}' for i in range(model.dimension)]+['log_density']
    diagnostics=diagnose_quantities(values,names)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n',encoding='utf-8')
    heights=model.floor_heights(physical)
    report={'draws_per_chain':args.draws,'elapsed_this_invocation_s':time.time()-start,
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostics['quantities'].values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostics['quantities'].values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in diagnostics['quantities'].values()),
        'floor_penetration_fraction_below_minus_2cm':float((heights<-.02).double().mean()),
        'step_size':chains['step_size'],'counts':chains['counts'],'future_geometry_used':False,'valid_for_ranking':False,
        'qualification':'Observed-only development with discarded parent-chain reference; final production additionally checks force/normalizer diagnostics and independently replays uncertainty predictions'}
    (out/'development_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','reference','settings','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--draws',type=int,default=256);p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--seed',type=int,default=940009)
    run(p.parse_args())
