"""Observed-only late covariance adaptation; all old states are discarded.

The physical target, initial geometry, force prior and guide stay fixed.
This is sampler development, never an eligible reconstruction/uncertainty run.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from pacnerf_recovery import restore_correlations
from pacnerf_floor import make_floor_model
from pacnerf_conditioned_transport import make_target
from pacnerf_warmup_reference import adapt_reference
from pacnerf_elliptical import sample,atomic_save
from physgauss.chain_diagnostics import diagnose_quantities


def run(args):
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    states=Path(args.states);model,_=make_floor_model(args.inputs,8)
    settings=json.loads(Path(args.settings).read_text());restore_correlations(model,settings['model'])
    protocol={'scene':'bird','case_index':8,'warmup':args.warmup,'seed':620009,'shrinkage':args.shrinkage,
        'training_checkpoint_sha256':hashlib.sha256((states/'chains.pt').read_bytes()).hexdigest(),
        'parent_reference_sha256':hashlib.sha256((states/'reference.pt').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (ROOT/'scripts').glob('*.py')},
        'block_size':args.block_size,'future_geometry_used':False,'valid_for_ranking':False}
    path=out/'development_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Development protocol changed')
    path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    if (out/'reference.pt').exists():reference=torch.load(out/'reference.pt',weights_only=True)
    else:
        parent=torch.load(states/'reference.pt',weights_only=True);training=torch.load(states/'chains.pt',weights_only=True)
        old=make_target(model,parent);reference=adapt_reference(parent,old,training,model.core_dimension,args.shrinkage)
        reference['development_training']=protocol;atomic_save(out/'reference.pt',reference)
        white=training['white'].flatten(0,1);variance=torch.linalg.eigvalsh(torch.cov(white.T))
        print('discarded covariance eigenvalue range',variance[[0,-1]].tolist(),flush=True)
    target=make_target(model,reference);start=time.time()
    blocks=[(i,min(i+args.block_size,model.dimension)) for i in range(0,model.dimension,args.block_size)]
    chains=sample(target.log_ratio_no_grad,model.dimension,out,args.draws,warmup=args.warmup,seed=620009,
        initial=reference['initial_white'],blocks=blocks,progress=lambda row:print(row,flush=True))
    w=chains['white'];physical=target.decode(w.flatten(0,1));native=target.coordinates(w.flatten(0,1))
    values=torch.cat((physical,native),-1).reshape(4,args.draws,-1)
    d=diagnose_quantities(values,[f'z{i}' for i in range(values.shape[-1])])
    (out/'diagnostics.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    report={'draws_per_chain':args.draws,'elapsed_this_invocation_s':time.time()-start,
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in d['quantities'].values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in d['quantities'].values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in d['quantities'].values()),
        'target_evaluations':chains['target_evaluations'],'future_geometry_used':False,'valid_for_ranking':False,
        'qualification':'All parent chains are discarded proposal-learning data; same physical target. Short diagnostic excludes force/log-density gates required for production.'}
    (out/'development_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','states','settings','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--draws',type=int,default=256);p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--shrinkage',type=float,default=.1);p.add_argument('--block-size',type=int,default=329)
    run(p.parse_args())
