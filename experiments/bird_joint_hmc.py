"""Observed-only joint-HMC diagnostic; not the production comparison worker.

Supply a previously saved observed-data reference. No future particle positions
or held-out images are scored. The full physical posterior is unchanged.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pacnerf_floor import make_floor_model
from pacnerf_conditioned_transport import make_target
from compare_pacnerf_discrepancy import file_hash,replay
from physgauss.checkpoint_hmc import sample_in_chunks
from physgauss.split_hmc import SplitHMCConfig
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.cli import source_fingerprint


def run(args):
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    settings={'case_index':8,'seed':590009,'warmup':100,'duration':[1.,4.],
        'source_fingerprint':source_fingerprint(),'reference_sha256':hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((Path(args.inputs)/'bird-inputs.pt').read_bytes()).hexdigest(),
        'inference_source_sha256_lf':{p.name:file_hash(p) for p in (ROOT/'scripts').glob('*.py')},
        'future_geometry_used':False,'qualification':'Observed-only sampler development; no predictive uncertainty claim'}
    protocol=out/'development_protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=settings:raise ValueError('Frozen development inputs/source changed')
    elif any(out.glob('*.pt')):raise ValueError('Untracked existing checkpoints; use a fresh output folder')
    protocol.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    if not (out/'reference.pt').exists():shutil.copyfile(args.reference,out/'reference.pt')
    model,_=make_floor_model(args.inputs,8)
    reference=torch.load(args.reference,weights_only=True);target=make_target(model,reference)
    path=out/'chains.pt';previous=torch.load(path,weights_only=True) if path.exists() else None
    retained=0 if previous is None else previous['white'].shape[1]
    if args.draws<retained:raise ValueError('Cannot shrink saved chains')
    count=args.draws-retained;start=time.time()
    if 0<count<16:raise ValueError('A continuation needs at least16 additional draws')
    config=SplitHMCConfig(warmup=100 if previous is None else 0,draws=count,
        min_time=1.,max_time=4.,target_accept=.9,seed=590009,
        step_size=.3 if previous is None else previous['step_size'])
    chains=sample_in_chunks(target,model.dimension,config,out,50,initial=reference.get('initial_white'),
        resume=previous,progress=lambda row:print(row,flush=True)) if count else previous
    if chains is None:raise ValueError('No samples')
    white=chains['white'];physical=target.decode(white.flatten(0,1))
    native=target.coordinates(white.flatten(0,1))
    values=torch.cat((physical,native),-1).reshape(4,args.draws,-1)
    diagnostics=diagnose_quantities(values,[f'z{i}' for i in range(values.shape[-1])])
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n',encoding='utf-8')
    result={'draws_per_chain':args.draws,'elapsed_this_invocation_s':time.time()-start,
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostics['quantities'].values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostics['quantities'].values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in diagnostics['quantities'].values()),
        'replay':replay(model,target,chains),'valid_for_ranking':False,
        'qualification':'Development only; final production gate additionally monitors conditional forces and density normalizers, uses longer warmup and independently validates raw predictions'}
    (out/'development_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','reference','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--draws',type=int,default=256)
    run(parser.parse_args())
