"""Strict four-shard observed-only convergence gate, before uncertainty scoring."""
import argparse,hashlib,json
from pathlib import Path
import torch
from pacnerf_floor import make_floor_model
from pacnerf_recovery import restore_correlations
from pacnerf_conditioned_transport import make_target
from compare_pacnerf_discrepancy import replay,force_moments_stream
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.cli import source_fingerprint
from pacnerf_elliptical import atomic_save


@torch.no_grad()
def aggregate(inputs,shards,output,required_draws):
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    paths=[Path(shards)/f'chain-{i}' for i in range(4)]
    data=[torch.load(p/'chains.pt',weights_only=True) for p in paths]
    protocols=[json.loads((p/'development_protocol.json').read_text()) for p in paths]
    refhash={hashlib.sha256((p/'reference.pt').read_bytes()).hexdigest() for p in paths}
    if len(refhash)!=1:raise ValueError('Shards used different reference blobs')
    common=lambda p:{k:v for k,v in p.items() if k not in ['chain_index','seed']}
    if any(common(p)!=common(protocols[0]) for p in protocols):raise ValueError('Shard source/input/model protocols differ')
    if next(iter(refhash))!=protocols[0]['reference_sha256']:raise ValueError('Reference bytes differ from worker protocol')
    if [p['chain_index'] for p in protocols]!=list(range(4)) or len({p['seed'] for p in protocols})!=4:
        raise ValueError('Expected four different chain indices and independent seeds')
    for p in Path(__file__).parent.glob('*.py'):
        if hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=protocols[0]['source_sha256_lf'].get(p.name):
            raise ValueError('Replay source changed: '+p.name)
    actual=hashlib.sha256((Path(inputs)/'bird-inputs.pt').read_bytes()).hexdigest()
    if actual!=protocols[0]['input_sha256']:raise ValueError('Input differs')
    for i,d in enumerate(data):
        if d['white'].shape!=(1,required_draws,329) or d['retained']!=required_draws:raise ValueError('Incomplete chain '+str(i))
        if d['completed_warmup']!=protocols[i]['warmup'] or d['protocol']['seed']!=protocols[i]['seed']:raise ValueError('Warmup/seed differs')
        if d['protocol']['chains']!=1:raise ValueError('Expected one independently seeded chain per worker')
    model,_=make_floor_model(inputs,8);restored=restore_correlations(model,protocols[0]['frozen_model'])
    reference=torch.load(paths[0]/'reference.pt',weights_only=True)
    if reference['frozen_model_specification']!=model.specification():raise ValueError('Reference physical model differs')
    if reference['reference_kind']!='analytic-floor-free-state-gaussian-with-five-parameter-moments':raise ValueError('Wrong proposal reference')
    target=make_target(model,reference)
    for i,d in enumerate(data):
        expected=reference['initial_white'][i:i+1].contiguous().numpy().tobytes()
        if d['protocol']['initial_state_sha256']!=hashlib.sha256(expected).hexdigest():raise ValueError('Chain initial state differs')
    white=torch.cat([d['white'] for d in data]);ratios=torch.cat([d['log_ratio'] for d in data])
    proof=replay(model,target,{'white':white,'log_ratio':ratios})
    native=target.coordinates(white.flatten(0,1));physical=target.decode(white.flatten(0,1))
    fm,fc,scaled_fm,scaled_fc,normalizers=force_moments_stream(model,physical)
    log_density=ratios-.5*white.square().sum(-1)-target.center
    monitored=torch.cat((physical.reshape(4,required_draws,-1),native.reshape(4,required_draws,-1),
        log_density[...,None],fm.reshape(4,required_draws,-1),normalizers.reshape(4,required_draws,2)),-1)
    names=[f'physical_{i}' for i in range(model.dimension)]+[f'native_{i}' for i in range(model.dimension)]+['log_target']+[
        f'conditional_load_{i}' for i in range(model.force_dimension)]+['logdet_Q','logdet_A']
    diagnostics=diagnose_quantities(monitored,names)
    invalid=sum(sum(d['counts']['retained']['invalid']) for d in data)
    divergent=sum(sum(d['counts']['retained']['divergent']) for d in data)
    passed=required_draws>=2000 and protocols[0]['warmup']>=500 and diagnostics['all_monitored_quantities_pass'] and invalid==0 and divergent==0
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n',encoding='utf-8')
    report={'case_index':8,'source_fingerprint':source_fingerprint(),'draws_per_chain':required_draws,
        'warmup':protocols[0]['warmup'],'four_independent_chains':True,'all_monitored_quantities_pass':bool(passed),
        'retained_invalid':invalid,'retained_divergent':divergent,
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostics['quantities'].values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostics['quantities'].values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in diagnostics['quantities'].values()),
        'source_protocol':common(protocols[0]),'constant_replay':restored,'density_replay':proof,
        'reference_sha256':next(iter(refhash)),'future_geometry_used':False,'valid_for_ranking':False,
        'next_step':'Independent uncertainty validation required before comparison' if passed else 'Bird has not converged; wider comparison remains blocked'}
    (out/'convergence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    atomic_save(out/'merged.pt',{'white':white,'log_ratio':ratios,'chain_protocols':[d['protocol'] for d in data]})
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','shards','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--draws',type=int,required=True)
    args=p.parse_args();aggregate(args.inputs,args.shards,args.output,args.draws)
