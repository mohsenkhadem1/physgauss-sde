"""Observed-only geometry and path-barrier audit of already aggregated shards."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    root=Path(args.shards);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path(args.convergence).read_text())
    worker=Path(args.worker).resolve() if args.worker else ROOT
    hashes={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (worker/'scripts').glob('*.py')}
    if hashes!=gate['source_protocol']['source_sha256_lf']:raise ValueError('Audit math source differs from frozen worker')
    if hashlib.sha256((Path(args.inputs)/'bird-inputs.pt').read_bytes()).hexdigest()!=gate['source_protocol']['input_sha256']:
        raise ValueError('Audit input differs')
    sys.path.insert(0,str(worker/'src'));sys.path.insert(0,str(worker/'scripts'))
    from pacnerf_floor import make_floor_model
    from pacnerf_recovery import restore_correlations
    from pacnerf_conditioned_transport import make_target
    w=torch.load(Path(args.convergence).parent/'merged.pt',weights_only=True)['white'];n=w.shape[1]
    if n!=gate['draws_per_chain']:raise ValueError('Audit budget differs')
    model,_=make_floor_model(args.inputs,8);restore_correlations(model,gate['source_protocol']['frozen_model'])
    target=make_target(model,torch.load(root/'chain-0/reference.pt',weights_only=True))
    physical=target.decode(w.flatten(0,1)).reshape(4,n,model.dimension)
    state=w[:,:,2:model.core_dimension];cov=torch.cov(state.flatten(0,1).T);eigen,vectors=torch.linalg.eigh(cov)
    projection=state@vectors[:,-1]
    diagnostic=json.loads((Path(args.convergence).parent/'diagnostics.json').read_text())['quantities']
    worst=max((i for i in range(model.dimension)),key=lambda i:diagnostic[f'physical_{i}']['rank_folded_split_rhat'] or 999)
    # For this preview the worst index is236: q at frame10, elastic mode6.
    values=physical[:,:,worst]
    moments=lambda z:{'chain_means':z.mean(1).tolist(),'chain_sd':z.std(1).tolist(),
        'within_chain_rms_jump':(z[:,1:]-z[:,:-1]).square().mean(1).sqrt().tolist(),
        'first_third_means':z[:,:n//3].mean(1).tolist(),'last_third_means':z[:,-(n//3):].mean(1).tolist()}
    report={'draws_per_chain':n,'worst_physical_coordinate':{'index':worst,**moments(values)},
        'leading_native_state_pc':{'variance':float(eigen[-1]),**moments(projection)},
        'native_state_covariance_eigenvalue_range':[float(eigen[0]),float(eigen[-1])],
        'native_state_covariance_condition_ratio':float(eigen[-1]/eigen[0]),
        'native_state_mean_norm':float(state.flatten(0,1).mean(0).norm()),
        'qualification':'Exploratory unconverged moments; empirical geometry, not posterior estimates or a stationary autocorrelation estimate',
        'future_geometry_used':False,'valid_for_ranking':False}
    (out/'geometry.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    pooled_sd=physical.flatten(0,1).std(0).clamp_min(1e-8);indices=[];representatives=[]
    for chain in physical:
        index=int((((chain-chain.mean(0))/pooled_sd)**2).sum(-1).argmin());indices.append(index);representatives.append(chain[index])
    representatives=torch.stack(representatives);rows=[]
    for a in range(4):
        for b in range(a+1,4):
            t=torch.linspace(0,1,41,dtype=w.dtype);path=(1-t[:,None])*representatives[a]+t[:,None]*representatives[b]
            parts=[]
            for z in path.split(8):parts.append(torch.stack((model.dynamics(z),.5*model.image_residual(z).square().sum(-1)+model.image_normalizer(z),model.floor_penalty(z)),-1))
            parts=torch.cat(parts);energy=parts.sum(-1);excess=energy-((1-t)*energy[0]+t*energy[-1])
            rows.append({'chains':[a,b],'t':t.tolist(),'energy_terms':parts.tolist(),
                'maximum_excess_above_endpoint_chord_nats':float(excess.max()),
                'maximum_uphill_from_higher_endpoint_nats':float(energy.max()-energy[[0,-1]].max())})
    (out/'barriers.json').write_text(json.dumps({'representative_draw_indices':indices,'paths':rows,
        'energy_columns':['dynamics_and_prior','image_likelihood','floor_feasibility'],
        'qualification':'Straight physical paths only; no proof of unimodality, no exclusion of entropic barriers or unvisited modes',
        'future_geometry_used':False},indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for chain in range(4):
        axes[0].plot(np.arange(n),values[chain].numpy(),lw=.8,label=f'Chain {chain+1}')
        axes[1].plot(np.arange(n),projection[chain].numpy(),lw=.8)
    axes[0].set_title(f'Worst physical coordinate: {worst}');axes[0].set_ylabel('Standardized state coordinate')
    axes[1].set_title('Leading preview covariance direction');axes[1].set_ylabel('Native state projection')
    for ax in axes:ax.set_xlabel('Retained draw');ax.grid(alpha=.15)
    axes[0].legend(ncol=2,fontsize=8);fig.suptitle('Bird: remaining motion bottleneck — unconverged preview; no predictive scores')
    fig.savefig(out/'slow-mode-traces.png',dpi=150);plt.close(fig)
    print(json.dumps({'worst_physical_coordinate':worst,'covariance_eigenvalues':[float(eigen[0]),float(eigen[-1])],
        'path_barriers_nats':[row['maximum_excess_above_endpoint_chord_nats'] for row in rows]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['inputs','shards','convergence','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--worker',help='Checkout whose math hashes match the frozen shard protocol')
    run(p.parse_args())
