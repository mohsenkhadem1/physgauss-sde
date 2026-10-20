"""Observed-only saved-chain audit: drift, path disagreement and energy barriers."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from pacnerf_recovery import restore_correlations
from physgauss.chain_diagnostics import scalar_diagnostics


@torch.no_grad()
def audit(args):
    if args.worker:
        worker=Path(args.worker).resolve()
        sys.path.insert(0,str(worker/'src'));sys.path.insert(0,str(worker/'scripts'))
    from compare_pacnerf_discrepancy import make_inference_model,file_hash,replay
    from pacnerf_conditioned_transport import make_target
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    states=Path(args.states);results=Path(args.results);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    settings=json.loads((results/'settings.json').read_text())
    for name,digest in settings['inference_files_sha256_lf'].items():
        if file_hash(Path(__file__).parent/name)!=digest:raise ValueError('Audit source differs: '+name)
    model,_=make_inference_model(args.inputs,8)
    restored=restore_correlations(model,settings['model'])
    reference=torch.load(states/'reference.pt',weights_only=True)
    chains=torch.load(states/'chains.pt',weights_only=True);w=chains['white'];c,n,d=w.shape
    target=make_target(model,reference);proof=replay(model,target,chains)
    native=target.coordinates(w.flatten(0,1)).reshape(c,n,d)
    print('replay',proof,flush=True)
    keys=[0,1,d-3,d-2,d-1]
    parameter_names=['stiffness','damping','diffusion_h','smooth_load_h','rgb_h']
    parameters={}
    for key,name in zip(keys,parameter_names):
        v=native[:,:,key]
        parameters[name]={'diagnostics':scalar_diagnostics(v),'chain_means':v.mean(1).tolist(),
            'chain_standard_deviations':v.std(1).tolist(),
            'first_third_means':v[:,:n//3].mean(1).tolist(),'last_third_means':v[:,-n//3:].mean(1).tolist()}
    stride=max(1,n//670);selected=w[:,::stride];retained=selected.shape[1]
    physical=target.decode(selected.flatten(0,1)).reshape(c,retained,d)
    q,_=model.decode(physical.flatten(0,1));q=q.reshape(c,retained,14,model.modes)
    # A saved actual sample nearest each chain's native mean, not an averaged
    # trajectory passed off as a posterior sample.
    sd=native.flatten(0,1).std(0).clamp_min(1e-8)
    representatives=[];indices=[]
    for chain in range(c):
        distance=((native[chain]-native[chain].mean(0))/sd).square().sum(-1)
        index=int(distance.argmin());indices.append(index);representatives.append(w[chain,index])
    representatives=torch.stack(representatives);z=target.decode(representatives)
    heights=model.floor_heights(physical.flatten(0,1))
    rgb=model.renderer(model.decode(z)[0][:,model.frames],model.specs)
    energy=[]
    for a in range(c):
        for b in range(a+1,c):
            t=torch.linspace(0,1,25,dtype=w.dtype)
            path=(1-t[:,None])*z[a]+t[:,None]*z[b]
            dynamics=[];images=[];floor=[]
            for batch in path.split(8):
                dynamics.append(model.dynamics(batch))
                images.append(.5*model.image_residual(batch).square().sum(-1)+model.image_normalizer(batch))
                floor.append(model.floor_penalty(batch))
            terms=torch.stack([torch.cat(x) for x in [dynamics,images,floor]],-1)
            total=terms.sum(-1);baseline=(1-t)*total[0]+t*total[-1]
            energy.append({'chains':[a,b],'interpolation':t.tolist(),'energy_terms':terms.tolist(),
                'maximum_excess_above_endpoint_interpolation_nats':float((total-baseline).max())})
    log_density=chains['log_ratio']-.5*w.square().sum(-1)-target.center
    old=json.loads((results/'diagnostics.json').read_text())
    worst=sorted(old['quantities'].items(),key=lambda item:item[1]['rank_folded_split_rhat'] or 999,reverse=True)[:15]
    report={'draws_per_chain':n,'case':settings['case'],'future_geometry_used':False,
        'saved_source_files_sha256_lf':settings['inference_files_sha256_lf'],
        'checkpoint_sha256':hashlib.sha256((states/'chains.pt').read_bytes()).hexdigest(),
        'correlation_recovery':restored,'cached_target_replay':proof,'parameters':parameters,
        'log_density_diagnostics':scalar_diagnostics(log_density),'worst_last_completed_diagnostics':worst,
        'representative_draw_indices':indices,'energy_columns':['dynamics_and_prior','image_likelihood','floor_event'],
        'physical_interpolation_barriers':energy,
        'minimum_height_quantiles_m':torch.quantile(heights.amin((1,2)),heights.new_tensor([0,.01,.5,.99,1])).tolist(),
        'path_chain_means':q.mean(1).tolist(),'path_chain_standard_deviations':q.std(1).tolist(),
        'retained_counts':chains['counts']['retained'],'step_size':chains['step_size'],
        'qualification':'Observed-data inference audit only; interpolation barriers are path-specific evidence, not proof of global disconnected modes; no withheld truth or uncertainty calibration claim'}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    torch.save({'physical':physical,'selected_white':selected,'representatives':representatives},states/'mixing-audit-selected.pt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,8))
    for ax,key,name in zip(axes.flat,keys+[None],parameter_names+['log posterior']):
        values=log_density if key is None else native[:,:,key]
        for chain in range(c):ax.plot(np.arange(n),values[chain].numpy(),lw=.6,alpha=.75,label=f'chain {chain+1}')
        ax.set_title(name);ax.set_xlabel('retained draw')
    axes[0,0].legend(ncol=2);fig.suptitle('Bird: observed-only chain audit (not converged)')
    fig.tight_layout();fig.savefig(out/'chain-traces.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,len(model.frames),figsize=(10,12))
    for col,(frame,camera) in enumerate(model.specs):
        axes[0,col].imshow(model.obs[col].numpy());axes[0,col].set_title(f'observed t={frame}, camera={camera}')
        for chain in range(c):axes[chain+1,col].imshow(rgb[chain,col].clamp(0,1).numpy())
    for row in range(5):
        for col in range(len(model.frames)):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel('observed' if row==0 else f'chain {row}')
    fig.suptitle('Actual chain samples versus available images; no withheld images')
    fig.tight_layout();fig.savefig(out/'observed-reprojections.png',dpi=140);plt.close(fig)
    print(json.dumps({'parameters':parameters,'barriers':[e['maximum_excess_above_endpoint_interpolation_nats'] for e in energy],
        'worst_diagnostics':worst[:5],'raw_replay':proof}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',help='Frozen worker checkout matching saved inference source hashes')
    for name in ['inputs','states','results','output']:p.add_argument('--'+name,required=True)
    audit(p.parse_args())
