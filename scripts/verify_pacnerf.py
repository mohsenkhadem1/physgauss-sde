"""Replay raw production checkpoints and independently recompute position scores.

NumPy recomputes modal means, Gaussian positions, interval quantiles and gap
errors directly from the serialized states. The original image/physics target
checks cached HMC log ratios. This verifies numerical provenance and scoring,
not the physical truth of the reduced model or global posterior exploration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pacnerf_comparison import cases,make_model,Target,make_target
from physgauss.cli import source_fingerprint
from physgauss.chain_diagnostics import scalar_diagnostics
from pacnerf_prediction_metrics import prediction_scores


def locate(root, pattern):
    found=list(Path(root).glob(pattern))
    if len(found)!=1:raise ValueError(f'Expected one file for {pattern}, got {len(found)}')
    return found[0]


@torch.no_grad()
def verify_case(raw,inputs,index):
    scene,views,noise,force=cases()[index]
    name=f'{scene}__{views}views__{noise}__{force}'
    result=Path(raw)/f'results-{index}';chain_root=Path(raw)/f'chains-{index}'
    summary=json.loads(locate(result,f'**/{name}/summary.json').read_text())
    published=json.loads(locate(result,f'**/{name}/posterior.json').read_text())
    diagnostics=json.loads(locate(result,f'**/{name}/diagnostics.json').read_text())
    if summary['settings']['source_fingerprint']!=source_fingerprint():raise ValueError('Numerical source changed')
    manifest=json.loads((Path(inputs)/f'{scene}-manifest.json').read_text())
    for suffix,key in [('inputs','inputs_sha256'),('evaluation','evaluation_sha256')]:
        actual=hashlib.sha256((Path(inputs)/f'{scene}-{suffix}.pt').read_bytes()).hexdigest()
        if actual!=manifest[key]:raise ValueError('Prepared tensor artifact hash differs')
    if manifest['inputs_sha256']!=summary['settings']['input_sha256']:raise ValueError('Worker input hash differs')
    reference=torch.load(locate(chain_root,f'**/{name}/reference.pt'),weights_only=True)
    chains=torch.load(locate(chain_root,f'**/{name}/chains.pt'),weights_only=True)
    model,bundle=make_model(inputs,index)
    c,n,d=chains['white'].shape
    if c!=4 or n<2000:raise ValueError('Incomplete final sampling budget')
    chol=np.linalg.cholesky(reference['precision'].numpy())
    transform=np.linalg.solve(chol.T,np.eye(d))
    white=chains['white'].numpy()
    coordinates=reference['mean'].numpy()+white.reshape(-1,d)@transform.T
    if reference.get('coordinates','physical')=='innovation':
        from pacnerf_innovations import InnovationModel
        innovation=InnovationModel(model)
        coordinates=torch.cat([innovation.to_physical(batch) for batch in torch.from_numpy(coordinates).split(32)]).numpy()
    m=model.modes
    # Explicitly decode the state layout rather than calling model.decode().
    q=np.zeros((c*n,14,m))
    actual_scale=model.scale[2+m:model.base_dimension].numpy().reshape(13,2*m)[:,:m]
    q[:,1:]=coordinates[:,2+m:model.base_dimension].reshape(c*n,13,2*m)[:,:,:m]*actual_scale[None]
    phi=np.einsum('gn,ncm->gcm',bundle['data']['weights'].numpy(),model.basis.numpy())
    initial=bundle['data']['gaussians'].numpy()
    qmean=q.mean(0)
    mean=initial[None]+np.einsum('tm,gcm->tgc',qmean,phi)
    # Production score() uses torch's default float32 linspace before rounding.
    # At 32000 draws its selected IDs can differ from float64 NumPy linspace.
    # Reuse only that recorded selection rule; NumPy still recomputes all moments.
    selected=torch.linspace(0,c*n-1,min(2000,c*n)).round().long().numpy()
    position=initial[None,None]+np.einsum('btm,gcm->btgc',q[selected],phi)
    lower,upper=np.quantile(position,[.05,.95],axis=0)
    maximum_mean_error=float(np.max(np.abs(mean-np.array(published['particle_mean_m']))))
    maximum_interval_error=float(max(np.max(np.abs(lower-np.array(published['particle_q05_m']))),
        np.max(np.abs(upper-np.array(published['particle_q95_m'])))))
    if maximum_mean_error>1e-9 or maximum_interval_error>1e-9:
        raise ValueError(f'Portable moments differ from raw states: mean {maximum_mean_error}, interval {maximum_interval_error}')
    truth=torch.load(Path(inputs)/f'{scene}-evaluation.pt',weights_only=True)['particles'].numpy()
    gap=list(range(4,10)) if views==4 else list(range(5,9))
    rmse=float(np.sqrt(np.mean((mean[gap]-truth[gap])**2)))
    coverage=float(np.mean((truth[gap]>=lower[gap])&(truth[gap]<=upper[gap])))
    width=float(np.mean(upper[gap]-lower[gap]))
    scores=summary['posterior_scores']
    for expected,actual in [(rmse,scores['gap_particle_coordinate_rmse_m']),
        (coverage,scores['gap_component_90_coverage']),(width,scores['gap_mean_width_m'])]:
        if abs(expected-actual)>1e-9:raise ValueError('Independent position score differs')
    matrix=phi.reshape(-1,m)
    oracle_q=np.linalg.lstsq(matrix,(truth-initial[None]).reshape(14,-1).T,rcond=None)[0].T
    oracle=initial[None]+np.einsum('tm,gcm->tgc',oracle_q,phi)
    floor=float(np.sqrt(np.mean((oracle[gap]-truth[gap])**2)))
    if abs(floor-scores['evaluation_only_subspace_oracle_gap_rmse_m'])>1e-9 or rmse+1e-9<floor:
        raise ValueError('Subspace lower bound or reported position score is inconsistent')
    target=make_target(model,reference)
    indices=torch.linspace(0,n-1,4).round().long()
    probe=chains['white'][:,indices].reshape(-1,d)
    replay,bad=target.log_ratio_no_grad(probe)
    stored=chains['log_ratio'][:,indices].flatten()
    cache_error=float((replay-stored).abs().max())
    if bool(bad.any()) or cache_error>1e-6:raise ValueError(f'Cached exact target differs: {cache_error}')
    # Recompute key scalar diagnostics from full raw chains, not summary flags.
    trace=-np.array(chains['log_ratio'])+.5*np.sum(white**2,axis=-1)+float(target.center)
    checked=[]
    for field,values in [('z0',coordinates[:,0].reshape(c,n)),('z1',coordinates[:,1].reshape(c,n)),('log_target',-trace)]:
        recomputed=scalar_diagnostics(torch.from_numpy(values.copy()))
        original=diagnostics['quantities'][field]
        if recomputed['diagnostic_pass']!=original['diagnostic_pass']:raise ValueError('Raw diagnostic flag differs')
        for metric in ['rank_folded_split_rhat','bulk_ess','tail_ess']:
            if original[metric] is not None and abs(recomputed[metric]-original[metric])>1e-4*max(1.,abs(original[metric])):
                raise ValueError('Raw diagnostic value differs')
        checked.append(field)
    f=published['force']
    covariance=np.array(f['coefficient_covariance'])
    np.testing.assert_allclose(covariance,np.array(f['mean_conditional_covariance'])+np.array(f['between_trajectory_covariance']),atol=1e-10,rtol=1e-10)
    if np.linalg.eigvalsh(covariance).min() < -1e-9:raise ValueError('Force covariance is not positive semidefinite')
    # Replay the full force posterior from all retained motion states, not only
    # the identity between three already-published covariance components.
    means=[];sum_conditional=torch.zeros(24,24,dtype=torch.float64)
    for batch in torch.from_numpy(coordinates).split(32):
        fm,fc,_=model.force_conditionals(batch)
        means.append(fm);sum_conditional+=fc.sum(0)
    all_means=torch.cat(means)
    mean_conditional=(sum_conditional/(c*n)).numpy()
    between=torch.cov(all_means.T).numpy()
    np.testing.assert_allclose(all_means.mean(0).numpy(),f['coefficient_mean'],atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(mean_conditional,f['mean_conditional_covariance'],atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(between,f['between_trajectory_covariance'],atol=1e-9,rtol=1e-8)
    if force=='joint':
        actual=coordinates[:,model.base_dimension:]
        np.testing.assert_allclose(actual.mean(0),f['explicit_sample_coefficient_mean'],atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(np.cov(actual.T),f['explicit_sample_coefficient_covariance'],atol=1e-9,rtol=1e-8)
    pose_indices=[2+m+(frame-1)*2*m+mode for frame in range(1,14) for mode in range(m)]
    pose_ess=[diagnostics['quantities'][f'z{i}']['bulk_ess'] for i in pose_indices]
    gradient_batches=int(chains['gradient_evaluations'])
    prediction=prediction_scores(position[:,gap],truth[gap])
    return {'case':name,'draws_per_chain':n,'actual_position_coordinate_scale':float(actual_scale[0,0]),
        'finite_chain_checks_pass':diagnostics['all_monitored_quantities_pass'],
        'interval_selection_rule':'producer CPU torch.linspace default float32, rounded to integer IDs; NumPy recomputes quantiles at those IDs',
        'maximum_position_mean_replay_error_m':maximum_mean_error,
        'maximum_interval_replay_error_m':maximum_interval_error,'maximum_cached_log_ratio_error':cache_error,
        'independent_gap_rmse_m':rmse,'subspace_oracle_gap_rmse_m':floor,'raw_diagnostics_replayed':checked,
        'force_total_covariance_identity':True,'full_force_posterior_replay_pass':True,
        'prepared_input_and_evaluation_hashes_pass':True,'gradient_batches':gradient_batches,
        'minimum_common_pose_bulk_ess':min(pose_ess),
        'gradient_batches_per_minimum_pose_bulk_ess':gradient_batches/max(min(pose_ess),1e-15),
        'efficiency_qualification':'reported four-chain gradient batches divided by worst common modal-pose ESS; excludes fitting, not a hardware-normalized speed claim',
        'prediction_scores':prediction,'all_pass':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--case-index',type=int)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    rows=[]
    for i in ([a.case_index] if a.case_index is not None else range(16)):
        row=verify_case(a.raw,a.inputs,i);rows.append(row);print(json.dumps(row),flush=True)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'all_pass':all(r['all_pass'] for r in rows),'source_fingerprint':source_fingerprint(),
        'cases':rows,'qualification':'raw numerical/scoring replay; does not assert finite-chain convergence or prove physical correctness, global exploration or calibration'},indent=2)+'\n')
