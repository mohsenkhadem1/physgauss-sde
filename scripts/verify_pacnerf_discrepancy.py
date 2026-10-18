"""Independent saved-state, density, moment and numerical-integral replay."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from pacnerf_comparison import cases
from pacnerf_discrepancy import make_model,make_target
from compare_pacnerf_discrepancy import file_hash
from physgauss.cli import source_fingerprint
from physgauss.chain_diagnostics import scalar_diagnostics


def locate(root,pattern):
    paths=list(Path(root).glob(pattern))
    if len(paths)!=1:raise ValueError(f'Expected unique artifact {pattern}; found {len(paths)}')
    return paths[0]


@torch.no_grad()
def independent_integrals(model,z):
    """Independent nodal force assembly and full balanced Van Loan exponential."""
    m=model.modes;d=2*m;b=len(z)
    q,v=model.decode(z)
    x=model.data['rest'][None,None]+torch.einsum('btm,ncm->btnc',q[:,:-1],model.basis)
    velocity=torch.einsum('btm,ncm->btnc',v[:,:-1],model.basis)
    delta=x[:,:,model.edges[:,1]]-x[:,:,model.edges[:,0]]
    length=(delta.square().sum(-1)+1e-12).sqrt();direction=delta/length[...,None]
    speed=((velocity[:,:,model.edges[:,1]]-velocity[:,:,model.edges[:,0]])*direction).sum(-1)
    features=torch.stack((length/model.length-1,speed),-1)
    amp=model.network(features)/model.constant if model.noise=='neural' else torch.ones_like(speed)
    nodal=torch.einsum('ne,btec,bte->btnce',model.incidence,direction,amp)
    factor=torch.einsum('ncm,btnce->btme',model.basis,nodal)*model.internal_scale/math.sqrt(len(model.edges)/len(model.data['rest']))
    sd,gs,_=model.scales(z)
    spectral=factor@factor.transpose(-1,-2)+(model.acc_noise_floor**2+sd.square())[:,None,None,None]*torch.eye(m,dtype=z.dtype)
    omega2=model.eigen[None]*torch.exp(.7*z[:,:1]);damping=.8*torch.exp(.7*z[:,1])
    balance=torch.cat((omega2.clamp_min(1).sqrt(),torch.ones(b,m,dtype=z.dtype)),-1)
    a=z.new_zeros(b,d,d);a[:,:m,m:]=torch.eye(m,dtype=z.dtype)
    a[:,m:,:m]=-torch.diag_embed(omega2);a[:,m:,m:]=-damping[:,None,None]*torch.eye(m,dtype=z.dtype)
    a=a*balance[:,:,None]/balance[:,None,:]
    f=torch.matrix_exp(a*model.dt)
    block=z.new_zeros(b,13,2*d,2*d)
    block[:,:,:d,:d]=a[:,None];block[:,:,d:,d:]=-a[:,None].transpose(-1,-2)
    block[:,:,m:d,d+m:]=spectral
    exp=torch.matrix_exp(block*model.dt)
    covariance=exp[:,:,:d,d:]@f[:,None].transpose(-1,-2)
    covariance=covariance/balance[:,None,:,None]/balance[:,None,None,:]
    covariance=(covariance+covariance.transpose(-1,-2))/2
    affine=z.new_zeros(b,d+m,d+m);affine[:,:d,:d]=a;affine[:,m:d,d:]=torch.eye(m,dtype=z.dtype)
    drive=torch.matrix_exp(affine*model.dt)[:,:d,d:]/balance[:,:,None]
    factors=torch.cat((torch.ones(b,model.original_force_dimension,dtype=z.dtype),
        gs[:,None].expand(-1,model.force_dimension-model.original_force_dimension)),-1)
    mapping=(drive[:,None]@model.acceleration_mapping[None])*factors[:,None,None,:]
    _,actual_cov,actual_map,_=model.terms(z)
    chol=torch.linalg.cholesky(covariance)
    relative=torch.linalg.solve_triangular(chol,actual_cov-covariance,upper=False)
    relative=torch.linalg.solve_triangular(chol,relative.transpose(-1,-2),upper=False).transpose(-1,-2)
    covariance_error=float(torch.linalg.eigvalsh((relative+relative.transpose(-1,-2))/2).abs().max())
    load_error=float(((mapping-actual_map).flatten(1).norm(dim=-1)/mapping.flatten(1).norm(dim=-1)).max())
    minimum=float(torch.linalg.eigvalsh(actual_cov).min())
    if covariance_error>1e-5 or load_error>1e-5 or minimum<=0:
        raise ValueError(f'Independent transition integrals failed: {covariance_error}, {load_error}, {minimum}')
    return {'states':b,'maximum_covariance_relative_eigenvalue_error':covariance_error,
            'maximum_load_relative_error':load_error,'minimum_covariance_eigenvalue':minimum,'all_pass':True}


@torch.no_grad()
def verify_case(raw,inputs,index):
    scene,views,noise,force=cases()[index];name=f'{scene}__{views}views__{noise}__{force}'
    result=Path(raw)/f'results-{index}';states=Path(raw)/f'chains-{index}'
    root=locate(result,f'**/{name}/summary.json').parent
    summary=json.loads((root/'summary.json').read_text());posterior=json.loads((root/'posterior.json').read_text())
    settings=summary['settings']
    if settings['source_fingerprint']!=source_fingerprint():raise ValueError('Source fingerprint differs')
    for name_,digest in settings['inference_files_sha256_lf'].items():
        if file_hash(Path(__file__).parent/name_)!=digest:raise ValueError('Inference file changed: '+name_)
    manifest=json.loads((Path(inputs)/f'{scene}-manifest.json').read_text())
    import hashlib
    for suffix,key in [('inputs','inputs_sha256'),('evaluation','evaluation_sha256')]:
        if hashlib.sha256((Path(inputs)/f'{scene}-{suffix}.pt').read_bytes()).hexdigest()!=manifest[key]:raise ValueError('Input/evaluation hash differs')
    if manifest['inputs_sha256']!=settings['input_sha256']:raise ValueError('Worker input hash differs')
    checkpoint=locate(states,f'**/{name}/chains.pt')
    chains=torch.load(checkpoint,weights_only=True);reference=torch.load(checkpoint.parent/'reference.pt',weights_only=True)
    c,n,d=chains['white'].shape
    if c!=4 or n<2000:raise ValueError('Insufficient final sampling budget')
    if reference['coordinates']!='hierarchical-discrepancy-innovation-v1':raise ValueError('Wrong target coordinates')
    stationarity=reference['stationarity'];last=stationarity['history'][-1]
    if not stationarity['stationary'] or stationarity['gauss_newton_decrement']>=.02 or last['gradient_max_abs']>=.02:raise ValueError('Reference is not stationary')
    model,_=make_model(inputs,index);target=make_target(model,reference)
    white=chains['white'];native=target.coordinates(white.flatten(0,1))
    physical=target.decode(white.flatten(0,1))
    probes=torch.linspace(0,c*n-1,16).round().long()
    decoded,jac=target.model.transform(native[probes])
    density=float((model.objective(decoded)-jac-target.model.objective(native[probes])).abs().max())
    cache,bad=target.log_ratio_no_grad(white.flatten(0,1)[probes])
    cache_error=float((cache-chains['log_ratio'].flatten()[probes]).abs().max())
    if bool(bad.any()) or density>1e-5 or cache_error>1e-6:raise ValueError('Independent posterior target replay failed')
    coordinates=physical.numpy();m=model.modes
    q=np.zeros((c*n,14,m));scale=model.scale[2+m:model.base_dimension].numpy().reshape(13,2*m)[:,:m]
    q[:,1:]=coordinates[:,2+m:model.base_dimension].reshape(c*n,13,2*m)[:,:,:m]*scale[None]
    phi=model.renderer.gaussian_basis.numpy();rest=model.data['gaussians'].numpy()
    mean=rest[None]+np.einsum('tm,gcm->tgc',q.mean(0),phi)
    selected=torch.linspace(0,c*n-1,min(2000,c*n)).round().long().numpy()
    positions=rest[None,None]+np.einsum('btm,gcm->btgc',q[selected],phi)
    lower,upper=np.quantile(positions,[.05,.95],axis=0)
    errors=[np.max(np.abs(mean-np.asarray(posterior['particle_mean_m']))),
        np.max(np.abs(lower-np.asarray(posterior['particle_q05_m']))),np.max(np.abs(upper-np.asarray(posterior['particle_q95_m'])))]
    if max(errors)>1e-9:raise ValueError('Independent NumPy moments or quantiles differ')
    truth=torch.load(Path(inputs)/f'{scene}-evaluation.pt',weights_only=True)['particles'].numpy()
    gap=list(range(4,10)) if views==4 else list(range(5,9))
    scores={'gap_particle_coordinate_rmse_m':float(np.sqrt(np.mean((mean[gap]-truth[gap])**2))),
        'gap_component_90_coverage':float(np.mean((truth[gap]>=lower[gap])&(truth[gap]<=upper[gap]))),
        'gap_mean_width_m':float(np.mean(upper[gap]-lower[gap]))}
    for key,value in scores.items():
        if abs(value-summary['posterior_scores'][key])>1e-9:raise ValueError('Independent score differs: '+key)
    # Recompute covariance decomposition instead of trusting a result flag.
    total=np.asarray(posterior['force']['coefficient_covariance'])
    components=np.asarray(posterior['force']['mean_conditional_covariance'])+np.asarray(posterior['force']['between_trajectory_covariance'])
    np.testing.assert_allclose(total,components,atol=1e-10,rtol=1e-10)
    diagnostic=json.loads((root/'diagnostics.json').read_text())
    arrays={f'physical_{i}':physical[:,i].reshape(c,n) for i in [0,1,d-3,d-2,d-1]}
    arrays['log_target']=chains['log_ratio']-.5*white.square().sum(-1)-target.center
    for key,values in arrays.items():
        fresh=scalar_diagnostics(values);old=diagnostic['quantities'][key]
        for metric in ['rank_folded_split_rhat','bulk_ess','tail_ess']:
            if (fresh[metric] is None)!=(old[metric] is None):raise ValueError('Scalar diagnostic differs')
            if fresh[metric] is not None and abs(fresh[metric]-old[metric])>1e-6*max(1,abs(old[metric])):raise ValueError('Scalar diagnostic differs: '+key)
    all_flags=[]
    for row in diagnostic['quantities'].values():
        flag=(row['rank_folded_split_rhat'] is not None and row['rank_folded_split_rhat']<1.01 and
            row['bulk_ess'] is not None and row['bulk_ess']>=400 and row['tail_ess'] is not None and row['tail_ess']>=400 and row['frozen_chains']==0)
        if bool(flag)!=row['diagnostic_pass']:raise ValueError('Saved diagnostic flag differs')
        all_flags.append(bool(flag))
    if summary['valid_for_ranking']!=(all(all_flags) and n>=2000 and not settings['pilot']):raise ValueError('Convergence eligibility flag differs')
    numerical=independent_integrals(model,decoded[:4])
    return {'case':name,'all_pass':True,'draws_per_chain':n,'maximum_numpy_moment_error':float(max(errors)),
        'maximum_density_identity_error':density,'maximum_cached_target_error':cache_error,
        'independent_gap_scores':scores,'scalar_diagnostics_replayed':list(arrays),'independent_integrals':numerical,
        'qualification':'raw target, moment, score and scalar diagnostic replay; not proof of global exploration or physical calibration'}
