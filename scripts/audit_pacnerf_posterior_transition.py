"""Post-hoc independent linear integration checks at saved posterior states.

Uses sixteen stratified retained states per completed case. Reassembles balanced
nodal diffusion factors and compares quadrature to full-matrix exponentials.
This neither validates contact integration nor bounds error everywhere in a tail.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from pacnerf_comparison import make_model,cases,Target
from physgauss.cli import source_fingerprint


@torch.no_grad()
def audit(raw,inputs,output):
    rows=[];raw=Path(raw)
    for index,(scene,views,noise,force) in enumerate(cases()):
        name=f'{scene}__{views}views__{noise}__{force}'
        paths=list((raw/f'chains-{index}').glob(f'**/{name}/chains.pt'))
        if not paths:continue
        settings=json.loads((raw/f'results-{index}'/name/'settings.json').read_text())
        if settings['source_fingerprint']!=source_fingerprint():raise ValueError('Numerical source differs')
        reference=torch.load(paths[0].parent/'reference.pt',weights_only=True)
        chains=torch.load(paths[0],weights_only=True);white=chains['white']
        selected=torch.linspace(0,white.shape[1]-1,4).round().long()
        model,_=make_model(inputs,index);target=Target(model,reference['mean'],reference['precision'])
        z=target.decode(white[:,selected].flatten(0,1));m=model.modes;d=2*m
        q,v=model.decode(z)
        x=model.data['rest'][None,None]+torch.einsum('btm,ncm->btnc',q[:,:-1],model.basis)
        velocity=torch.einsum('btm,ncm->btnc',v[:,:-1],model.basis)
        delta=x[:,:,model.edges[:,1]]-x[:,:,model.edges[:,0]]
        length=(delta.square().sum(-1)+1e-12).sqrt();direction=delta/length[...,None]
        speed=((velocity[:,:,model.edges[:,1]]-velocity[:,:,model.edges[:,0]])*direction).sum(-1)
        features=torch.stack((length/model.length-1,speed),-1)
        ratio=model.network(features)/model.constant if noise=='neural' else torch.ones_like(speed)
        # Explicit nodal assembly, independent of the production edge-basis shortcut.
        nodal=torch.einsum('ne,btec,bte->btnce',model.incidence,direction,ratio)
        factor=torch.einsum('ncm,btnce->btme',model.basis,nodal)*model.internal_scale/math.sqrt(len(model.edges)/len(model.data['rest']))
        spectral=factor@factor.transpose(-1,-2)+model.acc_noise_floor**2*torch.eye(m,dtype=z.dtype)
        a=z.new_zeros(len(z),d,d);a[:,:m,m:]=torch.eye(m,dtype=z.dtype)
        a[:,m:,:m]=-torch.diag_embed(model.eigen[None]*torch.exp(.7*z[:,:1]))
        a[:,m:,m:]=-torch.eye(m,dtype=z.dtype)*(.8*torch.exp(.7*z[:,1]))[:,None,None]
        f=torch.matrix_exp(a*model.dt)
        van_loan=z.new_zeros(len(z),13,2*d,2*d)
        van_loan[:,:,:d,:d]=a[:,None];van_loan[:,:,d:,d:]=-a[:,None].transpose(-1,-2)
        van_loan[:,:,m:d,d+m:]=spectral
        block=torch.matrix_exp(van_loan*model.dt)
        exact_q=block[:,:,:d,d:]@f[:,None].transpose(-1,-2)
        exact_q=(exact_q+exact_q.transpose(-1,-2))/2
        affine=z.new_zeros(len(z),d+m,d+m);affine[:,:d,:d]=a
        affine[:,m:d,d:]=torch.eye(m,dtype=z.dtype)
        drive=torch.matrix_exp(affine*model.dt)[:,:d,d:]
        exact_g=drive[:,None]@model.acceleration_mapping[None]
        _,approx_q,approx_g,_=model.terms(z)
        chol=torch.linalg.cholesky(exact_q)
        rel=torch.linalg.solve_triangular(chol,approx_q,upper=False)
        rel=torch.linalg.solve_triangular(chol,rel.transpose(-1,-2),upper=False).transpose(-1,-2)
        rel=(rel+rel.transpose(-1,-2))/2-torch.eye(d,dtype=z.dtype)
        cov_error=float(((approx_q-exact_q).flatten(2).norm(dim=-1)/exact_q.flatten(2).norm(dim=-1)).max())
        relative_eigen_error=float(torch.linalg.eigvalsh(rel).abs().max())
        load_error=float(((approx_g-exact_g).flatten(1).norm(dim=-1)/exact_g.flatten(1).norm(dim=-1)).max())
        theta=reference['mean'][:2]+white.flatten(0,1)@target.transform[:2].T
        rows.append({'case':name,'posterior_states_checked':len(z),
            'all_retained_parameter_quantiles_05_50_95':torch.quantile(theta,theta.new_tensor([.05,.5,.95]),dim=0).tolist(),
            'maximum_covariance_relative_frobenius_error':cov_error,
            'maximum_covariance_relative_eigenvalue_error':relative_eigen_error,
            'maximum_load_relative_frobenius_error':load_error,
            'minimum_approximate_covariance_eigenvalue':float(torch.linalg.eigvalsh(approx_q).min())})
    result={'source_fingerprint':source_fingerprint(),'cases':rows,
        'qualification':'post-hoc sixteen-state stratified posterior check using independent nodal/full-matrix assembly; not a bound on unsampled tails, contact/material validation or new production tuning'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    audit(a.raw,a.inputs,a.output)
