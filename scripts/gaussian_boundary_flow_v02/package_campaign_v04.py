"""Stable-package prospective validation: simulator-bank amortization, no forecast-target access.

Training is geometry/command specific. This is not a pretrained zero-shot model.
All inverse inference is feed-forward after the forward-simulation bank is built.
"""
import argparse,json,time,hashlib,re
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.feature_support import feature_support_diagnostic
from scripts.gaussian_boundary_flow_v02.checkpoint_contract import (save_transform,load_transform,save_bound_model,load_bound_model,inference_seal,relative_observation_noise)
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor,predictor_digest
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package, sha
from scripts.gaussian_boundary_flow_v02.softvt_validation_inputs import admission
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend
from scripts.gaussian_boundary_flow_v02.probabilistic import GaussianMixtureRegressor
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.external_benchmark.data import fps


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',choices=['convex','filtered'],required=True);p.add_argument('--package',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=128);p.add_argument('--steps',type=int,default=5000);p.add_argument('--draws',type=int,default=32);p.add_argument('--nodes',type=int,default=64);p.add_argument('--device',default='cuda:1');args=p.parse_args()
    if (args.samples,args.steps,args.draws,args.nodes,args.geometry)!=(128,5000,32,64,'convex'):raise ValueError('Frozen validation recipe requires128worlds/5000steps/32draws/convex64')
    torch.set_num_threads(2);torch.manual_seed(8211);rng=np.random.default_rng(7718)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    data=load_package(args.package);args.nodes=len(data['nodes']);case=data['name']
    if args.nodes!=64:raise ValueError('Frozen validation geometry requires64nodes')
    if data.get('data_role')!='prospective_validation':raise ValueError('This driver requires an explicitly admitted validation package')
    match=re.fullmatch(r'(object-soft|spatial-soft)-task([012])-demo([0-9]+)',case)
    if match is None:raise ValueError('Invalid validation case')
    protocol,_=admission(match[1],int(match[2]),int(match[3]),protocol_path=data['validation_protocol_path'],protocol_sha256=data['validation_protocol_sha256'])
    if protocol['recipe_sha256']!=data['validation_recipe_sha256']:raise ValueError('Validation recipe changed')
    backend=GeometryAwareGaussianBoundaryBackend if args.geometry=='filtered' else WideGaussianTableBoundaryBackend
    b=backend(data,args.device);frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    # Missing coordinates are omitted identically for simulation and observation.
    mask=np.repeat(data['observed_mask'][frames][:,ids,None],3,axis=-1).reshape(-1)
    plan={'case':case,'dataset':data['dataset'],'data_role':data['data_role'],'validation_protocol_path':data['validation_protocol_path'],'validation_protocol_sha256':data['validation_protocol_sha256'],'validation_recipe_sha256':data['validation_recipe_sha256'],'input_package':str(Path(args.package).resolve()),'input_package_manifest_sha256':sha(Path(args.package)/'manifest.json'),'prediction_methods':['flow_raw','flow_tracking','mixture_raw','mixture_tracking'],'args':vars(args),'scope':'Prospective repeated-asset validation; unchanged v03 numerical recipe; geometry-specific synthetic training bank, explicitly selected finite-strain quasistatic geometry with Gaussian table compliance. Raw mechanical and rebased tracking outputs separately scored','input_digest':predictor_digest(data),'parameters':b.names,'bounds':[b.lower.tolist(),b.upper.tolist()],'prefix_frames':frames.tolist(),'point_ids':ids.tolist(),'coordinate_mask_count':int(mask.sum()),'noise_m':.001,'noise_model':'Independent absolute observed-coordinate errors followed by shared frame-zero subtraction; nominal sigma is an assumed model, not calibrated sensor noise','prior':'independent standard normal logits within declared response-ratio bounds','failed_bank_worlds':'Recorded and excluded; posterior is conditional on solver-feasible prior worlds','source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/object_backend_geometry.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_wide.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_revision.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend_table.py'),Path('scripts/gaussian_boundary_flow_v02/benchmark_inputs.py'),Path('scripts/gaussian_boundary_flow_v02/object_backend.py'),Path('scripts/gaussian_boundary_flow_v02/softvt_validation_inputs.py'),Path('scripts/gaussian_boundary_flow_v02/checkpoint_contract.py'),Path('scripts/gaussian_boundary_flow_v02/feature_support.py'),Path('scripts/external_benchmark/fem.py'),Path('scripts/gaussian_boundary_flow_v02/probabilistic.py'),Path('scripts/gaussian_boundary_flow/flow.py')]}}
    if (out/'plan.json').exists():
        if json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Changed immutable plan')
    else:write(out/'plan.json',plan)
    bank_path=out/'bank.npz'
    if not bank_path.exists():
        latents=[];features=[];failures=[]
        for attempt in range(args.samples*2):
            z=rng.normal(size=b.parameter_dim);path=out/f'bank-world-{attempt:04d}.npz';fail=out/f'bank-failure-{attempt:04d}.json'
            if fail.exists():failures.append(json.loads(fail.read_text()));continue
            if path.exists():
                saved=np.load(path);latents.append(saved['z']);features.append(saved['features'])
            else:
                try:
                    positions,d=b.rollout(z,frames)
                    # Match relative changes; unknown initial prestress is not inferred.
                    obs=b.observation_features(positions,frames,ids);relative=obs-obs[:1]
                    feature=relative.reshape(-1)[mask];np.savez_compressed(path,z=z,features=feature,minJ=b.last_path_diagnostics['minimum_J'],maximum_residual=b.last_path_diagnostics['maximum_residual']);latents.append(z);features.append(feature)
                except (RuntimeError,ValueError) as e:
                    row={'attempt':attempt,'z':z.tolist(),'error':str(e)};failures.append(row);write(fail,row)
            if attempt%16==0:
                write(out/'progress.json',{'phase':'forward_bank','attempts':attempt+1,'accepted':len(latents),'failed':len(failures),'elapsed_s':time.time()-start});print('bank',attempt,len(latents),len(failures),flush=True)
            if len(latents)>=args.samples:break
        if len(latents)<args.samples:raise RuntimeError('Insufficient solver-feasible bank worlds')
        np.savez_compressed(bank_path,z=np.array(latents),features=np.array(features));write(out/'bank-diagnostics.json',{'accepted':len(latents),'failed':len(failures),'failures':failures})
    bank=np.load(bank_path);clean=torch.tensor(bank['features'],dtype=torch.float32,device=args.device);z=torch.tensor(bank['z'],dtype=torch.float32,device=args.device)
    transform_path=out/'transform.pt'
    if transform_path.exists():
        transform=load_transform(transform_path,bank_path=bank_path,plan_path=out/'plan.json',device=args.device)
        mean,V,scale=[transform[k] for k in ('mean','basis','scale')]
    else:
        mean=clean.mean(0);_,_,V=torch.pca_lowrank(clean-mean,q=min(16,len(z)-1,clean.shape[1]),center=False,niter=3);scale=((clean-mean)@V).std(0).clamp_min(.001)
        save_transform(transform_path,mean=mean,basis=V,scale=scale,bank_path=bank_path,plan_path=out/'plan.json')
    context=((clean-mean)@V)/scale;observed=data['observed'][frames][:,ids];observed=(observed-observed[:1]).reshape(-1)[mask]
    real=torch.tensor(observed,dtype=torch.float32,device=args.device);realcontext=((real-mean)@V)/scale
    outside=(real-mean)-((real-mean)@V)@V.T
    support=feature_support_diagnostic(clean,real,mean,V,scale,frames=len(frames),points=len(ids),noise_sigma=.001,coordinate_mask=mask)
    write(out/'feature-support.json',support)
    write(out/'context-diagnostic.json',{'feature_support':support,'out_of_bank_warning':support['out_of_bank_warning'],'prefix_unrepresented_rms_mm':float(outside.square().mean().sqrt()*1000),'real_standardized_context_norm':float(realcontext.norm()),'bank_context_norm_max':float(context.norm(dim=1).max()),'scope':'Simulator-bank support diagnostic; not a calibrated OOD probability'})
    torch.manual_seed(12000)
    models={'flow':ConditionalFlow(context.shape[1],b.parameter_dim),'mixture':GaussianMixtureRegressor(context.shape[1],b.parameter_dim)}
    posteriors={}
    for model_index,(name,net) in enumerate(models.items()):
        torch.manual_seed(22000+model_index)
        net=net.to(args.device)
        path=out/f'{name}.pt'
        if path.exists():load_bound_model(path,net,transform_path=transform_path,bank_path=bank_path,plan_path=out/'plan.json')
        else:
            opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5);g=torch.Generator(device=args.device).manual_seed(9921)
            for step in range(args.steps):
                ix=torch.randint(len(z),(128,),generator=g,device=args.device)
                # Single raw observation-noise draw before orthonormal projection.
                noise=relative_observation_noise(128,len(frames),len(ids),.001,mask=mask,generator=g,device=args.device)
                ctx=context[ix]+(noise@V)/scale
                loss=net.loss(z[ix],ctx);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            save_bound_model(path,net,transform_path=transform_path,bank_path=bank_path,plan_path=out/'plan.json')
        net.eval();posteriors[name]=net.sample(realcontext[None],args.draws,generator=torch.Generator(device=args.device).manual_seed(138))[0].detach().cpu().numpy()
    np.savez_compressed(out/'inference-sealed.npz',**posteriors,mean=mean.cpu().numpy(),basis=V.cpu().numpy(),scale=scale.cpu().numpy())
    seal=inference_seal(inference_path=out/'inference-sealed.npz',model_paths={name:out/f'{name}.pt'for name in models},transform_path=transform_path,bank_path=bank_path,plan_path=out/'plan.json');seal['train_end']=data['train_end'];write(out/'inference-sealed.json',seal)
    # Forecasts are computed only AFTER all inverse samples are sealed.
    allframes=np.arange(len(data['times']));predictions={};diagnostics={}
    for name,draws in posteriors.items():
        worlds=[];failures=[];path_diagnostics=[]
        for i,zsample in enumerate(draws):
            try:
                pos,d=b.rollout(zsample,allframes);worlds.append(pos);path_diagnostics.append(b.last_path_diagnostics)
            except (RuntimeError,ValueError) as e:failures.append({'sample':i,'error':str(e)})
        if not worlds:raise RuntimeError(f'All {name} forecasts failed')
        worlds=np.stack(worlds)
        for channel,values in [(name+'_raw',worlds),(name+'_tracking',data['nodes'][None,None]+worlds-worlds[:,:1])]:
            predictions[channel]=values.mean(0);predictions[channel+'_samples']=values
        diagnostics[name]={'requested':len(draws),'valid':len(worlds),'failures':failures,'path_diagnostics':path_diagnostics}
        print('forecast',name,len(worlds),flush=True)
    np.savez_compressed(out/'prediction.npz',**predictions,nodes=data['nodes'],observation_matrix=data['observation_matrix'],observed_initial=data['observed_initial'])
    write(out/'complete.json',{'completed':True,'fit_end':data['train_end'],'frames':len(data['times']),'forecast_samples':diagnostics,'elapsed_s':time.time()-start,'prediction_sha256':hashlib.sha256((out/'prediction.npz').read_bytes()).hexdigest(),'held_target_access':False})

if __name__=='__main__':main()
