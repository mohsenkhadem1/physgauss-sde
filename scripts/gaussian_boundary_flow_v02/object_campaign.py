"""Exposed real-object pilot: simulator-bank amortization, no forecast-target access.

Training is geometry/command specific. This is not a pretrained zero-shot model.
All inverse inference is feed-forward after the forward-simulation bank is built.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor,predictor_digest
from scripts.gaussian_boundary_flow_v02.object_backend import ObjectBoundaryBackend
from scripts.gaussian_boundary_flow_v02.probabilistic import GaussianMixtureRegressor
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.external_benchmark.data import fps


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='single_lift_sloth');p.add_argument('--samples',type=int,default=256);p.add_argument('--steps',type=int,default=5000);p.add_argument('--draws',type=int,default=16);p.add_argument('--nodes',type=int,default=64);p.add_argument('--device',default='cuda:1');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(8211);rng=np.random.default_rng(7718)
    out=Path('runs/gaussian-boundary-flow-v02-objects')/args.case;out.mkdir(parents=True,exist_ok=True);start=time.time()
    data=load_phystwin_predictor(args.case,max_nodes=args.nodes)
    b=ObjectBoundaryBackend(data,args.device);frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    # Missing coordinates are omitted identically for simulation and observation.
    mask=np.repeat(data['observed_mask'][frames][:,ids,None],3,axis=-1).reshape(-1)
    plan={'case':args.case,'args':vars(args),'scope':'Historically exposed PhysTwin; geometry-specific synthetic training bank, finite-strain quasistatic convex-hull pilot','input_digest':predictor_digest(data),'parameters':b.names,'bounds':[b.lower.tolist(),b.upper.tolist()],'prefix_frames':frames.tolist(),'point_ids':ids.tolist(),'coordinate_mask_count':int(mask.sum()),'noise_m':.001,'prior':'independent standard normal logits within declared response-ratio bounds','failed_bank_worlds':'Recorded and excluded; posterior is conditional on solver-feasible prior worlds','source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/object_backend.py'),Path('scripts/gaussian_boundary_flow_v02/benchmark_inputs.py')]}}
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
                    feature=relative.reshape(-1)[mask];np.savez_compressed(path,z=z,features=feature,minJ=min(v['minJ']for v in d));latents.append(z);features.append(feature)
                except (RuntimeError,ValueError) as e:
                    row={'attempt':attempt,'z':z.tolist(),'error':str(e)};failures.append(row);write(fail,row)
            if attempt%16==0:
                write(out/'progress.json',{'phase':'forward_bank','attempts':attempt+1,'accepted':len(latents),'failed':len(failures),'elapsed_s':time.time()-start});print('bank',attempt,len(latents),len(failures),flush=True)
            if len(latents)>=args.samples:break
        if len(latents)<args.samples:raise RuntimeError('Insufficient solver-feasible bank worlds')
        np.savez_compressed(bank_path,z=np.array(latents),features=np.array(features));write(out/'bank-diagnostics.json',{'accepted':len(latents),'failed':len(failures),'failures':failures})
    bank=np.load(bank_path);clean=torch.tensor(bank['features'],dtype=torch.float32);z=torch.tensor(bank['z'],dtype=torch.float32)
    mean=clean.mean(0);_,_,V=torch.pca_lowrank(clean-mean,q=min(16,len(z)-1,clean.shape[1]),center=False,niter=3);project=(clean-mean)@V;scale=project.std(0).clamp_min(.001)
    context=project/scale;observed=data['observed'][frames][:,ids];observed=(observed-observed[:1]).reshape(-1)[mask]
    real=torch.tensor(observed,dtype=torch.float32);realcontext=((real-mean)@V)/scale
    outside=(real-mean)-((real-mean)@V)@V.T
    write(out/'context-diagnostic.json',{'prefix_unrepresented_rms_mm':float(outside.square().mean().sqrt()*1000),'real_standardized_context_norm':float(realcontext.norm()),'bank_context_norm_max':float(context.norm(dim=1).max()),'scope':'Simulator-bank support diagnostic; not a calibrated OOD probability'})
    models={'flow':ConditionalFlow(context.shape[1],b.parameter_dim),'mixture':GaussianMixtureRegressor(context.shape[1],b.parameter_dim)}
    posteriors={}
    for name,net in models.items():
        path=out/f'{name}.pt'
        if path.exists():net.load_state_dict(torch.load(path,weights_only=True))
        else:
            opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5);g=torch.Generator().manual_seed(9921)
            for step in range(args.steps):
                ix=torch.randint(len(z),(128,),generator=g)
                # Single raw observation-noise draw before orthonormal projection.
                noise=torch.randn((128,clean.shape[1]),generator=g)*.001
                ctx=context[ix]+(noise@V)/scale
                loss=net.loss(z[ix],ctx);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            torch.save(net.state_dict(),path)
        net.eval();posteriors[name]=net.sample(realcontext[None],args.draws,generator=torch.Generator().manual_seed(138))[0].detach().numpy()
    np.savez_compressed(out/'inference-sealed.npz',**posteriors,mean=mean.numpy(),basis=V.numpy(),scale=scale.numpy())
    write(out/'inference-sealed.json',{'bank_sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),'forecast_target_access':False,'train_end':data['train_end']})
    # Forecasts are computed only AFTER all inverse samples are sealed.
    allframes=np.arange(len(data['times']));predictions={};diagnostics={}
    for name,draws in posteriors.items():
        worlds=[];failures=[]
        for i,zsample in enumerate(draws):
            try:
                pos,d=b.rollout(zsample,allframes);pos=data['nodes'][None]+pos-pos[:1];worlds.append(pos)
            except (RuntimeError,ValueError) as e:failures.append({'sample':i,'error':str(e)})
        if not worlds:raise RuntimeError(f'All {name} forecasts failed')
        worlds=np.stack(worlds);predictions[name]=worlds.mean(0);predictions[name+'_lo']=np.quantile(worlds,.05,axis=0);predictions[name+'_hi']=np.quantile(worlds,.95,axis=0)
        diagnostics[name]={'requested':len(draws),'valid':len(worlds),'failures':failures}
        print('forecast',name,len(worlds),flush=True)
    np.savez_compressed(out/'prediction.npz',**predictions,nodes=data['nodes'],observation_matrix=data['observation_matrix'],observed_initial=data['observed_initial'])
    write(out/'complete.json',{'completed':True,'fit_end':data['train_end'],'frames':len(data['times']),'forecast_samples':diagnostics,'elapsed_s':time.time()-start,'prediction_sha256':hashlib.sha256((out/'prediction.npz').read_bytes()).hexdigest(),'held_target_access':False})

if __name__=='__main__':main()
