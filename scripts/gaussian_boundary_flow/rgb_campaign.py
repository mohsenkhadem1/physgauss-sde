"""Raw rendered prefix -> fixed train-only PCA -> conditional physical flow.

Appearance and camera are known and fixed. This is not real-image reconstruction.
Uses the frozen track-study simulator/training parameters for an input ablation.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse,time,json,hashlib
from pathlib import Path
import torch
import numpy as np
from scripts.gaussian_boundary_flow.model import Block
from scripts.gaussian_boundary_flow.visual import render
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow.campaign import save_json


def images(block,z):
    with torch.no_grad():
        u=block.rollout(z)
        return render(block,u[:,0:17:2],size=24).flatten(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=15000);p.add_argument('--out',default='runs/gaussian-boundary-flow-rgb-v01');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(39441);start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=Path('runs/gaussian-boundary-flow-v01');bank=torch.load(source/'training.pt',weights_only=True);z=bank['z'];block=Block()
    image_path=out/'images.pt'
    if image_path.exists():raw=torch.load(image_path,weights_only=True)
    else:
        parts=[]
        for i in range(0,len(z),16):
            parts.append(images(block,z[i:i+16]))
            if i%1024==0:print('rendered',i,'seconds',time.time()-start,flush=True)
        raw=torch.cat(parts);torch.save(raw,image_path)
    feat_path=out/'features.pt'
    if feat_path.exists():features=torch.load(feat_path,weights_only=True)
    else:
        mean=raw.mean(0);_,_,V=torch.pca_lowrank(raw-mean,q=64,niter=3,center=False)
        projected=(raw-mean)@V;scale=projected.std(0).clamp_min(.01)
        features={'mean':mean,'V':V,'scale':scale,'context':projected/scale};torch.save(features,feat_path)
    context=features['context'];flow=ConditionalFlow(64);opt=torch.optim.AdamW(flow.parameters(),lr=3e-4,weight_decay=1e-5)
    checkpoint=out/'checkpoint.pt';begin=0
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False);flow.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);begin=state['step'];torch.set_rng_state(state['rng'])
    for step in range(begin,args.steps):
        idx=torch.randint(len(z),(256,));noisy=context[idx]+torch.randn(256,64)*(.005/features['scale'])
        loss=flow.loss(z[idx],noisy);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(flow.parameters(),5.);opt.step()
        if (step+1)%1000==0:
            tmp=out/'checkpoint.tmp';torch.save({'model':flow.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'rng':torch.get_rng_state()},tmp);tmp.replace(checkpoint)
            print('step',step+1,'loss',float(loss.detach()),'seconds',time.time()-start,flush=True)
    flow.eval();torch.manual_seed(99172)
    truth=torch.logit(torch.rand(128,6).clamp(.001,.999))
    with torch.no_grad():
        testimages=torch.cat([images(block,truth[i:i+16])for i in range(0,128,16)])
        observed=testimages+torch.randn_like(testimages)*.005
        testcontext=((observed-features['mean'])@features['V'])/features['scale']
        draws=flow.sample(testcontext,32)
        aug=torch.cat([context,torch.ones(len(context),1)],-1)
        W=torch.linalg.solve(aug.T@aug+torch.eye(aug.shape[1])*10,aug.T@z)
        ridge=torch.cat([testcontext,torch.ones(len(testcontext),1)],-1)@W
        report={'scope':'Synthetic RGB prefix with fixed known appearance/cameras, train-only 64D PCA; no real data','train':len(z),'test':128,'steps':args.steps,'pixel_noise_std':.005,'future_programs':{},'parameter_coverage_90':None}
        physical=block.parameters(truth);sp=block.parameters(draws);lo,hi=torch.quantile(sp,torch.tensor([.05,.95]),dim=1)
        report['parameter_coverage_90']=((physical>=lo)&(physical<=hi)).float().mean(0).tolist();stored={}
        for kind in ('sine','release','triangle','reverse'):
            target=block.rollout(truth,kind=kind);row={}
            for name,samples in [('flow',draws),('ridge',ridge[:,None])]:
                flat=samples.flatten(0,1);prediction=torch.cat([block.rollout(flat[i:i+128],kind=kind)for i in range(0,len(flat),128)]).reshape(128,samples.shape[1],*target.shape[1:])
                mean=prediction.mean(1);error=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
                lo,hi=torch.quantile(prediction,torch.tensor([.05,.95]),dim=1)
                row[name]={'future_rmse_mm_mean':float(error.mean()),'per_case_mm':error.tolist(),'coordinate_coverage_90':float(((target[:,17:]>=lo[:,17:])&(target[:,17:]<=hi[:,17:])).float().mean())}
                # Held-time, unseen-view rendering of predicted physical states.
                gt=torch.cat([render(block,target[i:i+8,17::4],size=24,offset=115.)for i in range(0,128,8)])
                pr=torch.cat([render(block,mean[i:i+8,17::4],size=24,offset=115.)for i in range(0,128,8)])
                mse=(gt-pr).square().mean((1,2,3,4));row[name]['novel_view_psnr_db_mean']=float((-10*mse.clamp_min(1e-12).log10()).mean())
                if name=='flow':stored[kind]=mean
            report['future_programs'][kind]=row
            print(kind,{k:v['future_rmse_mm_mean']for k,v in row.items()},flush=True)
        report['elapsed_seconds']=time.time()-start
        torch.save({'truth':truth,'posterior':draws,'prediction':stored},out/'evaluation.pt');save_json(out/'results.json',report)
        save_json(out/'provenance.json',{'source_training_sha256':hashlib.sha256((source/'training.pt').read_bytes()).hexdigest(),'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('scripts/gaussian_boundary_flow').glob('*.py')},'seed':39441,'test_seed':99172})

if __name__=='__main__':main()
