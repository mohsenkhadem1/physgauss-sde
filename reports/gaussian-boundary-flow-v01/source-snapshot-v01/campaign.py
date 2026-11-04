"""Resumable controlled development campaign. No external benchmark claims."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow.model import Block,observe,PARAMETERS
from scripts.gaussian_boundary_flow.flow import ConditionalFlow

ROOT=Path(__file__).resolve().parents[2]


def save_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/gaussian-boundary-flow-v01');p.add_argument('--train',type=int,default=8192);p.add_argument('--steps',type=int,default=10000);p.add_argument('--test',type=int,default=128);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(19041)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    model=Block(device=args.device);start=time.time()
    plan={'seed':19041,'scope':'Synthetic development, 8-node linearized tetrahedral block; known camera/geometry/mass; 2D tracks, not RGB inference','parameters':PARAMETERS,'train_count':args.train,'test_count':args.test,'steps':args.steps,'device':args.device,'prefix_end_s':.4,'noise_m':.00005,'observations':'9 frames x 8 tracked nodes x 2 projected coordinates','future_programs':['sine','release','triangle','reverse'],'test_selection':'Separate seed; no hyperparameter selection on test','source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*Path(__file__).parent.glob('*.py'),ROOT/'src/physgauss/gaussian_boundary.py']}}
    plan_path=out/'plan.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text())
        if old!=plan:raise ValueError('Existing campaign plan differs; use a new --out')
    else:save_json(plan_path,plan)
    data_path=out/'training.pt'
    if data_path.exists():data=torch.load(data_path,weights_only=True,map_location=args.device)
    else:
        # Uniform physical log/range coordinates; logit coordinates unconstrained.
        z=torch.logit(torch.rand(args.train,6,device=args.device).clamp(.001,.999));features=[]
        with torch.no_grad():
            for i in range(0,args.train,128):features.append(observe(model.rollout(z[i:i+128])))
        x=torch.cat(features);mean=x.mean(0);std=x.std(0).clamp_min(.0001)
        data={'z':z,'x':x,'mean':mean,'std':std};torch.save(data,data_path)
        print('training bank ready',time.time()-start,flush=True)
    x=(data['x']-data['mean'])/data['std'];z=data['z']
    flow=ConditionalFlow(x.shape[1]).to(args.device);opt=torch.optim.AdamW(flow.parameters(),lr=3e-4,weight_decay=1e-5)
    checkpoint=out/'checkpoint.pt';begin=0
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False,map_location=args.device);flow.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);begin=state['step'];torch.set_rng_state(state['rng'].cpu())
        if args.device.startswith('cuda') and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    losses=[];flow.train()
    for step in range(begin,args.steps):
        idx=torch.randint(len(x),(256,),device=x.device)
        # Fresh observation noise avoids memorizing a fixed acquisition-noise draw.
        context=x[idx]+torch.randn_like(x[idx])*.00005/data['std']
        loss=flow.loss(z[idx],context);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(flow.parameters(),5.);opt.step();losses.append(float(loss))
        if (step+1)%500==0 or step+1==args.steps:
            tmp=out/'checkpoint.tmp';torch.save({'model':flow.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None},tmp);tmp.replace(checkpoint)
            save_json(out/'progress.json',{'step':step+1,'loss_last_500':float(np.mean(losses[-500:])),'elapsed_this_invocation_s':time.time()-start})
            print('step',step+1,'loss',np.mean(losses[-500:]),'seconds',round(time.time()-start,1),flush=True)
    flow.eval()
    # Independent deterministic evaluation noise; no suffix frames supplied.
    torch.manual_seed(99172)
    truth=torch.logit(torch.rand(args.test,6,device=x.device).clamp(.001,.999))
    with torch.no_grad():
        true_base=model.rollout(truth);context=(observe(true_base)-data['mean'])/data['std']
        draws=flow.sample(context,count=32)
        prior=torch.logit(torch.rand(args.test,32,6,device=x.device).clamp(.001,.999))
        # Linear regression on exactly the same training contexts and physical latents.
        aug=torch.cat([x,torch.ones(len(x),1,device=x.device)],-1)
        W=torch.linalg.solve(aug.T@aug+torch.eye(aug.shape[1],device=x.device)*10,aug.T@z)
        ridge=torch.cat([context,torch.ones(len(context),1,device=x.device)],-1)@W
        results={'scope':plan['scope'],'n_test':args.test,'parameters':{},'programs':{}}
        physical=model.parameters(truth);samplephysical=model.parameters(draws)
        lo,hi=torch.quantile(samplephysical,torch.tensor([.05,.95],device=x.device),dim=1)
        results['parameters']={'nominal_coverage':.90,'coverage':((physical>=lo)&(physical<=hi)).float().mean(0).tolist(),'interval_width':(hi-lo).mean(0).tolist(),'absolute_mean_error':(samplephysical.mean(1)-physical).abs().mean(0).tolist()}
        predictions={}
        for kind in plan['future_programs']:
            target=model.rollout(truth,kind=kind);row={}
            for name,latent in [('flow',draws),('prior',prior),('ridge',ridge[:,None])]:
                flat=latent.flatten(0,1);parts=[]
                for i in range(0,len(flat),128):parts.append(model.rollout(flat[i:i+128],kind=kind))
                pred=torch.cat(parts).reshape(args.test,latent.shape[1],*target.shape[1:]);mean=pred.mean(1)
                error=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
                lower,upper=torch.quantile(pred,torch.tensor([.05,.95],device=x.device),dim=1)
                row[name]={'future_rmse_mm_mean':float(error.mean()),'future_rmse_mm_per_case':error.tolist(),'coordinate_90_coverage':float(((target[:,17:]>=lower[:,17:])&(target[:,17:]<=upper[:,17:])).float().mean()),'interval_width_mm':float((upper[:,17:]-lower[:,17:]).mean()*1000)}
                if name=='flow':predictions[kind]=mean.cpu()
            results['programs'][kind]=row
            print(kind,{k:round(v['future_rmse_mm_mean'],3) for k,v in row.items()},flush=True)
        # Temporal convergence of the numerical reference, not spatial qualification.
        fine=model.rollout(truth[:16],dt=.0025,stride=10)
        results['temporal_refinement']={'cases':16,'dt_s':[.005,.0025],'trajectory_rms_difference_mm':float((fine-true_base[:16]).square().mean().sqrt()*1000)}
        results['elapsed_this_invocation_s']=time.time()-start
        results['source_hashes_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in plan['source_sha256'].items())
        torch.save({'truth':truth.cpu(),'posterior':draws.cpu(),'trajectories':{k:model.rollout(truth,kind=k).cpu() for k in plan['future_programs']},'prediction':predictions},out/'evaluation.pt')
        save_json(out/'results.json',results)

if __name__=='__main__':main()
