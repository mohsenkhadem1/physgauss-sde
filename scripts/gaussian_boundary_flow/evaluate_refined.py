"""GPU evaluation of a finished CPU training checkpoint, immutable training inputs."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.model import observe
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow.campaign import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--kind',choices=['sine','release','triangle','reverse'],required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;root=Path('runs/gaussian-boundary-flow-refined-v03');plan=json.loads((root/'plan.json').read_text());device=args.device
    bank=torch.load(root/'training.pt',weights_only=True,map_location=device);state=torch.load(root/'checkpoint.pt',weights_only=False,map_location=device)
    if state['step']!=plan['steps']:raise ValueError('Training is not finished')
    model=RefinedBlock(device=device);flow=ConditionalFlow(bank['x'].shape[1]).to(device);flow.load_state_dict(state['model']);flow.eval()
    # Generate on CPU for the same seeded physical cohort across GPU workers.
    torch.manual_seed(plan['test_seed']);truth=torch.logit(torch.rand(plan['test_count'],6).clamp(.001,.999)).to(device)
    start=time.time()
    with torch.no_grad():
        base=model.rollout(truth);g=torch.Generator(device=device).manual_seed(1981)
        context=(observe(base,generator=g)-bank['mean'])/bank['std']
        draws=flow.sample(context,32,generator=torch.Generator(device=device).manual_seed(1841))
        torch.manual_seed(9920);prior=torch.logit(torch.rand(len(truth),32,6).clamp(.001,.999)).to(device)
        x=(bank['x']-bank['mean'])/bank['std'];aug=torch.cat([x,torch.ones(len(x),1,device=device)],-1)
        W=torch.linalg.solve(aug.T@aug+torch.eye(aug.shape[1],device=device)*10,aug.T@bank['z'])
        ridge=torch.cat([context,torch.ones(len(context),1,device=device)],-1)@W
        target=model.rollout(truth,kind=args.kind);rows={};means={}
        for name,latent in [('flow',draws),('prior',prior),('ridge',ridge[:,None])]:
            flat=latent.flatten(0,1);pred=torch.cat([model.rollout(flat[i:i+128],kind=args.kind)for i in range(0,len(flat),128)]).reshape(len(truth),latent.shape[1],*target.shape[1:])
            mean=pred.mean(1);error=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
            lo,hi=torch.quantile(pred,torch.tensor([.05,.95],device=device),dim=1)
            rows[name]={'future_rmse_mm_mean':float(error.mean()),'per_case_mm':error.tolist(),'coverage_90':float(((target[:,17:]>=lo[:,17:])&(target[:,17:]<=hi[:,17:])).float().mean()),'interval_width_mm':float((hi[:,17:]-lo[:,17:]).mean()*1000)};means[name]=mean.cpu()
        lo,hi=torch.quantile(model.parameters(draws),torch.tensor([.05,.95],device=device),dim=1);actual=model.parameters(truth)
        report={'scope':plan['scope'],'device':device,'program':args.kind,'methods':rows,'parameter_coverage_90':((actual>=lo)&(actual<=hi)).float().mean(0).tolist(),'elapsed_s':time.time()-start,'checkpoint_sha256':hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest(),'test_seed':plan['test_seed'],'acquisition_noise_seed':1981}
        if args.kind=='sine':
            fine=model.rollout(truth[:16],dt=.00125,stride=20)
            report['temporal_refinement_mm']=float((fine-base[:16]).square().mean().sqrt()*1000)
            cpu=RefinedBlock();cpuu=cpu.rollout(truth[:4].cpu())
            report['cpu_gpu_max_displacement_difference_m']=float((cpuu-base[:4].cpu()).abs().max())
        save_json(root/f'evaluation-{args.kind}.json',report)
        torch.save({'truth':truth.cpu(),'posterior':draws.cpu(),'target':target.cpu(),'prediction':means},root/f'evaluation-{args.kind}.pt')
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
