"""Different forward mesh at evaluation: exposes shared-simulator optimism."""
import json
from pathlib import Path
import torch
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.model import observe
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow.campaign import save_json


def main():
    torch.set_num_threads(2);device='cuda:2';root=Path('runs/gaussian-boundary-flow-refined-v03');bank=torch.load(root/'training.pt',weights_only=True,map_location=device);state=torch.load(root/'checkpoint.pt',weights_only=False,map_location=device)
    b=RefinedBlock(device=device,n=3);reference=RefinedBlock(device=device,n=4);flow=ConditionalFlow(144).to(device);flow.load_state_dict(state['model']);flow.eval()
    torch.manual_seed(7811);truth=torch.logit(torch.rand(64,6).clamp(.001,.999)).to(device)
    with torch.no_grad():
        true_prefix=reference.rollout(truth);context=(observe(true_prefix,generator=torch.Generator(device=device).manual_seed(991))-bank['mean'])/bank['std']
        draws=flow.sample(context,32,generator=torch.Generator(device=device).manual_seed(182))
        x=(bank['x']-bank['mean'])/bank['std'];aug=torch.cat([x,torch.ones(len(x),1,device=device)],-1)
        W=torch.linalg.solve(aug.T@aug+torch.eye(aug.shape[1],device=device)*10,aug.T@bank['z']);ridge=torch.cat([context,torch.ones(len(context),1,device=device)],-1)@W
        target=reference.rollout(truth,kind='reverse');rows={}
        for name,latent in [('flow',draws),('ridge',ridge[:,None]),('oracle_parameters_coarse_mesh',truth[:,None])]:
            flat=latent.flatten(0,1);pred=torch.cat([b.rollout(flat[i:i+128],kind='reverse')for i in range(0,len(flat),128)]).reshape(64,latent.shape[1],*target.shape[1:]);mean=pred.mean(1)
            error=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
            lo,hi=torch.quantile(pred,torch.tensor([.05,.95],device=device),dim=1)
            rows[name]={'future_rmse_mm_mean':float(error.mean()),'per_case_mm':error.tolist(),'coverage_90':float(((target[:,17:]>=lo[:,17:])&(target[:,17:]<=hi[:,17:])).float().mean())}
        save_json(root/'cross-mesh.json',{'scope':'64 unseen parameter worlds; infer on 64 nodes from 125-node observations; force reversal; no retraining','seed':7811,'methods':rows,'reference_limit':'125-node linear-elastic model, not a converged nonlinear physical ground truth'})
        print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
