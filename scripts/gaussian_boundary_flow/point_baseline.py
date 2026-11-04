"""Strong deterministic amortized baseline using the same bank and train budget."""
from pathlib import Path
import json
import torch
from torch import nn
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.model import observe
from scripts.gaussian_boundary_flow.campaign import save_json


def main():
    torch.set_num_threads(2);torch.manual_seed(1239);root=Path('runs/gaussian-boundary-flow-refined-v03');bank=torch.load(root/'training.pt',weights_only=True)
    net=nn.Sequential(nn.Linear(144,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,6))
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5);x=(bank['x']-bank['mean'])/bank['std'];z=bank['z']
    for i in range(15000):
        ids=torch.randint(len(x),(256,));noisy=x[ids]+torch.randn_like(x[ids])*.00005/bank['std'];loss=(net(noisy)-z[ids]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
    torch.save(net.state_dict(),root/'point-baseline.pt');device='cuda:1';net=net.to(device);b=RefinedBlock(device=device);net.eval()
    torch.manual_seed(77183);truth=torch.logit(torch.rand(128,6).clamp(.001,.999)).to(device)
    with torch.no_grad():
        base=b.rollout(truth);context=(observe(base,generator=torch.Generator(device=device).manual_seed(1981))-bank['mean'].to(device))/bank['std'].to(device);latent=net(context);rows={}
        for kind in ('sine','release','triangle','reverse'):
            target=b.rollout(truth,kind=kind);pred=b.rollout(latent,kind=kind);err=(pred[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
            rows[kind]={'future_rmse_mm_mean':float(err.mean()),'per_case_mm':err.tolist()}
    save_json(root/'point-baseline.json',{'scope':'Same training bank, 15000 steps, 256 batch, deterministic neural physical-parameter regressor','methods':rows})
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
