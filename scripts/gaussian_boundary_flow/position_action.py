"""Position-command OOD test: infer support from prefix, then lift its anchor."""
from pathlib import Path
import torch
from torch import nn
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.model import observe
from scripts.gaussian_boundary_flow.campaign import save_json


def lift(block,z):
    dt=.0025;system,load,gravity=block.operators(z,dt);theta=block.parameters(z)
    Ks=system.stiffness-block.K[None]*theta[:,0].exp()[:,None,None]
    Ds=system.damping-torch.diag_embed(system.mass*.25)
    u=torch.zeros_like(load);v=u.clone();history=[u]
    for step in range(1,401):
        t=step*dt
        if t<=.4:
            command=.5*(1-torch.cos(z.new_tensor(2*torch.pi*2*t)));force=gravity+load*command
        else:
            # Prescribed 6 mm upward support motion, no future applied-load input.
            phase=z.new_tensor((t-.4)/.6*torch.pi)
            target=torch.zeros_like(u).reshape(len(z),-1,3);target[...,1]=.003*(1-phase.cos())
            velocity=torch.zeros_like(target);velocity[...,1]=.003*torch.pi/.6*phase.sin()
            force=gravity+torch.einsum('bij,bj->bi',Ks,target.flatten(1))+torch.einsum('bij,bj->bi',Ds,velocity.flatten(1))
        u,v,_=system.step(u,v,force)
        if step%10==0:history.append(u)
    return torch.stack(history,1).reshape(len(z),41,-1,3)[:,:,block.corners]


def main():
    torch.set_num_threads(2);device='cuda:0';root=Path('runs/gaussian-boundary-flow-refined-v03');d=torch.load(root/'evaluation-reverse.pt',weights_only=True,map_location=device);bank=torch.load(root/'training.pt',weights_only=True,map_location=device)
    b=RefinedBlock(device=device);truth=d['truth'];draws=d['posterior']
    net=nn.Sequential(nn.Linear(144,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,6)).to(device);net.load_state_dict(torch.load(root/'point-baseline.pt',weights_only=True,map_location=device));net.eval()
    with torch.no_grad():
        baseline=b.rollout(truth);target=lift(b,truth)
        assert torch.equal(target[:,:17],baseline[:,:17]),'Position command leaked into prefix'
        context=(observe(baseline,generator=torch.Generator(device=device).manual_seed(1981))-bank['mean'])/bank['std'];point=net(context)
        rows={}
        for name,samples in [('flow',draws),('neural_regressor',point[:,None])]:
            flat=samples.flatten(0,1);pred=torch.cat([lift(b,flat[i:i+128])for i in range(0,len(flat),128)]).reshape(128,samples.shape[1],41,8,3)
            error=(pred.mean(1)[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
            rows[name]={'future_rmse_mm_mean':float(error.mean()),'per_case_mm':error.tolist()}
        save_json(root/'position-action.json',{'scope':'128 held-out worlds; unseen prescribed upward anchor motion after prefix; frozen models, no inverse fitting','command':'6 mm smooth upward displacement of inferred material attachment; applied external load released at .4s','known':'Which inferred attachment receives the position command','prefix_identical':True,'methods':rows})
        print(rows)

if __name__=='__main__':main()
