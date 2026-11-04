"""Diagnostic new-view acquisition from an amortized posterior proposal.

Importance weighting an approximate flow posterior is NOT an exact posterior.
Report ESS and coverage; never silently claim calibration from narrower intervals.
"""
import argparse,json,time
from pathlib import Path
import torch
from scripts.gaussian_boundary_flow.model import observe
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow.campaign import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');args=p.parse_args();device=args.device;torch.set_num_threads(2)
    root=Path('runs/gaussian-boundary-flow-refined-v03');bank=torch.load(root/'training.pt',weights_only=True,map_location=device);ck=torch.load(root/'checkpoint.pt',weights_only=False,map_location=device)
    flow=ConditionalFlow(144).to(device);flow.load_state_dict(ck['model']);flow.eval();b=RefinedBlock(device=device)
    # Independent acquisition cohort; policy does not see truth or new observations.
    torch.manual_seed(17339);truth=torch.logit(torch.rand(32,6).clamp(.001,.999)).to(device)
    angles=torch.tensor([0.,45.,90.,135.],device=device)*torch.pi/180
    axes=torch.stack([torch.stack([a.cos(),a*0,a.sin()])for a in angles]);up=torch.tensor([0.,1.,0.],device=device)
    projections=torch.stack([axes,up.expand_as(axes)],1);sigma=.0005
    with torch.no_grad():
        target=b.rollout(truth);context=(observe(target,generator=torch.Generator(device=device).manual_seed(118))-bank['mean'])/bank['std']
        samples=flow.sample(context,128,generator=torch.Generator(device=device).manual_seed(188))
        flat=samples.flatten(0,1);motion=torch.cat([b.rollout(flat[i:i+128])for i in range(0,len(flat),128)]).reshape(32,128,41,8,3)
        # A new view of last observed physical time; no hidden future needed by policy.
        predicted_measurements=torch.einsum('bsni,aki->bsank',motion[:,:,16],projections).flatten(-2)
        cov=predicted_measurements-predicted_measurements.mean(1,keepdim=True)
        gram=torch.einsum('bsai,bsaj->baij',cov,cov)/(127*sigma**2)
        scores=torch.linalg.slogdet(torch.eye(16,device=device)+gram).logabsdet/2
        selected=scores.argmax(-1)
        actual_measurements=torch.einsum('bni,aki->bank',target[:,16],projections).flatten(-2)
        noise=torch.randn(actual_measurements.shape,device=device,generator=torch.Generator(device=device).manual_seed(321))*sigma
        measurements=actual_measurements+noise
        row={};allindices=torch.arange(32,device=device)
        for method,choice in [('selected',selected),('fixed',torch.zeros(32,dtype=torch.long,device=device)),('random',torch.arange(32,device=device)%4)]:
            residual=(predicted_measurements[allindices,:,choice]-measurements[allindices,choice][:,None])/sigma
            weights=torch.softmax(-.5*residual.square().sum(-1),dim=-1)
            mean=torch.einsum('bs,bstni->btni',weights,motion)
            err=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
            ess=1/weights.square().sum(-1)
            row[method]={'future_rmse_mm_mean':float(err.mean()),'per_case_mm':err.tolist(),'ESS_mean':float(ess.mean()),'ESS_min':float(ess.min()),'fraction_ESS_below_10':float((ess<10).float().mean()),'chosen_view_index':choice.tolist()}
        mean=motion.mean(1);row['no_acquisition']={'future_rmse_mm_mean':float((mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt().mean()*1000)}
        save_json(root/'acquisition.json',{'scope':'Approximate importance-weighted new-view diagnostic, not validated posterior update or active-poke policy','cases':32,'samples':128,'measurement_noise_m':sigma,'view_angles_degrees':[0,45,90,135],'methods':row,'selection':'Gaussian logdet measurement covariance; no truth or future used in selection'})
        print(json.dumps(row,indent=2))

if __name__=='__main__':main()
