"""Split-conformal trajectory envelope, explicitly calibrated on the finer domain.

This is predictive calibration, NOT calibration of material/force posteriors.
"""
from pathlib import Path
import math
import torch
from scripts.gaussian_boundary_flow.model import observe
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow.campaign import save_json


def main():
    torch.set_num_threads(2);device='cuda:2';root=Path('runs/gaussian-boundary-flow-refined-v03');bank=torch.load(root/'training.pt',weights_only=True,map_location=device);state=torch.load(root/'checkpoint.pt',weights_only=False,map_location=device)
    flow=ConditionalFlow(144).to(device);flow.load_state_dict(state['model']);flow.eval();model=RefinedBlock(device=device);reference=RefinedBlock(device=device,n=4)
    with torch.no_grad():
        records=[]
        for seed in (55287,19473):
            torch.manual_seed(seed);truth=torch.logit(torch.rand(64,6).clamp(.001,.999)).to(device)
            prefix=reference.rollout(truth);context=(observe(prefix,generator=torch.Generator(device=device).manual_seed(seed+1))-bank['mean'])/bank['std']
            draws=flow.sample(context,32,generator=torch.Generator(device=device).manual_seed(seed+2));flat=draws.flatten(0,1)
            pred=torch.cat([model.rollout(flat[i:i+128],kind='reverse')for i in range(0,len(flat),128)]).reshape(64,32,41,8,3)[:,:,17:]
            target=reference.rollout(truth,kind='reverse')[:,17:]
            lo,hi=torch.quantile(pred,torch.tensor([.05,.95],device=device),dim=1)
            score=torch.maximum(lo-target,target-hi).flatten(1).max(-1).values.clamp_min(0)
            records.append((lo,hi,target,score))
        cal=records[0][3];n=len(cal);order=math.ceil((n+1)*.9);margin=cal.sort().values[order-1]
        lo,hi,target,scores=records[1]
        report={'scope':'Trajectory predictive envelope calibrated on separate 125-node simulations; not zero-shot OOD calibration or physical-parameter coverage','calibration_seed':55287,'test_seed':19473,'calibration_cases':64,'test_cases':64,'target_trajectory_coverage':.9,'finite_sample_order_statistic':order,'additive_margin_mm':float(margin*1000),'raw_coordinate_coverage':float(((target>=lo)&(target<=hi)).float().mean()),'calibrated_coordinate_coverage':float(((target>=lo-margin)&(target<=hi+margin)).float().mean()),'raw_trajectory_coverage':float((scores<=0).float().mean()),'calibrated_trajectory_coverage':float((scores<=margin).float().mean()),'raw_average_interval_width_mm':float((hi-lo).mean()*1000),'calibrated_average_interval_width_mm':float((hi-lo+2*margin).mean()*1000),'calibration_scores_mm':(cal*1000).tolist(),'test_scores_mm':(scores*1000).tolist()}
        save_json(root/'calibration.json',report);print(report)

if __name__=='__main__':main()
