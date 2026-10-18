"""Evaluation-only likelihood preference diagnostic at fixed, pre-existing poses.

This runs AFTER the new model is frozen/dispatched. It cannot select priors,
fit any new parameter, seed inference, or establish calibrated coverage.
"""
import argparse
import json
from pathlib import Path
import torch
from pacnerf_comparison import cases,write
from pacnerf_discrepancy import make_model,spatial_whiten
from physgauss.pacnerf_model import PACRenderer


@torch.no_grad()
def audit(inputs,baseline,output):
    records=[]
    for index in [1,4,8]:
        scene,views,noise,force=cases()[index]
        name=f'{scene}__{views}views__{noise}__{force}'
        model,_=make_model(inputs,index)
        posterior=json.loads((Path(baseline)/'cases'/name/'posterior.json').read_text())
        mean=torch.tensor(posterior['modal_pose_mean_m'],dtype=torch.float64)
        truth=torch.load(Path(inputs)/f'{scene}-evaluation.pt',weights_only=True)['particles']
        phi=model.renderer.gaussian_basis.flatten(0,1)
        oracle=torch.linalg.lstsq(phi,(truth-model.data['gaussians'][None]).flatten(1).T).solution.T
        physical=torch.zeros(2,model.dimension,dtype=torch.float64)
        state=physical[:,2+model.modes:model.base_dimension].reshape(2,13,2*model.modes)
        state[:,:,:model.modes]=torch.stack((mean[1:],oracle[1:]))/.1
        residual=model.image_residual(physical)
        new=.5*residual.square().sum(-1)
        rgb=model.renderer(torch.stack((mean[model.frames],oracle[model.frames])),model.specs)
        old=.5*((rgb-model.obs[None])/.08).square().flatten(1).sum(-1)
        direct=[]
        for frame,camera in model.specs:
            renderer=PACRenderer(dict(model.data,gaussians=truth[frame]),model.basis)
            renderer.colors=model.renderer.colors
            direct.append(renderer(torch.zeros(1,1,model.modes,dtype=torch.float64),[(frame,camera)])[0,0])
        true_rgb=torch.stack(direct)
        true_error=true_rgb-model.obs
        true_old=.5*(true_error/.08).square().sum()
        true_new=.5*(spatial_whiten(true_error,model.rho_x,model.rho_y)/model.scales(physical[:1])[2][0]).square().sum()
        records.append({'scene':scene,'frames':views,'frozen_baseline_case':name,
            'old_iid_oracle_minus_inferred_image_nll':float(old[1]-old[0]),
            'new_correlated_oracle_minus_inferred_image_nll_at_hyperprior_center':float(new[1]-new[0]),
            'old_iid_true_centers_minus_inferred_image_nll':float(true_old-old[0]),
            'new_correlated_true_centers_minus_inferred_image_nll_at_hyperprior_center':float(true_new-new[0]),
            'fixed_new_hyperparameter_values':[0.,0.,0.],'new_spatial_correlation':[model.rho_x,model.rho_y],
            'interpretation':'positive means the likelihood favors the existing biased estimate over the evaluation-only projected truth; no posterior or calibration claim'})
    result={'cases':records,'new_model_frozen_before_this_diagnostic':'a575e18; Actions 39143185132',
        'future_truth_used':'evaluation only, after model specification and dispatch',
        'qualification':'likelihood-only preference at fixed poses and fixed scale, not a causal ablation, posterior predictive validation or proof the correction removes renderer bias; Gaussian pixel independence and renderer mismatch can favor a biased pose even under converged inference'}
    write(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--baseline',default='examples/pacnerf-exact');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.inputs,a.baseline,a.output)
