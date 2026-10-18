"""Separate conditional interval arithmetic from model error; evaluation only.

Future particle data are read ONLY here, never by the discrepancy-model fitter.
The projected-truth audit asks if poor coverage survives removal of truncation
error. Axis-specific bias separates a confidently wrong mean from MC error.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pacnerf_comparison import cases, make_model, write


@torch.no_grad()
def audit(report, inputs, output):
    records=[]
    for index,(scene,views,noise,force) in enumerate(cases()):
        name=f'{scene}__{views}views__{noise}__{force}'
        folder=Path(report)/'cases'/name
        if not (folder/'posterior.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text())
        posterior=json.loads((folder/'posterior.json').read_text())
        model,_=make_model(inputs,index)
        truth=torch.load(Path(inputs)/f'{scene}-evaluation.pt',weights_only=True)['particles'].numpy()
        mean,lo,hi=[np.asarray(posterior[k]) for k in ['particle_mean_m','particle_q05_m','particle_q95_m']]
        phi=model.renderer.gaussian_basis.numpy().reshape(-1,model.modes)
        rest=model.data['gaussians'].numpy()
        q=np.linalg.lstsq(phi,(truth-rest[None]).reshape(14,-1).T,rcond=None)[0].T
        projected=rest[None]+(q@phi.T).reshape(truth.shape)
        gap=list(range(4,10)) if views==4 else list(range(5,9))
        error=mean[gap]-truth[gap]
        qmean=torch.tensor(posterior['modal_pose_mean_m'],dtype=torch.float64)
        rgb=model.renderer(qmean[model.frames][None],model.specs)[0]
        residual=(rgb-model.obs).numpy()
        oracle_rgb=model.renderer(torch.tensor(q[model.frames],dtype=torch.float64)[None],model.specs)[0]
        oracle_residual=(oracle_rgb-model.obs).numpy()
        # This post-hoc oracle renderer comparison must never select the new
        # priors, initialize inference, or be described as a sparse-view fit.
        oracle_image_nll=.5*np.sum((oracle_residual/.08)**2)
        inferred_image_nll=.5*np.sum((residual/.08)**2)
        def correlation(a,b):
            a,b=a.flatten(),b.flatten();a=a-a.mean();b=b-b.mean()
            return float(a@b/np.sqrt((a@a)*(b@b)))
        records.append({'case':name,'finite_chain_checks_pass':summary['valid_for_ranking'],
            'gap_rmse_m':float(np.sqrt(np.mean(error**2))),
            'gap_bias_xyz_m':error.mean((0,1)).tolist(),
            'gap_rmse_xyz_m':np.sqrt(np.mean(error**2,axis=(0,1))).tolist(),
            'nominal_90_coverage':float(np.mean((truth[gap]>=lo[gap])&(truth[gap]<=hi[gap]))),
            'nominal_90_coverage_xyz':np.mean((truth[gap]>=lo[gap])&(truth[gap]<=hi[gap]),axis=(0,1)).tolist(),
            'projected_truth_90_coverage':float(np.mean((projected[gap]>=lo[gap])&(projected[gap]<=hi[gap]))),
            'representation_only_rmse_m':float(np.sqrt(np.mean((projected[gap]-truth[gap])**2))),
            'mean_90_width_m':float((hi[gap]-lo[gap]).mean()),
            'observed_rgb_rmse':float(np.sqrt(np.mean(residual**2))),
            'evaluation_only_projected_truth_observed_rgb_rmse':float(np.sqrt(np.mean(oracle_residual**2))),
            'evaluation_only_oracle_minus_inferred_image_nll':float(oracle_image_nll-inferred_image_nll),
            'observed_residual_adjacent_x_correlation':correlation(residual[:,:,1:],residual[:,:,:-1]),
            'observed_residual_adjacent_y_correlation':correlation(residual[:,1:],residual[:,:-1]),
            'original_rgb_sigma':.08,
            'interval_definition':'latent modal trajectory conditional on fixed noise/renderer/contact/subspace, not total physical prediction uncertainty'})
    result={'cases':records,'future_truth_used':'post-hoc evaluation only',
        'findings':[
            'Original interval quantiles independently replayed from raw draws: not a standard-error-for-standard-deviation substitution.',
            'Converged chains sample the specified target, not necessarily a calibrated physical model.',
            'Fixed transferred diffusion and fixed restricted loads exclude uncertainty in model-error amplitudes.',
            'Independent-pixel fixed-sigma rendering excludes spatial residual correlation and renderer bias.',
            'Representation projection and residual correlation are descriptive diagnostics, not a causal decomposition or calibration fit.'],
        'correction_protocol':'No future particle or held-out image residual determines new priors or hyperparameters; inference uses supplied initial geometry/appearance and declared sparse RGB only.'}
    write(Path(output)/'audit.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=[r['case'].replace('__',' / ') for r in records]
    axes[0].barh(labels,[r['nominal_90_coverage'] for r in records],label='Full truth')
    axes[0].scatter([r['projected_truth_90_coverage'] for r in records],range(len(records)),color='black',label='Projected truth')
    axes[0].axvline(.9,color='red',linestyle='--');axes[0].set_xlim(0,1);axes[0].set_xlabel('90% interval coverage');axes[0].legend(fontsize=7)
    axes[1].scatter([1000*r['mean_90_width_m'] for r in records],[1000*r['gap_rmse_m'] for r in records]);axes[1].set_xlabel('Mean interval width (mm)');axes[1].set_ylabel('Gap reconstruction RMSE (mm)')
    fig.tight_layout();fig.savefig(Path(output)/'audit.png',dpi=150);plt.close(fig)
    print(json.dumps({'cases':len(records),'report':str(Path(output)/'audit.json')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',default='examples/pacnerf-exact');p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.report,a.inputs,a.output)
