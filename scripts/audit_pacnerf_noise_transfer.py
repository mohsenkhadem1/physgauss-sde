"""Post-hoc descriptive feature support audit; never retrains the frozen noise.

Compare a fixed stratified subset of saved PAC posterior states to the documented
broad exploratory training component. This is not the full empirical training
support and being inside a bounding box does not establish distributional validity.
"""
import argparse
import json
from pathlib import Path
import torch
from pacnerf_comparison import make_model,cases,Target
from physgauss.cli import source_fingerprint


def local_position_variance(model,z):
    """One-step variance at identical paths; no force or posterior decomposition."""
    original=model.noise
    try:
        model.noise='constant';constant=model.terms(z)[1]
        model.noise='neural';neural=model.terms(z)[1]
    finally:model.noise=original
    m=model.modes;floor=torch.zeros_like(constant)
    for point,weight in zip([-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526],
        [.3478548451374538,.6521451548625461,.6521451548625461,.3478548451374538]):
        f=model.transition(z,model.dt*(point+1)/2)[:,:,m:]
        floor+=weight*model.dt/2*model.acc_noise_floor**2*(f@f.transpose(-1,-2))[:,None]
    for total in [constant,neural]:
        if float(torch.linalg.eigvalsh(total-floor).min()) < -1e-10:
            raise ValueError('Floor is not a positive-semidefinite covariance component')
    phi=model.renderer.gaussian_basis.flatten(0,1)
    metric=phi.T@phi/len(phi)
    projected=lambda covariance: (covariance[...,:m,:m]*metric.T).sum((-1,-2))
    a,b,c=map(projected,[constant,neural,floor])
    return b/a,c/a,c/b


@torch.no_grad()
def audit(raw,inputs,output):
    raw,inputs=Path(raw),Path(inputs)
    # train_calibrated() draws the broad half of training features from
    # strain U(-.25,.45), speed U(-1.5,1.5). The other half is operating data.
    # Their original byte hashes did not reproduce on this host; do not pretend
    # a new trajectory replay is an exact copy of that archived training data.
    upper=torch.tensor([.45,1.5],dtype=torch.float64)
    records=[]
    for index,(scene,views,noise,force) in enumerate(cases()):
        name=f'{scene}__{views}views__{noise}__{force}'
        matches=list((raw/f'chains-{index}').glob(f'**/{name}/chains.pt'))
        if not matches:continue
        model,_=make_model(inputs,index)
        reference=torch.load(matches[0].parent/'reference.pt',weights_only=True)
        chains=torch.load(matches[0],weights_only=True);white=chains['white']
        indices=torch.linspace(0,white.shape[1]-1,min(64,white.shape[1])).round().long()
        target=Target(model,reference['mean'],reference['precision'])
        z=target.decode(white[:,indices].flatten(0,1));features=[];variance_parts=[]
        for batch in z.split(16):
            features.append(model.terms(batch)[3].reshape(-1,2))
            variance_parts.append(local_position_variance(model,batch))
        feature=torch.cat(features);absolute=feature.abs()
        ratio=model.network(feature)/model.constant
        finite=json.loads((raw/f'results-{index}'/name/'summary.json').read_text())['valid_for_ranking']
        records.append({'case':name,'finite_chain_checks_pass':finite,'posterior_states':len(z),'feature_pairs':len(feature),
            'feature_magnitude_quantiles':torch.quantile(absolute,absolute.new_tensor([0.,.05,.5,.95,1.]),dim=0).tolist(),
            'fraction_outside_documented_broad_component_magnitude_box':float((absolute>upper).any(-1).double().mean()),
            'frozen_neural_over_constant_amplitude_quantiles':torch.quantile(ratio,ratio.new_tensor([0.,.05,.5,.95,1.])).tolist(),
            'local_one_step_particle_coordinate_variance':{
                key:torch.quantile(torch.cat([p[i].flatten() for p in variance_parts]),ratio.new_tensor([0.,.05,.5,.95,1.])).tolist()
                for i,key in enumerate(['neural_over_constant','floor_fraction_constant','floor_fraction_neural'])}})
    result={'source_fingerprint':source_fingerprint(),'quantile_probabilities':[0.,.05,.5,.95,1.],
        'feature_order':['edge_strain','radial_speed_m_per_s'],
        'broad_training_component_magnitude_upper_bounds':upper.tolist(),
        'cases':records,
        'local_variance_definition':'same saved preceding paths and parameters; trace of one-step process position covariance projected into 128 Gaussian centers, divided by 384 coordinates; excludes uncertain global loads, propagation beyond one interval and posterior/epistemic uncertainty; quantiles pool sampled states and 13 intervals',
        'qualification':'post-hoc comparison to documented broad exploration component, not the full empirical training support, PAC noise calibration or a causal error decomposition; archived operating-state byte hashes did not reproduce on this host; frozen network and production experiment unchanged'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    audit(a.raw,a.inputs,a.output)
