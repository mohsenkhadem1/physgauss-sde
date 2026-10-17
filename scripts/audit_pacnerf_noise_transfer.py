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
        z=target.decode(white[:,indices].flatten(0,1));features=[]
        for batch in z.split(16):features.append(model.terms(batch)[3].reshape(-1,2))
        feature=torch.cat(features);absolute=feature.abs()
        ratio=model.network(feature)/model.constant
        records.append({'case':name,'posterior_states':len(z),'feature_pairs':len(feature),
            'feature_magnitude_quantiles':torch.quantile(absolute,absolute.new_tensor([0.,.05,.5,.95,1.]),dim=0).tolist(),
            'fraction_outside_documented_broad_component_magnitude_box':float((absolute>upper).any(-1).double().mean()),
            'frozen_neural_over_constant_amplitude_quantiles':torch.quantile(ratio,ratio.new_tensor([0.,.05,.5,.95,1.])).tolist()})
    result={'quantile_probabilities':[0.,.05,.5,.95,1.],
        'feature_order':['edge_strain','radial_speed_m_per_s'],
        'broad_training_component_magnitude_upper_bounds':upper.tolist(),
        'cases':records,'qualification':'post-hoc comparison to documented broad exploration component, not the full empirical training support, PAC noise calibration or a causal error decomposition; archived operating-state byte hashes did not reproduce on this host; frozen network and production experiment unchanged'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    audit(a.raw,a.inputs,a.output)
