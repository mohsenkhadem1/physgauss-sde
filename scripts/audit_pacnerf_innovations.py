"""Check the causal coordinate change against saved physical-path densities.

This is a post-hoc algebra audit, not a replacement sampling experiment. It
reads fitted states and prepared initial geometry/images, never future geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pacnerf_comparison import cases,make_model,Target,make_target,write
from pacnerf_innovations import InnovationModel
from physgauss.cli import source_fingerprint


@torch.no_grad()
def audit(raw,inputs):
    raw=Path(raw);rows=[]
    for index,case in enumerate(cases()):
        name=f'{case[0]}__{case[1]}views__{case[2]}__{case[3]}'
        folder=raw/f'chains-{index}'/name
        if not (folder/'chains.pt').exists():continue
        reference=torch.load(folder/'reference.pt',weights_only=True)
        chains=torch.load(folder/'chains.pt',weights_only=True)
        model,_=make_model(inputs,index);target=make_target(model,reference)
        selected=torch.linspace(0,chains['white'].shape[1]-1,4).round().long()
        physical=target.decode(chains['white'][:,selected].flatten(0,1))
        innovation=InnovationModel(model)
        coordinates,jac=innovation.transform(physical,inverse=True)
        reconstructed,forward_jac=innovation.transform(coordinates)
        round_trip=float((reconstructed-physical).abs().max())
        jac_error=float((forward_jac-jac).abs().max())
        density_error=float((model.objective(physical)-jac-innovation.objective(coordinates)).abs().max())
        passed=round_trip<1e-6 and jac_error<1e-6 and density_error<1e-5
        row={'case_index':index,'case':name,'probes':len(physical),
            'maximum_standardized_path_round_trip_error':round_trip,
            'maximum_log_jacobian_round_trip_error':jac_error,
            'maximum_transformed_negative_log_density_error':density_error,
            'chains_sha256':hashlib.sha256((folder/'chains.pt').read_bytes()).hexdigest(),
            'all_pass':passed}
        rows.append(row);print(json.dumps(row),flush=True)
        if not passed:raise ValueError('Innovation transformation differs from original physical density')
    if not rows:raise ValueError('No saved chains found')
    return {'source_fingerprint':source_fingerprint(),'all_pass':all(r['all_pass'] for r in rows),
        'cases':rows,'qualification':'algebra/round-trip probes only; production sampler remains unchanged; does not demonstrate better mixing, calibration or physical accuracy'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True)
    p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    write(a.output,audit(a.raw,a.inputs))
