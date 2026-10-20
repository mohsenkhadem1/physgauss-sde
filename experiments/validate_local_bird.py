"""Independently replay local Bird convergence before opening withheld labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from bird_local import aggregate,load_model,make_target,sha,write
from validate_reflected_bird import require_gate


@torch.no_grad()
def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # This rereads the raw chains, validates frozen provenance and recomputes
    # all 733 diagnostics. Evaluation tensors are not opened before it passes.
    observed=argparse.Namespace(inputs=args.inputs,gate=args.gate,shards=args.shards,
        output=str(out/'observed-replay'),draws=args.draws)
    aggregate(observed)
    gate=json.loads((out/'observed-replay/convergence.json').read_text())
    require_gate(gate)
    from pacnerf_floor import FloorRenderer
    from physgauss.pacnerf_model import PACRenderer
    from verify_pacnerf_discrepancy import independent_integrals
    from pacnerf_discrepancy import spatial_whiten
    from pacnerf_prediction_metrics import prediction_scores
    from compare_pacnerf_discrepancy import uncertainty_scores
    from verify_pacnerf import verify_case as verify_baseline
    from publish_pacnerf_bird_check import readiness

    parent=json.loads(Path(args.gate).read_text())
    model=load_model(args.inputs,parent)
    reference=torch.load(Path(args.shards)/'chain-0/reference.pt',weights_only=True)
    target=make_target(model,reference)
    merged=torch.load(out/'observed-replay/merged.pt',weights_only=True)
    white=merged['white'].flatten(0,1)
    physical=target.decode(white)
    indices=torch.linspace(0,len(white)-1,16).round().long()
    probes=physical[indices]
    integral_proof=independent_integrals(model,probes[:4])
    q,_=model.decode(probes)
    dense=PACRenderer(model.data,model.basis)
    dense.colors=model.renderer.colors;dense=FloorRenderer(dense)
    rgb=dense(q[:,model.frames],model.specs)
    fast=model.renderer(q[:,model.frames],model.specs)
    rgb_error=float((rgb-fast).abs().max())
    residual=spatial_whiten(rgb-model.obs[None],model.rho_x,model.rho_y)/model.scales(probes)[2][:,None,None,None,None]
    nll_error=float((.5*residual.square().flatten(1).sum(-1)-.5*model.image_residual(probes).square().sum(-1)).abs().max())
    heights=model.floor_rest.numpy()[None,None]+np.einsum('btm,nm->btn',q[:,1:].numpy(),model.floor_basis.numpy())
    floor_error=float(np.abs(np.logaddexp(0.,-heights/model.floor_tolerance).sum((1,2))-model.floor_penalty(probes).numpy()).max())
    if rgb_error>5e-13 or nll_error>1e-8 or floor_error>1e-7:
        raise ValueError('Independent renderer/floor replay failed')

    # Independent NumPy state layout and position decoder.
    z=physical.numpy();m=model.modes
    poses=np.zeros((len(z),14,m))
    scale=model.scale[2+m:model.base_dimension].numpy().reshape(13,2*m)[:,:m]
    poses[:,1:]=z[:,2+m:model.base_dimension].reshape(len(z),13,2*m)[:,:,:m]*scale[None]
    selected=torch.linspace(0,len(z)-1,min(2000,len(z))).round().long().numpy()
    positions=model.data['gaussians'].numpy()[None,None]+np.einsum('btm,gcm->btgc',poses[selected],model.renderer.gaussian_basis.numpy())
    manifest=json.loads((Path(args.inputs)/'bird-manifest.json').read_text())
    evaluation=Path(args.inputs)/'bird-evaluation.pt'
    if sha(evaluation)!=manifest['evaluation_sha256']:
        raise ValueError('Evaluation hash differs')
    # FIRST access to withheld positions: all observed-data gates passed above.
    truth=torch.load(evaluation,weights_only=True)['particles'].numpy()
    gap=list(range(4,10))
    independent=prediction_scores(positions[:,gap],truth[gap])
    torch_scores=uncertainty_scores(model,physical,args.inputs)
    differences=[abs(independent['mean_crps_m']-torch_scores['mean_crps_m'])]
    for a,b in zip(independent['coverage_curve'],torch_scores['coverage_curve']):
        differences.extend(abs(a[k]-b[k]) for k in ['nominal','coverage','mean_width_m','mean_interval_score_m'])
    if max(differences)>1e-9:
        raise ValueError('Independent NumPy/Torch uncertainty scores differ')
    baseline=verify_baseline(args.baseline,args.inputs,8)
    penetration=float((model.floor_heights(physical)<-.02).double().mean())
    summary={'valid_for_ranking':True,'settings':{'model':model.specification()},
        'posterior_ground_diagnostic':{'all_constrained_points_fraction_below_height':{'-0.02':penetration}}}
    decision=readiness(summary,{'all_pass':True,'independent_prediction_scores':independent},baseline)
    report={'convergence':gate,'independent_prediction_scores':independent,
        'baseline_independent_verification':baseline,'readiness':decision,
        'independent_integrals':integral_proof,'maximum_dense_renderer_rgb_error':rgb_error,
        'maximum_image_nll_replay_error':nll_error,'maximum_floor_nll_replay_error':floor_error,
        'maximum_numpy_torch_score_difference':max(differences),
        'validator_sha256':sha(__file__),
        'qualification':'One deterministic development scene; convergence and score replay do not establish independent uncertainty calibration.'}
    write(out/'uncertainty-validation.json',report)
    print(json.dumps(decision),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/inputs')
    p.add_argument('--gate',default='runs/bird-8000/convergence.json')
    p.add_argument('--baseline',default='runs/baseline')
    p.add_argument('--shards',required=True)
    p.add_argument('--draws',type=int,required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(a)
