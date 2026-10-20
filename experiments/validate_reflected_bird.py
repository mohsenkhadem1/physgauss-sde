"""Independent uncertainty replay, only after a fresh four-chain observed gate.

Run with --worker pointing to the immutable experiment checkout. Importing its
math is deliberate: validation must not accidentally use a later model version.
The NumPy state decode and ensemble scores are independent implementations.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch


def require_gate(gate):
    if (not gate['all_monitored_quantities_pass'] or gate['draws_per_chain']<2000 or gate['warmup']<500
            or gate['retained_invalid'] or gate['retained_divergent']):
        raise ValueError('Bird has not passed convergence; future labels must remain unopened')


@torch.no_grad()
def run(args):
    worker=Path(args.worker).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker,text=True).strip()!=args.source_ref:
        raise ValueError('Independent validation worker commit differs')
    sys.path.insert(0,str(worker/'scripts'))
    from aggregate_pacnerf_reflected import aggregate
    # Recompute every monitored scalar from raw independently seeded shards.
    gate=aggregate(args.inputs,args.shards,Path(args.output)/'observed-replay',args.draws)
    require_gate(gate)
    from pacnerf_floor import make_floor_model,FloorRenderer
    from pacnerf_recovery import restore_correlations
    from pacnerf_conditioned_transport import make_target
    from verify_pacnerf_discrepancy import independent_integrals
    from physgauss.pacnerf_model import PACRenderer
    from pacnerf_discrepancy import spatial_whiten
    from pacnerf_prediction_metrics import prediction_scores
    from compare_pacnerf_discrepancy import uncertainty_scores
    from verify_pacnerf import verify_case as verify_baseline
    from publish_pacnerf_bird_check import readiness
    root=Path(args.shards)/'chain-0'
    model,_=make_floor_model(args.inputs,8)
    restore_correlations(model,gate['source_protocol']['frozen_model'])
    target=make_target(model,torch.load(root/'reference.pt',weights_only=True))
    merged=torch.load(Path(args.output)/'observed-replay/merged.pt',weights_only=True)
    white=merged['white'].flatten(0,1);physical=target.decode(white)
    indices=torch.linspace(0,len(white)-1,16).round().long();probes=physical[indices]
    integrals=independent_integrals(model,probes[:4])
    q,_=model.decode(probes)
    dense=PACRenderer(model.data,model.basis);dense.colors=model.renderer.colors;dense=FloorRenderer(dense)
    rgb=dense(q[:,model.frames],model.specs);fast=model.renderer(q[:,model.frames],model.specs)
    rgb_error=float((rgb-fast).abs().max())
    error=spatial_whiten(rgb-model.obs[None],model.rho_x,model.rho_y)/model.scales(probes)[2][:,None,None,None,None]
    nll_error=float((.5*error.square().flatten(1).sum(-1)-.5*model.image_residual(probes).square().sum(-1)).abs().max())
    heights=model.floor_rest.numpy()[None,None]+np.einsum('btm,nm->btn',q[:,1:].numpy(),model.floor_basis.numpy())
    floor_error=float(np.abs(np.logaddexp(0.,-heights/model.floor_tolerance).sum((1,2))-model.floor_penalty(probes).numpy()).max())
    if rgb_error>5e-13 or nll_error>1e-8 or floor_error>1e-7:raise ValueError('Independent renderer/floor replay failed')
    # Decode positions independently of model.decode and the Torch score path.
    z=physical.numpy();m=model.modes
    poses=np.zeros((len(z),14,m));scales=model.scale[2+m:model.base_dimension].numpy().reshape(13,2*m)[:,:m]
    poses[:,1:]=z[:,2+m:model.base_dimension].reshape(len(z),13,2*m)[:,:,:m]*scales[None]
    selected=torch.linspace(0,len(z)-1,min(2000,len(z))).round().long().numpy()
    positions=model.data['gaussians'].numpy()[None,None]+np.einsum('btm,gcm->btgc',poses[selected],model.renderer.gaussian_basis.numpy())
    # First access to withheld particle positions occurs below this line.
    manifest=json.loads((Path(args.inputs)/'bird-manifest.json').read_text())
    evaluation=Path(args.inputs)/'bird-evaluation.pt'
    if hashlib.sha256(evaluation.read_bytes()).hexdigest()!=manifest['evaluation_sha256']:raise ValueError('Evaluation hash differs')
    truth=torch.load(evaluation,weights_only=True)['particles'].numpy();gap=list(range(4,10))
    fresh=prediction_scores(positions[:,gap],truth[gap]);saved=uncertainty_scores(model,physical,args.inputs)
    differences=[abs(fresh['mean_crps_m']-saved['mean_crps_m'])]
    for a,b in zip(fresh['coverage_curve'],saved['coverage_curve']):
        differences.extend(abs(a[key]-b[key]) for key in ['nominal','coverage','mean_width_m','mean_interval_score_m'])
    if max(differences)>1e-9:raise ValueError('Independent NumPy/Torch uncertainty scores differ')
    old=verify_baseline(args.baseline,args.inputs,8)
    penetration=float((model.floor_heights(physical)<-.02).double().mean())
    summary={'valid_for_ranking':True,'settings':{'model':model.specification()},
        'posterior_ground_diagnostic':{'all_constrained_points_fraction_below_height':{'-0.02':penetration}}}
    proof={'all_pass':True,'independent_prediction_scores':fresh}
    decision=readiness(summary,proof,old)
    report={'convergence':gate,'independent_prediction_scores':fresh,'baseline_independent_verification':old,
        'readiness':decision,'independent_integrals':integrals,'maximum_dense_renderer_rgb_error':rgb_error,
        'maximum_image_nll_replay_error':nll_error,'maximum_floor_nll_replay_error':floor_error,
        'maximum_numpy_torch_score_difference':max(differences),
        'qualification':'Observed-only convergence passed before evaluation. One deterministic development scene; not independent real-world uncertainty calibration.'}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'uncertainty-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(decision),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['worker','source-ref','inputs','shards','baseline','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--draws',type=int,choices=[2000,4000,8000],required=True)
    run(p.parse_args())
