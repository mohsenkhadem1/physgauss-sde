"""Fixed-budget Gaussian forward experiments on the crossed engineering cohort.

Same force-linear Gaussian architecture and ablation as the earlier pilot;
only the excitation/material/support design changes. No observations enter:
physical hypotheses are supplied, not inferred. Final checkpoint only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    from campaign_hybrid_response import HybridGaussianResponse
    from campaign_hybrid_predictor import prepare_features
    from campaign_material_gaussians import MaterialGaussians
    from train_campaign_hybrid_response import load_data, subset
except ImportError:
    from experiments.campaign_hybrid_response import HybridGaussianResponse
    from experiments.campaign_hybrid_predictor import prepare_features
    from experiments.campaign_material_gaussians import MaterialGaussians
    from experiments.train_campaign_hybrid_response import load_data, subset

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('fit','development_force_transfer','unseen_middle_material')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def metrics(prediction, truth, rest, times, roles, ids, tets):
    """Scores are calculated only after the fixed final checkpoint exists."""
    rows = []
    for i in range(len(rest)):
        kernels = MaterialGaussians.from_tetrahedra(torch.as_tensor(rest[i],dtype=torch.float64),
                    torch.as_tensor(tets,dtype=torch.long),1.)
        J = kernels.transport(torch.as_tensor(prediction[i],dtype=torch.float64),validate=False).jacobians.numpy()
        rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
        rows.append(dict(id=str(ids[i]), evaluation_role=str(roles[i]),
            position_rmse_m=rmse((prediction[i]-truth[i])[times>0]),
            after_1s_rmse_m=rmse((prediction[i]-truth[i])[times>1]),
            static_rmse_m=rmse((truth[i]-rest[i])[times>0]),
            predicted_motion_rms_m=rmse((prediction[i]-rest[i])[times>0]),
            actual_motion_rms_m=rmse((truth[i]-rest[i])[times>0]),
            minimum_J=float(J.min()), inverted_tetrahedron_frames=int((J<=0).sum())))
    summaries = {}
    for role in ROLES:
        group = [r for r in rows if r['evaluation_role']==role]
        if len(group)!=32:
            raise ValueError('Expected 32 cases per frozen experimental role')
        summaries[role] = dict(cases=len(group),
            mean_position_rmse_m=float(np.mean([r['position_rmse_m'] for r in group])),
            mean_after_1s_rmse_m=float(np.mean([r['after_1s_rmse_m'] for r in group])),
            mean_static_rmse_m=float(np.mean([r['static_rmse_m'] for r in group])),
            inverted_tetrahedron_frames=sum(r['inverted_tetrahedron_frames'] for r in group))
    return dict(cases=rows, role_summaries=summaries,
                supplied_physics=True, observed_motion_frames=0,
                unseen_layout_generalization=False, calibrated_uncertainty=False)


def run(dataset, output, *, device='cuda:1', arm='exchange', steps=4000, seed=20261022):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError('Fresh training output required')
    if arm not in ('exchange','no_exchange') or steps<=0:
        raise ValueError('Declared arm and positive fixed budget required')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    data, arrays = load_data(dataset,device)
    roles = arrays['evaluation_role']
    if len(roles)!=96 or any(np.sum(roles==r)!=32 for r in ROLES):
        raise ValueError('Exact crossed96 cohort required')
    fit = np.flatnonzero(roles=='fit')
    if not np.array_equal(arrays['train_mask'],roles=='fit'):
        raise ValueError('Train mask differs from the predeclared roles')
    tets = torch.as_tensor(arrays['tets'],dtype=torch.long,device=device)
    faces = torch.as_tensor(arrays['faces'],dtype=torch.long,device=device)
    model = HybridGaussianResponse(data['rest'][0],tets,faces,width=64,layers=3,
                                  spatial_exchange=arm=='exchange').to(device)
    with torch.no_grad():
        prepared = prepare_features(data,tets,faces)
    fit_index = torch.as_tensor(fit,device=device)
    fit_data = subset(data,fit_index)
    fit_prepared = tuple(v[fit_index] for v in prepared)
    target = torch.as_tensor(arrays['q'][fit],dtype=torch.float32,device=device)
    scales = ((target-fit_data['rest'][:,None]).square().mean((1,2,3))).sqrt().clamp_min(.0005)
    sources = [Path(__file__), *[ROOT/'experiments'/n for n in (
        'campaign_hybrid_response.py','campaign_hybrid_predictor.py',
        'campaign_material_gaussians.py','train_campaign_hybrid_response.py')]]
    pins = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    output.mkdir(parents=True,exist_ok=False)
    (output/'source').mkdir()
    for p in sources:
        (output/'source'/p.name).write_bytes(p.read_bytes())
    protocol = dict(schema='crossed-known-hypothesis-Gaussian-forward-v0.1',
        dataset_sha256=sha(dataset), source_sha256=pins, arm=arm, seed=seed,
        device=str(device), steps=steps, width=64,layers=3,batch=4,query_times=8,
        learning_rate=.001,weight_decay=1e-5, fit_cases=32,
        selection='Fixed final checkpoint; no development or middle-material score during fitting',
        information='Supplied true material/support hypothesis, known rest/hardware/commands; zero measured motion frames',
        loss='TRAIN-balanced coordinate MSE plus .05*mean(relu(.2-J)^2)',
        fit_normalization='TRAIN-only RMS motion with .5mm floor; never a model feature',
        limitations=['Same four hardware contexts, not unseen layouts',
                     'Three joint material tuples, not four independently identified effects',
                     'Engineering FEM data, not native or continuum truth',
                     'No sparse-view posterior or uncertainty calibration claim'],
        raw_outputs_LOCAL=True, GitHub_Actions_enabled=False)
    write(output/'protocol.json',protocol)
    write(output/'execution.json',dict(terminal=False,completed=False))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    vertices = fit_data['rest'][:,tets]
    inverse = torch.linalg.inv((vertices[:,:,1:]-vertices[:,:,:1]).transpose(-1,-2))
    history = []
    started = time.monotonic()
    try:
        for step in range(1,steps+1):
            ids = torch.as_tensor(rng.integers(len(fit),size=4),device=device)
            tids = torch.as_tensor(np.sort(rng.choice(np.arange(1,61),8,replace=False)),device=device)
            batch = subset(fit_data,ids)
            prediction = model(batch,time_indices=tids,prepared=tuple(v[ids] for v in fit_prepared))['positions_m']
            position_loss = ((prediction-target[ids][:,tids])/scales[ids,None,None,None]).square().mean()
            sample = torch.as_tensor(rng.choice(len(tets),128,replace=False),device=device)
            v = prediction[:,:,tets[sample]]
            J = torch.linalg.det((v[...,1:,:]-v[...,:1,:]).transpose(-1,-2)@inverse[ids][:,None,sample])
            loss = position_loss+.05*torch.relu(.2-J).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite fit loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%500==0 or step==steps:
                row = dict(step=step,fit_loss=float(loss.detach()),elapsed_s=time.monotonic()-started)
                history.append(row)
                write(output/'history.json',history)
                print(json.dumps(row),flush=True)
                if any(sha(ROOT/p)!=digest for p,digest in pins.items()):
                    raise RuntimeError('Training source changed')
        torch.save(dict(model_state=model.state_dict(),protocol=protocol,step=steps),output/'final.pt')
        # Evaluation is deliberately after the final checkpoint is immutable.
        model.eval()
        predictions = []
        with torch.no_grad():
            for lo in range(0,96,4):
                index = torch.arange(lo,min(lo+4,96),device=device)
                predictions.append(model(subset(data,index),prepared=tuple(v[index] for v in prepared))['positions_m'].cpu().numpy())
        prediction = np.concatenate(predictions)
        report = metrics(prediction,arrays['q'],arrays['rest'],arrays['times'],roles,arrays['ids'],arrays['tets'])
        write(output/'metrics.json',report)
        np.savez_compressed(output/'predictions.npz',predicted_positions_m=prediction,
                            times_s=arrays['times'],ids=arrays['ids'],train_mask=arrays['train_mask'],evaluation_role=roles)
        write(output/'execution.json',dict(terminal=True,completed=True,elapsed_s=time.monotonic()-started,
            checkpoint_sha256=sha(output/'final.pt'),predictions_sha256=sha(output/'predictions.npz')))
        return report['role_summaries']
    except BaseException as error:
        write(output/'execution.json',dict(terminal=True,completed=False,error=type(error).__name__,message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--arm',choices=['exchange','no_exchange'],required=True)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=20261022)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset,args.output,device=args.device,arm=args.arm,steps=args.steps,seed=args.seed)))
