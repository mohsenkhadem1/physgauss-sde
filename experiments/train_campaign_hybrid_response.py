"""Exploratory V02 force-linear Gaussian response pilot on the unchanged split.

Twelve fitting action/layout pairs and four fixed cross-action diagnostic pairs
share four known layouts. This is not a fresh-layout or sparse-image benchmark.
No held-pair losses select weights, checkpoints, steps or hyperparameters.
"""
from __future__ import annotations
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

try:
    from campaign_hybrid_predictor import prepare_features
    from campaign_hybrid_response import HybridGaussianResponse as HybridGaussianPredictor
    from campaign_material_gaussians import MaterialGaussians
except ImportError:
    from experiments.campaign_hybrid_predictor import prepare_features
    from experiments.campaign_hybrid_response import HybridGaussianResponse as HybridGaussianPredictor
    from experiments.campaign_material_gaussians import MaterialGaussians

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_data(path,device):
    with np.load(path,allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    # Only this explicit allowlist can enter the model. IDs/q/split are audit/labels.
    names=['rest','material','support_centers','support_active','fixture_mask','support_offsets',
        'handle_patches','handle_centers','handle_masses','handle_inertias','handle_com_offsets',
        'handle_initial_rotations','controls','times']
    data={}
    for name in names:
        a=arrays[name]
        dtype=torch.bool if a.dtype.kind=='b' else torch.long if a.dtype.kind in 'iu' else torch.float32
        data[name]=torch.as_tensor(a,dtype=dtype,device=device)
    return data,arrays


def subset(data,indices):
    return {k:v if k=='times' else v[indices] for k,v in data.items()}


def evaluate(model,data,arrays,prepared,output):
    with torch.no_grad():prediction=model(data,prepared=prepared)['positions_m'].cpu().numpy()
    truth=arrays['q'];rest=arrays['rest'];times=arrays['times']
    rows=[];kernel_outputs={}
    for i in range(len(rest)):
        error=prediction[i]-truth[i]
        forecast=times>1.
        temporal=times>0
        pred=prediction[i].astype(np.float64);reference=rest[i].astype(np.float64)
        material=MaterialGaussians.from_tetrahedra(torch.from_numpy(reference),torch.from_numpy(arrays['tets']).long(),float(arrays['material'][i,1]))
        # Invalid deformation stays a reported failure; no covariance/J repair.
        transported=material.transport(torch.from_numpy(pred),validate=False)
        J=transported.jacobians.numpy()
        rmse=lambda value:float(np.sqrt(np.mean(value**2)))
        row={'id':str(arrays['ids'][i]),'fit_pair':bool(arrays['train_mask'][i]),
            'position_rmse_m':rmse(error[temporal]),'future_after_1s_rmse_m':rmse(error[forecast]),
            'static_rmse_m':rmse((truth[i]-rest[i])[temporal]),
            'predicted_rms_motion_m':rmse((prediction[i]-rest[i])[temporal]),
            'actual_rms_motion_m':rmse((truth[i]-rest[i])[temporal]),
            'minimum_J':float(J.min()),'inverted_tetrahedron_frames':int((J<=0).sum()),
            'represented_mass_kg':float(material.total_mass),
            'input_measured_motion_frames':0,'input_physics':'supplied true hypothesis for this forward diagnostic'}
        rows.append(row)
        if not row['fit_pair']:
            kernel_outputs['centers_'+str(i)]=transported.centers.numpy()
            kernel_outputs['covariances_'+str(i)]=transported.covariances.numpy()
            kernel_outputs['jacobians_'+str(i)]=J
    for mask,name in [(arrays['train_mask'],'fitting'),(~arrays['train_mask'],'held_cross_action')]:
        selected=[r for r,on in zip(rows,mask) if on]
        write(output/(name+'-metrics.json'),{'pairs':selected,'mean_rmse_m':float(np.mean([r['position_rmse_m'] for r in selected])),
            'mean_static_rmse_m':float(np.mean([r['static_rmse_m'] for r in selected])),
            'mean_future_after_1s_rmse_m':float(np.mean([r['future_after_1s_rmse_m'] for r in selected])),
            'sum_inverted_tetrahedron_frames':sum(r['inverted_tetrahedron_frames'] for r in selected),
            'fresh_layout_generalization_established':False,'sparse_view_inference_established':False,
            'calibrated_uncertainty_established':False,'production_dataset_admitted':False})
    np.savez_compressed(output/'predictions.npz',predicted_positions_m=prediction,times_s=times,ids=arrays['ids'],train_mask=arrays['train_mask'])
    np.savez_compressed(output/'held-material-gaussians.npz',**kernel_outputs)
    return rows


def run(dataset,output,device,arm,steps,seed):
    if output.exists():raise ValueError('Fresh pilot output directory required')
    if steps<=0:raise ValueError('Positive fixed optimization budget required')
    torch.set_num_threads(2);torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    data,arrays=load_data(dataset,device)
    fit=np.flatnonzero(arrays['train_mask'])
    if len(fit)!=12 or len(arrays['rest'])!=16:raise ValueError('Exact predeclared 12/4 cross-action diagnostic required')
    tets=torch.as_tensor(arrays['tets'],dtype=torch.long,device=device)
    faces=torch.as_tensor(arrays['faces'],dtype=torch.long,device=device)
    model=HybridGaussianPredictor(data['rest'][0],tets,faces,width=64,layers=3,spatial_exchange=arm=='exchange').to(device)
    with torch.no_grad():prepared=prepare_features(data,tets,faces)
    fit_q=torch.as_tensor(arrays['q'][fit],dtype=torch.float32,device=device)
    fit_scales=torch.sqrt(((fit_q-data['rest'][fit,None])**2).mean((1,2,3))).clamp_min(.0005)
    fit_data=subset(data,torch.as_tensor(fit,device=device))
    fit_prepared=tuple(v[fit] for v in prepared)
    sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for name in ['campaign_hybrid_predictor.py','campaign_hybrid_response.py','campaign_material_gaussians.py']:
        p=Path(__file__).with_name(name);sources[str(p.relative_to(ROOT))]=sha(p)
    output.mkdir(parents=True,exist_ok=False)
    protocol={'task':'exploratory V02 force-linear known-hypothesis prediction; reused development split',
        'dataset':str(dataset),'dataset_sha256':sha(dataset),'arm':arm,'seed':seed,'steps':steps,
        'width':64,'layers':3,'batch':4,'query_times':8,'learning_rate':.001,'weight_decay':1e-5,
        'source_sha256':sources,'fit_pairs':12,'held_cross_action_pairs':4,
        'fit_target_dependent_loss_scale':'TRAIN-only RMS with .5mm floor; never model input',
        'position_loss':'balanced coordinate MSE','Jacobian_loss':'.05*mean(relu(.2-J)^2)',
        'selection':'fixed final checkpoint only; no held-pair score during fitting',
        'initial_state':'known rest; zero observed motion frames','known_force_history':'causal mean and centeredfirstmoment/intervalduration²',
        'device':str(device),'CPU_threads':2,'parameters':sum(p.numel() for p in model.parameters()),
        'reserved_response_access':False,'GitHub_Actions_enabled':False,'raw_outputs':'LOCAL_ONLY',
        'posterior_or_calibration_claim':False,'native_or_continuum_reference_claim':False,
        'development_reuse':'V01 held scores were inspected before this physical-inductive-bias redesign; not fresh confirmation',
        'model_class':'HybridGaussianResponse'}
    write(output/'protocol.json',protocol)
    (output/'source').mkdir()
    for rel in sources:(output/'source'/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    reference_edges=(data['rest'][fit][...,tets[:,1:],:]-data['rest'][fit][...,tets[:,:1],:]).transpose(-1,-2)
    inverse=torch.linalg.inv(reference_edges)
    started=time.monotonic();history=[]
    for step in range(1,steps+1):
        ids=torch.as_tensor(rng.integers(len(fit),size=4),device=device)
        tids=torch.as_tensor(np.sort(rng.choice(np.arange(1,61),8,replace=False)),device=device)
        batch=subset(fit_data,ids);features=tuple(v[ids] for v in fit_prepared)
        predicted=model(batch,time_indices=tids,prepared=features)['positions_m']
        loss_position=((predicted-fit_q[ids][:,tids])/fit_scales[ids,None,None,None]).square().mean()
        # Only a fitting-target-independent compatibility barrier, not a solver.
        sampled=torch.as_tensor(rng.choice(len(tets),128,replace=False),device=device)
        vertices=predicted[:,:,tets[sampled]]
        edges=(vertices[...,1:,:]-vertices[...,:1,:]).transpose(-1,-2)
        J=torch.linalg.det(edges@inverse[ids][:,None,sampled])
        loss_J=.05*torch.relu(.2-J).square().mean()
        loss=loss_position+loss_J
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite pilot training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%100==0 or step==steps:
            row={'step':step,'fit_loss':float(loss.detach()),'position_loss':float(loss_position.detach()),
                'Jacobian_loss':float(loss_J.detach()),'elapsed_s':time.monotonic()-started}
            history.append(row);write(output/'history.json',history);print(json.dumps(row),flush=True)
            if any(sha(ROOT/p)!=digest for p,digest in sources.items()):raise RuntimeError('Frozen training source changed')
    checkpoint={'model_state':model.state_dict(),'model_kwargs':{'width':64,'layers':3,'spatial_exchange':arm=='exchange'},
        'protocol':protocol,'source_sha256':sources,'step':steps}
    torch.save(checkpoint,output/'final.pt')
    model.eval();evaluate(model,data,arrays,prepared,output)
    write(output/'completed.json',{'completed':True,'steps':steps,'elapsed_s':time.monotonic()-started,
        'checkpoint_sha256':sha(output/'final.pt'),'predictions_sha256':sha(output/'predictions.npz'),
        'final_checkpoint_selected_without_held_scores':True,'calibrated_uncertainty_established':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2')
    p.add_argument('--arm',choices=['exchange','no_exchange'],required=True)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--seed',type=int,default=20261021)
    a=p.parse_args();run(a.dataset,a.output,a.device,a.arm,a.steps,a.seed)
