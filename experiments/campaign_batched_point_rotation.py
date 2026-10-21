"""Batched candidate geometry retraction; no dynamics or model qualification.

An explicit candidate point node anchors a uniform all-node least-squares
infinitesimal rotation. Replace only that rotation by its exponential while
retaining the elastic residual. Never infer a pin from observations/truth.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch


def retract_positions(rest_points_m,positions_m,candidate_point_nodes):
    """float64 X[N,3], q[B,T,N,3], int64 pins[B], common device.

    Pins must already equal public rest coordinates exactly. No projection,
    regularization, rigid-handle enforcement or corrected force law occurs.
    """
    X,q,pins=rest_points_m,positions_m,candidate_point_nodes
    if (not isinstance(X,torch.Tensor) or not isinstance(q,torch.Tensor) or not isinstance(pins,torch.Tensor)
            or X.dtype!=torch.float64 or q.dtype!=torch.float64 or pins.dtype!=torch.int64
            or X.device!=q.device or pins.device!=q.device or X.ndim!=2 or X.shape[1]!=3 or len(X)<3
            or q.ndim!=4 or q.shape[2:]!=X.shape or pins.shape!=(len(q),) or not len(q) or not q.shape[1]
            or not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(q).all())
            or bool((pins<0).any()) or bool((pins>=len(X)).any())):
        raise ValueError('Explicit finite float64 rest[N,3], positions[B,T,N,3], int64 candidatepins[B] required')
    batch=torch.arange(len(q),device=q.device)
    if bool((q[batch,:,pins,:]!=X[pins,None,:]).any()):raise ValueError('Candidate pin must already be exact; no drift projection')
    r=X[None]-X[pins,None];u=q-X
    normal=torch.eye(3,dtype=q.dtype,device=q.device)*r.square().sum((-2,-1))[:,None,None]-r.transpose(-1,-2)@r
    condition=torch.linalg.cond(normal)
    if not bool(torch.isfinite(condition).all()) or bool((condition>1e12).any()):raise ValueError('Nondegenerate anchored normal matrix required; no regularizer')
    rhs=torch.linalg.cross(r[:,None],u,dim=-1).sum(-2)
    omega=torch.linalg.solve(normal[:,None],rhs.unsqueeze(-1)).squeeze(-1)
    wx,wy,wz=omega.unbind(-1);zero=torch.zeros_like(wx)
    k=torch.stack([zero,-wz,wy,wz,zero,-wx,-wy,wx,zero],-1).reshape(*omega.shape[:-1],3,3)
    theta2=omega.square().sum(-1);small=theta2<1e-4
    # The inactive large-angle branch also stays finite/differentiable at0.
    safe_theta=torch.sqrt(torch.where(small,torch.ones_like(theta2),theta2))
    a_small=-theta2/6+theta2.square()/120-theta2**3/5040+theta2**4/362880
    b_small=.5-theta2/24+theta2.square()/720-theta2**3/40320+theta2**4/3628800
    a=torch.where(small,a_small,torch.sin(safe_theta)/safe_theta-1)
    b=torch.where(small,b_small,.5*(torch.sin(safe_theta/2)/(safe_theta/2)).square())
    matrix=a[...,None,None]*k+b[...,None,None]*(k@k)
    correction=torch.einsum('btij,bnj->btni',matrix,r)
    linear_rotation=torch.linalg.cross(omega[:,:,None],r[:,None],dim=-1)
    return dict(positions_m=q+correction,correction_m=correction,
        fitted_infinitesimal_rotation_rad=omega,fitted_rotation_matrix=torch.eye(3,dtype=q.dtype,device=q.device)+k+matrix,
        linear_residual_coordinate_RMS_m=torch.sqrt((u-linear_rotation).square().mean((-2,-1))),
        rotation_fit_normal_condition_number=condition,
        role='Candidate geometry-only anchored firstorder-to-exp correction; unchanged elastic residual, no improved dynamics/exactrigidhandles/calibration claim')


def synthetic():
    X=np.array([[0.,0.,0.],[.2,0,0],[0,.3,0],[0,0,.1],[-.2,0,0],[0,-.3,0],[0,0,-.1],[.1,.2,.1]],dtype=np.float64)
    pins=np.array([0,2,5],dtype=np.int64);rng=np.random.default_rng(2026116001)
    omega=rng.normal(size=(3,7,3))*.1;omega[:,0]=0;omega[:,1]*=1e-11
    r=X[None]-X[pins,None];u=np.cross(omega[:,:,None],r[:,None])+.002*r[:,None]
    return X,X+u,pins


def benchmark(output,device='cuda:2'):
    """Only fixed synthetic controls, never current calibration/evaluation data."""
    try:from campaign_point_rotation_retraction import retract_point_rotation
    except ImportError:from experiments.campaign_point_rotation_retraction import retract_point_rotation
    torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol=dict(status='FROZEN_BEFORE_SYNTHETIC_ADAPTER_CALLS',source_sha256={p:sha(root/p) for p in
        ['experiments/campaign_batched_point_rotation.py','experiments/campaign_point_rotation_retraction.py']},
        scope='Synthetic3candidates7frames8nodes only; no currentreserved32 use, observations, fittedparameters, physics or posterior',
        CPU_threads=2,GPU_device=device,random_seed=2026116001)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');X,q,pins=synthetic();started=time.monotonic()
    cpu=retract_positions(torch.tensor(X),torch.tensor(q),torch.tensor(pins))
    references=[retract_point_rotation(X,q[b]-X,int(pin)) for b,pin in enumerate(pins)]
    errors={}
    for key in ['correction_m','fitted_infinitesimal_rotation_rad','fitted_rotation_matrix']:
        errors[key]=float(np.max(abs(cpu[key].numpy()-np.stack([r[key] for r in references]))))
    gpu=retract_positions(torch.tensor(X,device=device),torch.tensor(q,device=device),torch.tensor(pins,device=device))
    parity={key:float((value.cpu()-cpu[key]).abs().max()) for key,value in gpu.items() if isinstance(value,torch.Tensor)}
    if max(errors.values())>3e-14 or max(parity.values())>3e-13:raise ValueError('Synthetic parity failed; retain protocol/output')
    report=dict(status='COMPLETED_SYNTHETIC_CANDIDATE_ROTATION_ADAPTER_PARITY',CPU_NumPy_max_absolute_errors=errors,
        CPU_GPU2_max_absolute_errors=parity,elapsed_s=time.monotonic()-started,source_sha256=protocol['source_sha256'],
        role=cpu['role'],current_reserved_data_or_truth_used=False,new_physics_or_inference=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(benchmark(args.output),indent=2))
