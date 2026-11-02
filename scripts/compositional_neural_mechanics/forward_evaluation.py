"""Common oracle/inferred-descriptor forward API, without reference-label access.

Physical descriptors must be supplied identically to competing forward models.
The matched analytic model is an oracle ceiling when it generated the data.
Neo-Hookean is a misspecified small-strain-matched baseline, not that ceiling.
"""
import numpy as np
import torch
from torch import nn
from scripts.compositional_neural_mechanics.materials import invariants
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.feature_pack import build_features
from scripts.compositional_neural_mechanics.qualification import nonrigid


class SmallStrainMatchedNeoHookean(nn.Module):
    def energy_density(self,F,z):
        q1,_,volume=invariants(F)
        return torch.exp(z[...,0])*(2000*q1+20000*volume)


def solve_forward(scene,pose,queries,*,kind,n=4,order=3,device='cpu',
                  bulk_law=None,interface_law=None,continuation=(.25,.5,.75,1.)):
    if kind not in ('matched_analytic','neo_hookean','shared_laws'):raise ValueError('Explicit physical model required')
    if kind=='shared_laws' and (bulk_law is None or interface_law is None):raise ValueError('Both frozen laws required')
    if list(continuation)!=[.25,.5,.75,1.]:raise ValueError('Declared independent-action continuation required')
    bulk=bulk_law if kind=='shared_laws' else SmallStrainMatchedNeoHookean() if kind=='neo_hookean' else None
    interface=interface_law if kind=='shared_laws' else None
    model=make_model(scene,n=n,order=order,device=device,bulk_law=bulk,interface_law=interface)
    state=None
    for fraction in continuation:
        model.set_controls(np.asarray(pose)*fraction)
        state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
    w=model.reaction_wrenches(state)
    return dict(displacement=model.sample_material(state['displacement_local_m'],queries),
                wrenches=np.concatenate([w['forces_N'],w['torques_Nm']],axis=-1),
                diagnostics=dict(minimum_J=float(np.min(state['J'])),residual_N=float(np.linalg.norm(state['residual'])),
                                 force_balance_N=float(w['force_balance_N']),n=n,kind=kind))


def direct_forward(scene,pose,queries,*,predictor,nodes,bulk_law,interface_law,probes,checkpoint_sha256):
    features=build_features(scene,pose,nodes,queries,bulk_law,interface_law,probes=probes,checkpoint_sha256=checkpoint_sha256)
    param=next(predictor.parameters())
    tensors=[torch.as_tensor(features[k],device=param.device,dtype=param.dtype)[None] for k in ('nodefeatures','globalfeatures','queryfeatures')]
    with torch.no_grad():u,w=predictor(*tensors)
    return dict(displacement=u[0].cpu().numpy(),wrenches=w[0].cpu().numpy(),
                diagnostics=dict(kind='custom_direct',equilibrium_constraint=False))


def score_prediction(prediction,reference,rest_queries):
    p=np.asarray(prediction['displacement']);r=np.asarray(reference['displacement'])
    pw=np.asarray(prediction['wrenches']);rw=np.asarray(reference['wrenches']);rest=np.asarray(rest_queries)
    if p.shape!=r.shape or p.shape!=rest.shape or p.ndim!=2 or p.shape[1]!=3 or pw.shape!=(2,6) or rw.shape!=(2,6):raise ValueError('Matching physical query and wrench shapes required')
    if not all(np.isfinite(x).all() for x in (p,r,pw,rw,rest)):raise ValueError('Nonfinite predictions must be recorded as failures, not silently scored')
    difference=p-r;error=np.linalg.norm(difference,axis=-1)
    return dict(displacement_coordinate_rmse_m=float(np.sqrt(np.mean(difference**2))),
                displacement_point_error_mean_m=float(error.mean()),displacement_point_error_max_m=float(error.max()),
                canonical_nonrigid_coordinate_rmse_m=float(np.sqrt(np.mean((nonrigid(rest,p)-nonrigid(rest,r))**2))),
                force_coordinate_rmse_N=float(np.sqrt(np.mean((pw[:,:3]-rw[:,:3])**2))),
                torque_coordinate_rmse_Nm=float(np.sqrt(np.mean((pw[:,3:]-rw[:,3:])**2))),
                reference_nonrigid_coordinate_rms_m=float(np.sqrt(np.mean(nonrigid(rest,r)**2))))
