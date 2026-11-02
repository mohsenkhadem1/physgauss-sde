"""Local mechanics uncertainty bridge; no true candidate/held outcome API.

This is a conditional linear-Gaussian approximation in the15D parameter pack,
not calibrated posterior inference or a guarantee under bounds/model mismatch.
"""
import numpy as np
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,default_bounds,NAMES
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.local_uncertainty import covariance_from_jacobian

def split_prior_rows(gn,theta):
    values=np.asarray(theta,float);r,J,diagnostics=gn.residual_jacobian(values)
    if J.shape[1]!=15 or J.shape[0]<=15 or r.shape!=(J.shape[0],):raise ValueError('Full15parameter Jacobian with observations required')
    prior=np.diag(1/gn.estimator.std)
    if not np.allclose(J[-15:],prior,rtol=0,atol=1e-12)or not np.allclose(r[-15:],(values-gn.estimator.mean)/gn.estimator.std,rtol=0,atol=1e-12):raise ValueError('Expected exact final15 prior rows')
    return r[:-15],J[:-15],diagnostics

def local_covariance(gn,theta,*,fit_status=None,bounds=None,near_bound_fraction=.02):
    values=np.asarray(theta,float);limits=np.asarray(default_bounds()if bounds is None else bounds,float)
    if limits.shape!=(15,2)or not np.isfinite(limits).all()or (limits[:,1]<=limits[:,0]).any()or not 0<=near_bound_fraction<.5:raise ValueError('Valid explicit parameter bounds required')
    residual,J,diagnostics=split_prior_rows(gn,values)
    prior=np.diag(1/gn.estimator.std**2);result=covariance_from_jacobian(J,prior)
    distance=np.minimum(values-limits[:,0],limits[:,1]-values)/(limits[:,1]-limits[:,0])
    near=[NAMES[i]for i in np.flatnonzero(distance<=near_bound_fraction)]
    flags=['Local linearization only; model/observation discrepancy not estimated; no posterior calibration claim.','Conditional on supplied categorical families, geometry, known loads and frozen laws.']
    if fit_status is None:flags.append('MAP convergence status unavailable; this need not be a posterior mode.')
    elif fit_status.get('incomplete',True)or not fit_status.get('optimizer_success',False):flags.append('MAP is incomplete/unconverged; covariance is only a local sensitivity diagnostic.')
    if near:flags.append('Near parameter bounds: untruncated Gaussian covariance does not enforce these bounds.')
    result.update(observation_jacobian=J,observation_residual=residual,prior_precision=prior,observation_rows=len(J),removed_prior_rows=15,parameter_names=list(NAMES),near_bound_parameters=near,outside_bounds=bool((distance<0).any()),fit_status_supplied=fit_status is not None,diagnostics=diagnostics,limitations=flags,calibrated=False)
    return result

def public_position_jacobian(template,theta,commands):
    """Commands are{name,pose[2,6],points[P,3]} only; no observed outcomes.

The returned rows concatenate each command's material-point xyz displacement.
Reference positions are fixed, so displacement and absolute-position derivatives
are identical. Noise-whitenedJ uses the template's declared position sigma.
"""
    if not commands:raise ValueError('At least one public command required')
    packets=[];names=[];slices=[];start=0
    for command in commands:
        if set(command)!={'name','pose','points'}:raise ValueError('Only public name/pose/points allowed; no candidate truth channels')
        pose=np.asarray(command['pose'],float);points=np.asarray(command['points'],float)
        if pose.shape!=(2,6)or points.ndim!=2 or points.shape[1]!=3 or not len(points)or not np.isfinite(pose).all()or not np.isfinite(points).all()or (points<0).any()or(points>1).any():raise ValueError('Finite public command/material points required')
        names.append(command['name']);packets.append(dict(pose=pose,points=points,displacement=np.zeros_like(points)));slices.append((start,start+points.size));start+=points.size
    if len(set(names))!=len(names):raise ValueError('Unique public command names required')
    e=template;scene=e.scene
    estimator=JointMAP(GeometrySpec(**scene['geometry']),bulk_family=scene['bulk_family'],interface_family=int(scene['interface_latents'][0][2]),heterogeneity=scene['factors']['heterogeneity'],observations=packets,position_sigma=e.sigmas[0],prior_mean=e.mean,prior_std=e.std,n=e.kw['n'],order=e.kw['order'],bulk_law=e.kw['bulk_law'],interface_law=e.kw['interface_law'],device=e.device,gravity=scene['gravity'],density=scene['density_kg_m3'],continuation=e.continuation)
    # Only explicit public conditioning is copied; template observation values are never read.
    residual,J,diagnostics=split_prior_rows(GaussNewtonMAP(estimator),theta)
    sigma=float(e.sigmas[0])
    return dict(predicted_displacement_m=residual*sigma,position_jacobian_m=J*sigma,whitened_jacobian=J,position_sigma_m=sigma,command_slices=dict(zip(names,slices)),diagnostics=diagnostics,scope='Predictions at current estimatedtheta with frozen laws/publiccommands; zero dummy measurements only. No held/candidate true responses read.')
