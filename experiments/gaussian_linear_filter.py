"""Analytical load conditioning for a neural Green response and RGB-D ranges.

This separate hybrid baseline uses an UNCONDITIONAL learned physical prior.
Consequently conditioning on the initial video does not double-count that
video. Nonlinear material/support hypotheses are particles; load amplitudes
are a category-conditional Gaussian. Reference-normal range linearization and
the Gaussian prior are approximations. No mechanics solve or parameter search
is performed; only a small Gaussian posterior update is required per particle.
"""
import numpy as np
import torch
try:
    from gaussian_acquisition_sensor import public_ray_queries
except ImportError:
    from experiments.gaussian_acquisition_sensor import public_ray_queries


def gaussian_linear_update(observation,offset,design,prior_mean,prior_covariance,noise_covariance):
    """Exact posterior and marginal log likelihood for a LINEAR Gaussian model."""
    y=np.asarray(observation,dtype=float);b=np.asarray(offset,dtype=float)
    h=np.asarray(design,dtype=float);m=np.asarray(prior_mean,dtype=float)
    c=np.asarray(prior_covariance,dtype=float);r=np.asarray(noise_covariance,dtype=float)
    if h.shape!=(len(y),len(m)) or b.shape!=y.shape or c.shape!=(len(m),len(m)) or r.shape!=(len(y),len(y)):
        raise ValueError('Incompatible linear Gaussian dimensions')
    if not all(np.isfinite(v).all() for v in [y,b,h,m,c,r]):raise ValueError('Nonfinite Gaussian model')
    if not np.allclose(c,c.T) or not np.allclose(r,r.T):raise ValueError('Symmetric covariances required')
    lc=np.linalg.cholesky(c);lr=np.linalg.cholesky(r)
    # Work in whitened prior coordinates so the solve dimension is the number
    # of loads (3or6), rather than the number of observed pixels.
    a=np.linalg.solve(lr,h@lc);residual=np.linalg.solve(lr,y-b-h@m)
    precision=np.eye(len(m))+a.T@a;factor=np.linalg.cholesky(precision)
    inverse=np.linalg.solve(factor.T,np.linalg.solve(factor,np.eye(len(m))))
    mean=m+lc@(inverse@(a.T@residual))
    covariance=lc@inverse@lc.T
    quadratic=residual@residual-(a.T@residual)@inverse@(a.T@residual)
    logdet=2*np.log(np.diag(lr)).sum()+2*np.log(np.diag(factor)).sum()
    log_evidence=-.5*(len(y)*np.log(2*np.pi)+logdet+quadratic)
    return mean,covariance,float(log_evidence)


def range_observations(inputs,rays_per_frame=12,seed=225):
    """Static reference-ray normals and observed ranges, no deformed mesh read."""
    if inputs['observation_mode']!='rgbd':raise ValueError('This baseline consumes observed depth')
    ref=inputs['reference'];points=np.asarray(ref['rest_points_m'])
    lower,upper=points.min(0),points.max(0);c=inputs['cameras'];rows=[]
    for frame,pose in enumerate(c['camera_to_world_usd_row']):
        query=public_ray_queries(ref,pose,inputs['native_K'],rays_per_frame,seed+frame)
        xs,ys=query['xs'],query['ys'];k=query['K']
        rays=np.stack(((xs+.5-k[0,2])/k[0,0],-(ys+.5-k[1,2])/k[1,1],-np.ones(len(xs))),-1)@pose[:3,:3]
        rest=pose[3,:3]+query['rest_depth_m'][:,None]*rays
        distances=np.concatenate((abs(rest-lower),abs(rest-upper)),-1)
        faces=distances.argmin(-1);normal=np.eye(3)[faces%3]*np.where(faces<3,-1.,1.)[:,None]
        denominator=(normal*rays).sum(-1)
        measured=inputs['depth_native_m'][frame,ys,xs]
        valid=np.isfinite(measured)&(measured>0)&(abs(denominator)>.05)
        rows.append({'points_m':rest[valid],'range_jacobian':normal[valid]/denominator[valid,None],
                     'value_m':measured[valid]-query['rest_depth_m'][valid],
                     'time_s':float(c['times_s'][frame]),'frame_index':frame,
                     'missing_queries':int((~valid).sum())})
    return rows


def gaussian_control_prior(boundary_type,force_std_N=.12,motion_std_m=.004):
    dimension=6 if boundary_type==3 else 3
    std=np.r_[np.full(3,force_std_N),np.full(3,motion_std_m)][:dimension]
    return np.zeros(dimension),np.diag(std**2)


def linear_design(operator,reference,physical,boundary_type,force_type,points,times):
    """Return offset and force/motion response. Physical inputs remain explicit."""
    if len(physical)!=1:raise ValueError('One nonlinear physical hypothesis at a time')
    p=physical.clone();p[:,16:]=0
    offset,basis=operator.load_response_basis(points,times,p,boundary_type,force_type)
    # Canonical categories use one resultant load or an equal/opposite couple.
    design=basis[...,:3]-basis[...,3:] if int(force_type[0])==3 else basis[...,:3]
    if int(boundary_type[0])==3:
        motion=[]
        for axis in range(3):
            displaced=p.clone();displaced[:,22+axis]=.004
            motion.append((operator(points,times,displaced,boundary_type,force_type)-offset)/.004)
        design=torch.cat([design,torch.stack(motion,-1)],-1)
    return offset,design
