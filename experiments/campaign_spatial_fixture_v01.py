"""Prospective six-parameter spatial Gaussian fixture, manufactured15DOF model.

No default truth, experiment execution, posterior sampling, or result-based
object selection. This changes the parameter domain; earlier results stay fixed.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import logsumexp
from experiments import campaign_compliant_patch_inverse as old
from experiments import campaign_compliant_force_challenge_v01 as loads
from experiments.campaign_boundary_gaussians import SurfaceQuadrature,GaussianPatch
from experiments.campaign_compliant_gaussian_linear import GaussianFixtureMatrix
from experiments.campaign_linear_fem import assemble_tetrahedra

PARAMETERS=('E_Pa','center_x_material_m','center_y_material_m','center_z_material_m','width_m','fixture_stiffness_N_m')
LOWER=np.array([np.log(2000.),.005,.005,.005,np.log(.018),np.log(5.)])
UPPER=np.array([np.log(8000.),.095,.095,.095,np.log(.05),np.log(30.)])
TARGET_FRAMES=(30,45,60)
FUNCTION_NAMES=PARAMETERS+tuple('fixture_K_node'+str(i) for i in range(5))+('fixture_centroid_x','fixture_centroid_y','fixture_centroid_z')+tuple('A_final_node1_'+a for a in 'xyz')+tuple('B_final_node4_'+a for a in 'xyz')


def design():
    return dict(schema='spatial-fixture-v01',parameters=list(PARAMETERS),intrinsic=['logE','cx','cy','cz','logwidth','logK'],
        prior_lower=LOWER.tolist(),prior_upper=UPPER.tolist(),prior='Independent uniform intrinsic coordinates; center in reference/material frame',
        geometry='Unchanged5nodes/2linear tetrahedra;15freeDOFs; no hard pins',fixture='One normalized surface Gaussian, finite isotropicK; damping.05 times nodalweight',
        density=625.,poisson=.35,h=old.H,steps=60,gravity=False,contact=False,
        prefix=dict(force='A',steps=24,frames=[12,24],nodes=[1,4],channels=12,sigma_m=old.SIGMA),
        features=dict(target_frames=list(TARGET_FRAMES),target_components=135,poke_components=72,functions=list(FUNCTION_NAMES),shared_RHS=11),
        scope='Controlled small-strain finite reference; no continuum, camera, physicalcalibration or probing advantage claim',
        no_default_truth=True,no_automatic_inference=True,
        structural_nonidentifiability='Fiveverticescospherical around(.05,.05,.05): weights depend on(center-mid)/width² only; centerandwidth not separately identifiable, even with noiseless fullmotion',
        parameter_recovery_claim=False)


def physical(unit):
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or unit.shape[1]!=6 or not len(unit) or not np.isfinite(unit).all() or np.any((unit<0)|(unit>1)):
        raise ValueError('Finite unit[B,6] in prospective prior')
    theta=LOWER+unit*(UPPER-LOWER);theta=theta.copy();theta[:,[0,4,5]]=np.exp(theta[:,[0,4,5]])
    return theta


def validate_theta(theta):
    x=np.asarray(theta,float)
    if x.ndim!=2 or x.shape[1]!=6 or not len(x) or not np.isfinite(x).all() or np.any(x[:,[0,4,5]]<=0):
        raise ValueError('Finite physical theta[B,6] with positiveE,width,K')
    return x


class SpatialForward:
    def __init__(self):
        self.X,tets,faces=old.tiny_geometry()
        self.Kunit,self.mass,_,_=assemble_tetrahedra(self.X,tets,1.,.35,625.)
        self.areas=SurfaceQuadrature.from_mesh(self.X,faces).node_area_m2

    def weights(self,theta):
        theta=validate_theta(theta)
        lw=np.log(self.areas)[None]-.5*np.sum((self.X[None]-theta[:,None,1:4])**2,axis=2)/theta[:,4,None]**2
        return np.exp(lw-logsumexp(lw,axis=1)[:,None])

    def matrices(self,theta):
        theta=validate_theta(theta);w=self.weights(theta);diagonal=torch.tensor(np.repeat(w,3,axis=1),dtype=torch.float64)
        K=torch.tensor(theta[:,0,None,None])*self.Kunit+torch.diag_embed(diagonal*torch.tensor(theta[:,5,None]));C=.05*diagonal
        return K,C

    def solve(self,theta,forces,*,steps=60):
        """Exact unchanged backward Euler, all RHS sharing their own Cholesky."""
        theta=validate_theta(theta);forces=np.asarray(forces,float)
        if forces.ndim!=4 or forces.shape[1:]!=(60,5,3) or not len(forces) or not np.isfinite(forces).all() or steps not in(24,60):
            raise ValueError('Known forces[R,60,5,3], steps24or60')
        K,C=self.matrices(theta);h=old.H;m=self.mass;n=len(theta);r=len(forces)
        chol=torch.linalg.cholesky(K+torch.diag_embed(m[None]/h**2+C/h))
        held=torch.tensor(forces.reshape(r,60,15).transpose(1,2,0));u=torch.zeros((n,15,r),dtype=torch.float64);v=u.clone();states=[u.numpy().copy()]
        for step in range(steps):
            rhs=held[step][None]+m[None,:,None]*(u/h**2+v/h)+C[:,:,None]*u/h
            new=torch.cholesky_solve(rhs,chol);v=(new-u)/h;u=new;states.append(u.numpy().copy())
        # [draw,RHS,clock,node,xyz], no solver/time approximation.
        return np.stack(states,axis=1).transpose(0,3,1,2).reshape(n,r,steps+1,5,3)

    def __call__(self,theta,program='A'):
        force=old.known_force(program) if isinstance(program,str) else program
        return self.solve(theta,np.asarray(force)[None])[:,0]

    def prefix(self,theta):
        u=self.solve(theta,old.known_force('A')[None],steps=24)[:,0]
        return old.selected_observations(u)

    def log_likelihood(self,unit,observation,*,sigma=old.SIGMA):
        observation=np.asarray(observation,float)
        if observation.shape!=(12,) or not np.isfinite(observation).all() or not np.isfinite(sigma) or sigma<=0:
            raise ValueError('Explicit finite12channel observation and positive sigma')
        predicted=self.prefix(physical(unit))
        return -.5*np.sum(((predicted-observation)/sigma)**2,axis=1)-12*np.log(sigma*np.sqrt(2*np.pi))


def fixture_functions(forward,theta):
    theta=validate_theta(theta);w=forward.weights(theta)
    return np.concatenate([theta,w*theta[:,5,None],w@forward.X],axis=1)


def joint_features(forward,theta,*,chunk=512,with_functions=True):
    theta=validate_theta(theta)
    if type(chunk)is not int or chunk<1:raise ValueError('Positive chunk')
    programs=list(loads.force_programs().values())+list(loads.poke_programs().values())+[old.known_force('A'),old.known_force('B')]
    targets=[];pokes=[];functions=[]
    for first in range(0,len(theta),chunk):
        th=theta[first:first+chunk];u=forward.solve(th,np.stack(programs));n=len(th)
        targets.append(u[:,:3][:,:,TARGET_FRAMES].reshape(n,135))
        pokes.append(u[:,3:9][:,:,[12,24]][:,:,:,old.OBS_NODES].reshape(n,6,12))
        if with_functions:functions.append(np.concatenate([fixture_functions(forward,th),u[:,9,-1,1],u[:,10,-1,4]],axis=1))
    output=(np.concatenate(targets),np.concatenate(pokes).transpose(1,0,2))
    return (*output,np.concatenate(functions)) if with_functions else output


def independent_matrices(theta):
    """Unvectorized GaussianFixtureMatrix assembly, not vectorweight reuse."""
    theta=validate_theta(np.asarray(theta)[None])[0];X,tets,faces=old.tiny_geometry()
    ku,mass,_,_=assemble_tetrahedra(X,tets,1.,.35,625.)
    quad=SurfaceQuadrature.from_mesh(X,faces)
    patches=[GaussianPatch(theta[1:4],np.eye(3)*theta[4]**2,'candidate_boundary_hypothesis'),GaussianPatch(X[0],np.eye(3)*.03**2,'candidate_boundary_hypothesis',active=False)]
    fixture=GaussianFixtureMatrix.from_patches(quad,patches,[np.eye(3)*theta[5],np.zeros((3,3))],[np.eye(3)*.05,np.zeros((3,3))])
    return theta[0]*ku.numpy()+fixture.K.numpy(),fixture.C.numpy(),mass.numpy(),fixture.weights[0].numpy()


def independent_numpy_forward(theta,force,*,steps=60):
    force=np.asarray(force,float)
    if force.shape!=(60,5,3) or not np.isfinite(force).all() or steps not in(24,60):raise ValueError('Finite force and24or60steps')
    K,C,mass,_=independent_matrices(theta);M=np.diag(mass);h=old.H
    L=np.block([[np.eye(15),-h*np.eye(15)],[h*K,M+h*C]])
    u=np.zeros(15);v=u.copy();states=[u.copy()]
    for F in force[:steps].reshape(steps,15):
        uv=np.linalg.solve(L,np.concatenate([u,M@v+h*F]));u=uv[:15];v=uv[15:];states.append(u.copy())
    return np.asarray(states).reshape(steps+1,5,3)


def observation_jacobian(forward,unit,*,step=1e-4,program='A'):
    unit=np.asarray(unit,float)
    if unit.shape!=(6,) or not np.isfinite(step) or step<=0 or np.any(unit-step<=0) or np.any(unit+step>=1):raise ValueError('Interior6D central derivative')
    points=np.concatenate([unit[None]+step*np.eye(6),unit[None]-step*np.eye(6)])
    theta=physical(points)
    if isinstance(program,str) and program=='A':predicted=forward.prefix(theta)
    else:predicted=old.selected_observations(forward(theta,program))
    return ((predicted[:6]-predicted[6:])/(2*step)).T


def local_fisher_scores(forward,unit_mean,cov,*,step=1e-4,sigma=old.SIGMA):
    cov=np.asarray(cov,float)
    if cov.shape!=(6,6) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-13,rtol=0) or not np.isfinite(sigma) or sigma<=0:raise ValueError('Symmetric6D covariance and positive sigma')
    np.linalg.cholesky(cov) # Fail closed; no jitter/removed directions.
    inverse=np.linalg.inv(cov);scores=[]
    for force in loads.poke_programs().values():
        J=observation_jacobian(forward,unit_mean,step=step,program=force)
        scores.append(float(np.trace(cov)-np.trace(np.linalg.inv(inverse+J.T@J/sigma**2))))
    return np.asarray(scores)
