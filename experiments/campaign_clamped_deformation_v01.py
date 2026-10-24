"""Prospective clamped block finite-model reference; no automatic experiments.

Known reference-space Gaussian support location/width; unknown materialE and
supportK. Exact hard-clamp elimination, float64 backwardEuler, no gravity or
contact. Numerical qualification does not establish continuum convergence.
"""
from __future__ import annotations
import hashlib,itertools
from pathlib import Path
import numpy as np
import torch
from experiments.campaign_linear_fem import assemble_tetrahedra
from experiments.campaign_boundary_gaussians import SurfaceQuadrature

LOWER=np.log([2000.,5.]);UPPER=np.log([8000.,30.]);SIGMA=20e-6
DURATION=2.;PREFIX_END=.8;OBS_TIMES=(.2,.4,.6,.8)
DEFORMATION_GATES=dict(minimum_nonrigid_RMS_m=100e-6,minimum_nonrigid_fraction=.3,maximum_linear_strain=.05,minimum_J=.8)
TIME_GATES=dict(h_values=[1/120,1/240,1/480],comparison_output_hz=20,maximum_last_pair_RMS_m=10e-6,maximum_pair_contraction=.5,p_floor=1.,conditional_position_indicator='2timeslastpair<=20um assumingcontinuingp>=1; no velocityqualificationclaim')


def geometry():
    X=np.array(list(itertools.product(np.linspace(0,.24,5),[0.,.08],[0.,.08])))
    node=lambda i,j,k:i*4+j*2+k
    tets=[]
    for i in range(4):
        for permutation in itertools.permutations(range(3)):
            corner=np.array([i,0,0]);vertices=[corner.copy()]
            for axis in permutation:corner=corner.copy();corner[axis]+=1;vertices.append(corner)
            ids=np.array([node(*v) for v in vertices])
            if np.linalg.det((X[ids[1:]]-X[ids[0]]).T)<0:ids[[1,2]]=ids[[2,1]]
            tets.append(ids)
    tets=np.array(tets);counts={}
    for tet in tets:
        for face in itertools.combinations(tet,3):
            key=tuple(sorted(face));counts[key]=counts.get(key,0)+1
    faces=np.array([f for f,n in counts.items() if n==1]);fixed=np.flatnonzero(X[:,0]==0)
    free=np.flatnonzero(np.repeat(X[:,0]!=0,3));fixed.setflags(write=False);free.setflags(write=False)
    return X,tets,faces,fixed,free


def design():
    return dict(schema='clamped-deformation-v01',intrinsic=['logE','logsupportK'],prior_lower=LOWER.tolist(),prior_upper=UPPER.tolist(),
        geometry='20nodes24positivetets,.24x.08x.08m; allx=0DOFs exactlyeliminated',known_support_center_m=[.16,.08,.04],known_support_width_m=.025,
        support='Area-normalizedGaussian, isotropicfiniteK, known.05Ns/m multipliedbynodalweights',density=625.,poisson=.35,
        qualification_forces=dict(bending_tip_Fz_N=.03,stretching_tip_Fx_N=.24,ramp_s=.3,hold_until_s=2.,distribution='Equal1/4atfourtipnodes; netstatedforce'),
        analytical_pre_result_design='Beforeanyresponse: E8kEI=.0273067Nm²,.03Nbendtip5.0625mm with approximate properrigidremovedcoordinateRMS.07delta/sqrt3=205um; .24Naxialtip1.125mm gives propervolumecoordinateRMSdelta/6=187.5um. E2kaxialstrain.01875 andbendingrootstrain.0422. Choose100um(5sigma),fraction.3 beforecalls; fixture/lockingmayreducebelowgate, retainfailures.',
        gravity=False,contact=False,duration_s=DURATION,prefix_end_s=PREFIX_END,observation_times_s=list(OBS_TIMES),observation_nodes=[11,19],observation_channels=24,sigma_m=SIGMA,
        deformation_gates=DEFORMATION_GATES,time_gates=TIME_GATES,
        causality='Allfutureprogramsbitidenticalprefixforce through.8s; matchingprefixstates required. No heldoutobservations inlikelihood.',
        scope='Linearcoarsefinite-modeldevelopment only; spatial/continuum convergence not established',no_automatic_inference=True)


def source_provenance():
    names=['experiments/campaign_clamped_deformation_v01.py','experiments/campaign_linear_fem.py','experiments/campaign_boundary_gaussians.py']
    root=Path(__file__).resolve().parents[1]
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def physical(unit):
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or unit.shape[1]!=2 or not len(unit) or not np.isfinite(unit).all() or np.any((unit<0)|(unit>1)):raise ValueError('Finiteunit[B,2] insideprior')
    return np.exp(LOWER+unit*(UPPER-LOWER))


def _theta(theta):
    theta=np.asarray(theta,float)
    if theta.ndim!=2 or theta.shape[1]!=2 or not len(theta) or not np.isfinite(theta).all() or np.any(theta<=0):raise ValueError('Physical[E,K]positive[B,2]')
    return theta


def time_grid(h):
    if not np.isfinite(h) or h<=0 or abs(round(DURATION/h)*h-DURATION)>1e-12 or abs(round(PREFIX_END/h)*h-PREFIX_END)>1e-12:raise ValueError('Positivehdividing2sand.8s')
    return np.arange(1,round(DURATION/h)+1)*h


def _ramp(t):return np.sin(np.pi*np.minimum(t,.3)/.6)**2


def programs(h):
    """Endpoint-held controls; analytical amplitudes frozen before responses."""
    X,*_=geometry();t=time_grid(h);tip=np.flatnonzero(X[:,0]==.24);n=len(X)
    def force(direction,amplitude,nodes=tip):
        f=np.zeros((len(t),n,3));f[:,nodes]=_ramp(t)[:,None,None]*amplitude*np.asarray(direction)[None,None]/len(nodes);return f
    bending=force([0,0,1],.03);stretching=force([1,0,0],.24)
    prefix=force([1,0,0],.06)+bending
    future={}
    for name,continuation in [('axial',stretching),('opposite_bend',-bending),('transverse',force([0,1,0],.02))]:
        f=prefix.copy();f[t>PREFIX_END]=continuation[t>PREFIX_END];future[name]=f
    mid=np.flatnonzero(X[:,0]==.12)
    pokes={f'{location}_{axis}':force(np.eye(3)[axis],.24 if axis==0 else .03,nodes) for location,nodes in [('mid',mid),('tip',tip)] for axis in range(3)}
    return dict(qualification={'bending':bending,'stretching':stretching},prefix=prefix,future=future,pokes=pokes)


class ClampedForward:
    def __init__(self):
        self.X,self.tets,self.faces,self.fixed,self.free=geometry()
        ku,m,self.gradients,self.volumes=assemble_tetrahedra(self.X,self.tets,1.,.35,625.)
        ids=torch.tensor(self.free.copy(),dtype=torch.long)
        self.Kunit=ku[ids][:,ids];self.mass=m[ids]
        area=SurfaceQuadrature.from_mesh(self.X,self.faces).node_area_m2
        raw=area*np.exp(-.5*np.sum((self.X-[.16,.08,.04])**2,axis=1)/.025**2)
        self.weights=raw/raw.sum();self.diagonal=torch.tensor(np.repeat(self.weights,3)[self.free])
    def trajectory(self,theta,forces,h,*,stop_time=DURATION):
        theta=_theta(theta);t=time_grid(h);f=np.asarray(forces,float)
        if f.ndim==3:f=f[None]
        if f.ndim!=4 or f.shape[1:]!=(len(t),len(self.X),3) or not len(f) or not np.isfinite(f).all():raise ValueError('Knownforce[R,steps,N,3]')
        if not np.isfinite(stop_time) or stop_time<=0 or stop_time>DURATION or abs(round(stop_time/h)*h-stop_time)>1e-12:raise ValueError('Validexactstopclock')
        count=round(stop_time/h);f=f[:,:count];t=t[:count]
        n=len(theta);r=len(f);K=torch.tensor(theta[:,0,None,None])*self.Kunit+torch.diag_embed(torch.tensor(theta[:,1,None])*self.diagonal)
        C=.05*self.diagonal;m=self.mass
        chol=torch.linalg.cholesky(K+torch.diag(m/h**2+C/h)[None]);u=torch.zeros((n,len(m),r),dtype=torch.float64);v=u.clone()
        force=torch.tensor(f.reshape(r,len(t),-1)[:,:,self.free].transpose(1,2,0));states=[u.numpy().copy()]
        for F in force:
            rhs=F[None]+m[None,:,None]*(u/h**2+v/h)+C[None,:,None]*u/h
            new=torch.cholesky_solve(rhs,chol);v=(new-u)/h;u=new;states.append(u.numpy().copy())
        reduced=np.stack(states,axis=2).transpose(0,3,2,1);full=np.zeros((n,r,len(t)+1,len(self.X)*3));full[:,:,:,self.free]=reduced
        return full.reshape(n,r,len(t)+1,len(self.X),3)
    def prefix(self,theta,*,h=1/120):
        # Integrate onlyobservedprefix; never solve futureclocks forlikelihood.
        force=programs(h)['prefix']
        u=self.trajectory(theta,force,h,stop_time=PREFIX_END)[:,0]
        ids=np.rint(np.asarray(OBS_TIMES)/h).astype(int)
        return u[:,ids][:,:,[11,19]].reshape(len(u),24)
    def log_likelihood(self,unit,observation,*,h=1/120,sigma=SIGMA):
        y=np.asarray(observation,float)
        if y.shape!=(24,) or not np.isfinite(y).all() or not np.isfinite(sigma) or sigma<=0:raise ValueError('Explicit24channelobservation')
        predicted=self.prefix(physical(unit),h=h)
        return -.5*np.sum(((predicted-y)/sigma)**2,axis=1)-24*np.log(sigma*np.sqrt(2*np.pi))
    def features(self,theta,*,h=1/120):
        p=programs(h);future=self.trajectory(theta,np.stack(list(p['future'].values())),h);poke=self.trajectory(theta,np.stack(list(p['pokes'].values())),h)
        frames=np.rint(np.array([1.2,1.6,2.])/h).astype(int);obs=np.rint(np.asarray(OBS_TIMES)/h).astype(int)
        return dict(future=future[:,:,frames].reshape(len(future),-1),pokes=poke[:,:,obs][:,:,:,[11,19]].reshape(len(poke),6,24).transpose(1,0,2))


def independent_numpy_assembly():
    """Separate loop-based NumPystrain assembly; no TorchK reuse."""
    X,tets,faces,fixed,free=geometry();K=np.zeros((3*len(X),)*2);m=np.zeros(len(X));nu=.35
    mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));D=np.zeros((6,6));D[:3,:3]=lam;D[:3,:3]+=np.eye(3)*2*mu;D[3:,3:]=np.eye(3)*mu
    for ids in tets:
        dm=(X[ids[1:]]-X[ids[0]]).T;volume=np.linalg.det(dm)/6;g=np.linalg.inv(dm);g=np.concatenate([-g.sum(0)[None],g]);B=np.zeros((6,12))
        for j,(dx,dy,dz) in enumerate(g):
            a=3*j;B[0,a]=dx;B[1,a+1]=dy;B[2,a+2]=dz;B[3,a+1]=dz;B[3,a+2]=dy;B[4,a]=dz;B[4,a+2]=dx;B[5,a]=dy;B[5,a+1]=dx
        dofs=(ids[:,None]*3+np.arange(3)).reshape(-1);K[np.ix_(dofs,dofs)]+=B.T@D@B*volume;m[ids]+=625*volume/4
    area=np.zeros(len(X))
    for face in faces:area[face]+=np.linalg.norm(np.cross(X[face[1]]-X[face[0]],X[face[2]]-X[face[0]]))/6
    w=area*np.exp(-.5*np.sum((X-[.16,.08,.04])**2,axis=1)/.025**2);w/=w.sum()
    return K[np.ix_(free,free)],np.repeat(m,3)[free],np.repeat(w,3)[free]


def independent_numpy_trajectory(theta,force,h):
    theta=_theta(np.asarray(theta)[None])[0];ku,m,w=independent_numpy_assembly();X,_,_,_,free=geometry();f=np.asarray(force,float)
    if f.shape!=(len(time_grid(h)),len(X),3) or not np.isfinite(f).all():raise ValueError('Singlefiniteforce')
    K=theta[0]*ku+np.diag(theta[1]*w);C=np.diag(.05*w);M=np.diag(m);d=len(m)
    L=np.block([[np.eye(d),-h*np.eye(d)],[h*K,M+h*C]]);u=np.zeros(d);v=u.copy();saved=[u.copy()]
    for F in f.reshape(len(f),-1)[:,free]:
        state=np.linalg.solve(L,np.concatenate([u,M@v+h*F]));u,v=state[:d],state[d:];saved.append(u.copy())
    out=np.zeros((len(f)+1,len(X)*3));out[:,free]=saved;return out.reshape(len(f)+1,len(X),3)


def displacement_gradient(forward,u):
    u=np.asarray(u,float)
    if u.shape[-2:]!=forward.X.shape or not np.isfinite(u).all():raise ValueError('Finite displacement[...,N,3]')
    return np.einsum('btnc,tni->btci',u.reshape(-1,len(forward.X),3)[:,forward.tets],forward.gradients.numpy())


def deformation_metrics(forward,u):
    """Uniformnodal infinitesimalfit diagnostic ONLY; notvolumeproperfitgate."""
    u=np.asarray(u,float)
    if u.shape[-2:]!=forward.X.shape or not np.isfinite(u).all():raise ValueError('Finite displacement[...,N,3]')
    # Remove best infinitesimal translation/rotation, not strain/scale/shear.
    X=forward.X-forward.X.mean(0);A=np.zeros((len(X)*3,6))
    for i,(x,y,z) in enumerate(X):A[3*i:3*i+3,:3]=np.eye(3);A[3*i:3*i+3,3:]=[[0,z,-y],[-z,0,x],[y,-x,0]]
    flat=u.reshape(-1,len(X)*3);fit=np.linalg.lstsq(A,flat.T,rcond=None)[0].T@A.T;nonrigid=flat-fit
    rms=np.sqrt(np.mean(flat**2,axis=1));nr=np.sqrt(np.mean(nonrigid**2,axis=1))
    grad=displacement_gradient(forward,u)
    strain=.5*(grad+grad.transpose(0,1,3,2));J=np.linalg.det(np.eye(3)+grad)
    return dict(fit_scope='Uniformnodal infinitesimalrigidfit; notvolumeproperfitqualification',motion_RMS_m=rms,nonrigid_RMS_m=nr,nonrigid_fraction=np.divide(nr,rms,out=np.zeros_like(nr),where=rms>0),maximum_linear_strain=np.max(abs(strain),axis=(1,2,3)),minimum_J=J.min(1))
