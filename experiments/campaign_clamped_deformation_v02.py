"""Prospective richer fixed-fixed block; exact finite-LTI sinusoid/hold response.

No automatic production cases/inference. Exact matrix semigroup removes BE
phase/dissipation error, not finite-element spatial error or linear-model limits.
"""
from __future__ import annotations
import itertools,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from experiments.campaign_linear_fem import assemble_tetrahedra
from experiments.campaign_boundary_gaussians import SurfaceQuadrature
LOWER=np.log([2000.,5.]);UPPER=np.log([8000.,30.]);SIGMA=20e-6
LENGTH=.36;DURATION=2.;RAMP=.3;OUTPUT_H=.05
GATES=dict(minimum_nonrigid_RMS_m=100e-6,minimum_nonrigid_fraction=.3,maximum_linear_strain=.05,minimum_J=.8,maximum_independent_trajectory_error_m=2e-6)


def geometry():
    X=np.array(list(itertools.product(np.linspace(0,LENGTH,9),np.linspace(0,.08,3),np.linspace(0,.08,3))))
    node=lambda i,j,k:i*9+j*3+k;tets=[]
    for i,j,k in itertools.product(range(8),range(2),range(2)):
        for permutation in itertools.permutations(range(3)):
            corner=np.array([i,j,k]);vertices=[corner.copy()]
            for axis in permutation:corner=corner.copy();corner[axis]+=1;vertices.append(corner)
            ids=np.array([node(*v) for v in vertices])
            if np.linalg.det((X[ids[1:]]-X[ids[0]]).T)<0:ids[[1,2]]=ids[[2,1]]
            tets.append(ids)
    tets=np.array(tets);counts={}
    for tet in tets:
        for face in itertools.combinations(tet,3):
            key=tuple(sorted(face));counts[key]=counts.get(key,0)+1
    faces=np.array([f for f,n in counts.items() if n==1]);fixed=np.flatnonzero((X[:,0]==0)|(X[:,0]==LENGTH))
    free=np.flatnonzero(np.repeat((X[:,0]!=0)&(X[:,0]!=LENGTH),3));fixed.setflags(write=False);free.setflags(write=False)
    return X,tets,faces,fixed,free


def physical(unit):
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or unit.shape[1]!=2 or not len(unit) or not np.isfinite(unit).all() or np.any((unit<0)|(unit>1)):raise ValueError('Unit[B,2]insideprior')
    return np.exp(LOWER+unit*(UPPER-LOWER))


def programs():
    X,*_=geometry();mid=np.flatnonzero(np.isclose(X[:,0],LENGTH/2,rtol=0,atol=1e-14));out={}
    for name,direction,amplitude in [('bending',[0,0,1],.16),('stretching',[1,0,0],.72)]:
        f=np.zeros_like(X);f[mid]=amplitude*np.asarray(direction)/len(mid);out[name]=f
    return out


def amplitude(t):
    t=np.asarray(t,float)
    if not np.isfinite(t).all() or np.any(t<0):raise ValueError('Nonnegativefinite time')
    return np.where(t<RAMP,.5*(1-np.cos(np.pi*t/RAMP)),1.)


def design():
    return dict(schema='clamped-deformation-v02',geometry='9x3x3nodes192positiveT4;.36x.08x.08;bothendfaceshardeliminated',
        intrinsic=['logE','logsupportK'],prior_lower=LOWER.tolist(),prior_upper=UPPER.tolist(),support_center_m=[.24,.08,.04],support_width_m=.025,support_damping_Ns_m=.05,
        density_kg_m3=625.,poisson=.35,gravity=False,contact=False,forces=dict(bending_mid_Fz_N=.16,stretching_mid_Fx_N=.72,spatial='Equalat9midcrosssectionnodes'),
        ramp='(1-cos(pi*t/.3))/2until.3s;then1;C1transition',seconds=DURATION,output_h_s=OUTPUT_H,
        solution='Exactaugmentedmatrixexponentialstepandsemigrouppropagation; notBE orothernumericaltimeintegration',
        independent_validation='SeparatelyassembledNumPyK,M,C andDOP853splitat.3;rtol1e-10/atol1e-12,maxstep.0025s;compareall41outputclocks',
        gates=GATES,analytic_before_calls='E8k fixedfixedbenddelta=.16*.36³/(192*8000*.08*.08³/12)=1.424mm;propervolumeNR~.201187delta=286um. Axialdelta=.72*.36/(4*8000*.0064)=1.266mm,properNRdelta/6=211um. E2krootbendstrain.0421875,axial.028125. Fixtures/lockingmayfailgates; noafteroutcometuning.',
        qualification_cases='Allfour(E,K)priorcornersplusgeometriccenter,bothloads; nofilters/retries',
        scope='Finitecoarselinearmodelonly; no continuum/spatialconvergence, calibration oractiveprobingclaim',no_automatic_inference=True)


def source_provenance():
    root=Path(__file__).resolve().parents[1]
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['experiments/campaign_clamped_deformation_v02.py','experiments/campaign_linear_fem.py','experiments/campaign_boundary_gaussians.py']}


def _theta(theta):
    theta=np.asarray(theta,float)
    if theta.ndim!=2 or theta.shape[1]!=2 or not len(theta) or not np.isfinite(theta).all() or np.any(theta<=0):raise ValueError('Positivefinitephysical[E,K]')
    return theta


class ExactClampedForward:
    def __init__(self):
        self.X,self.tets,self.faces,self.fixed,self.free=geometry();ku,m,self.gradients,self.volumes=assemble_tetrahedra(self.X,self.tets,1.,.35,625.)
        ids=torch.tensor(self.free.copy(),dtype=torch.long);self.Kunit=ku[ids][:,ids].numpy();self.mass=m[ids].numpy()
        area=SurfaceQuadrature.from_mesh(self.X,self.faces).node_area_m2
        w=area*np.exp(-.5*np.sum((self.X-[.24,.08,.04])**2,axis=1)/.025**2);self.weights=w/w.sum();self.diagonal=np.repeat(self.weights,3)[self.free]
    def matrices(self,theta):
        E,k=np.asarray(theta,float)
        if not np.isfinite([E,k]).all() or min(E,k)<=0:raise ValueError('PositivephysicalE,K')
        return E*self.Kunit+np.diag(k*self.diagonal),np.diag(.05*self.diagonal),np.diag(self.mass)
    def exacttrajectory(self,theta,forces,times=None,*,return_state=False):
        theta=_theta(theta);forces=np.asarray(forces,float)
        if times is not None and not np.allclose(np.asarray(times),np.arange(41)*OUTPUT_H,atol=1e-14,rtol=0):raise ValueError('Exactly41clocksat20Hz')
        if forces.ndim==2:forces=forces[None]
        if forces.ndim!=3 or forces.shape[1:]!=self.X.shape or not len(forces) or not np.isfinite(forces).all():raise ValueError('Finiteplateauforce[R,N,3]')
        output=[];velocities=[];residuals=[];accels=[];times=np.arange(41)*OUTPUT_H;r=len(forces);d=len(self.mass)
        G=np.zeros((2*d,r));G[d:]=forces.reshape(r,-1)[:,self.free].T/self.mass[:,None]
        for th in theta:
            K,C,M=self.matrices(th);A=np.block([[np.zeros((d,d)),np.eye(d)],[-K/self.mass[:,None],-C/self.mass[:,None]]])
            # Independent constant/cos/sin states for each force RHS.
            L=np.zeros((2*d+3*r,2*d+3*r));L[:2*d,:2*d]=A;L[:2*d,2*d:2*d+r]=.5*G;L[:2*d,2*d+r:2*d+2*r]=-.5*G
            omega=np.pi/RAMP;L[2*d+r:2*d+2*r,2*d+2*r:]=-omega*np.eye(r);L[2*d+2*r:,2*d+r:2*d+2*r]=omega*np.eye(r)
            Z=np.zeros((len(L),r));Z[2*d:2*d+r]=np.eye(r);Z[2*d+r:2*d+2*r]=np.eye(r)
            H=np.zeros((2*d+r,2*d+r));H[:2*d,:2*d]=A;H[:2*d,2*d:]=G
            rampstep=expm(OUTPUT_H*L);holdstep=expm(OUTPUT_H*H);states=[Z[:2*d].copy()]
            for step in range(1,41):
                if step<=6:Z=rampstep@Z
                else:
                    if step==7:Z=np.concatenate([Z[:2*d],np.eye(r)],axis=0)
                    Z=holdstep@Z
                states.append(Z[:2*d].copy())
            state=np.stack(states);u=state[:,:d];v=state[:,d:];acc=np.einsum('ij,tjr->tir',A[d:],state)+amplitude(times)[:,None,None]*G[d:][None]
            residual=self.mass[None,:,None]*acc+np.einsum('ij,tjr->tir',K,u)+np.einsum('ij,tjr->tir',C,v)-amplitude(times)[:,None,None]*forces.reshape(r,-1)[:,self.free].T[None]
            full=np.zeros((r,41,len(self.X)*3));full[:,:,self.free]=u.transpose(2,0,1);output.append(full.reshape(r,41,len(self.X),3))
            fv=np.zeros_like(full);fv[:,:,self.free]=v.transpose(2,0,1);velocities.append(fv.reshape(r,41,len(self.X),3))
            fa=np.zeros_like(full);fa[:,:,self.free]=acc.transpose(2,0,1);accels.append(fa.reshape(r,41,len(self.X),3));residuals.append(np.max(abs(residual),axis=(1,2)))
        trajectory=np.asarray(output)
        return dict(displacement=trajectory,velocity=np.asarray(velocities),acceleration=np.asarray(accels),times=times,maximum_free_residual_N=np.asarray(residuals)) if return_state else trajectory
    trajectory=exacttrajectory


def independent_numpy_assembly():
    X,tets,faces,_,free=geometry();K=np.zeros((3*len(X),)*2);m=np.zeros(len(X));nu=.35
    mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));D=np.zeros((6,6));D[:3,:3]=lam;D[:3,:3]+=np.eye(3)*2*mu;D[3:,3:]=np.eye(3)*mu
    for ids in tets:
        dm=(X[ids[1:]]-X[ids[0]]).T;vol=np.linalg.det(dm)/6;g=np.linalg.inv(dm);g=np.concatenate([-g.sum(0)[None],g]);B=np.zeros((6,12))
        for j,(dx,dy,dz) in enumerate(g):
            a=3*j;B[0,a]=dx;B[1,a+1]=dy;B[2,a+2]=dz;B[3,a+1]=dz;B[3,a+2]=dy;B[4,a]=dz;B[4,a+2]=dx;B[5,a]=dy;B[5,a+1]=dx
        dofs=(ids[:,None]*3+np.arange(3)).reshape(-1);K[np.ix_(dofs,dofs)]+=B.T@D@B*vol;m[ids]+=625*vol/4
    area=np.zeros(len(X))
    for face in faces:area[face]+=np.linalg.norm(np.cross(X[face[1]]-X[face[0]],X[face[2]]-X[face[0]]))/6
    w=area*np.exp(-.5*np.sum((X-[.24,.08,.04])**2,axis=1)/.025**2);w/=w.sum()
    return K[np.ix_(free,free)],np.repeat(m,3)[free],np.repeat(w,3)[free]


def independent_dop853(theta,forces,times=None,*,return_state=False):
    if times is not None and not np.allclose(np.asarray(times),np.arange(41)*OUTPUT_H,atol=1e-14,rtol=0):raise ValueError('Exactly41clocksat20Hz')
    theta=_theta(np.asarray(theta)[None])[0];ku,m,w=independent_numpy_assembly();X,_,_,_,free=geometry();forces=np.asarray(forces,float)
    if forces.ndim==2:forces=forces[None]
    if forces.ndim!=3 or forces.shape[1:]!=X.shape or not np.isfinite(forces).all():raise ValueError('Finiteplateauforces')
    K=theta[0]*ku+np.diag(theta[1]*w);C=np.diag(.05*w);d=len(m);r=len(forces);F=forces.reshape(r,-1)[:,free].T
    def derivative(t,z):
        z=z.reshape(2*d,r);u,v=z[:d],z[d:];return np.concatenate([v,(float(amplitude(t))*F-K@u-C@v)/m[:,None]]).reshape(-1)
    first=np.linspace(0,RAMP,7);second=np.linspace(RAMP,DURATION,35)
    a=solve_ivp(derivative,(0,RAMP),np.zeros(2*d*r),method='DOP853',t_eval=first,rtol=1e-10,atol=1e-12,max_step=.0025)
    if not a.success:raise RuntimeError(a.message)
    b=solve_ivp(derivative,(RAMP,DURATION),a.y[:,-1],method='DOP853',t_eval=second,rtol=1e-10,atol=1e-12,max_step=.0025)
    if not b.success:raise RuntimeError(b.message)
    state=np.concatenate([a.y,b.y[:,1:]],axis=1).T.reshape(41,2*d,r);full=np.zeros((r,41,len(X)*3));full[:,:,free]=state[:,:d].transpose(2,0,1)
    diagnostics=dict(first_nfev=a.nfev,second_nfev=b.nfev,success=True,rtol=1e-10,atol=1e-12,max_step_s=.0025)
    if return_state:
        velocity=np.zeros_like(full);velocity[:,:,free]=state[:,d:].transpose(2,0,1)
        return dict(displacement=full.reshape(r,41,len(X),3),velocity=velocity.reshape(r,41,len(X),3),times=np.arange(41)*OUTPUT_H,diagnostics=diagnostics)
    return full.reshape(r,41,len(X),3),diagnostics


def displacement_gradient(forward,u):
    u=np.asarray(u,float)
    if u.shape[-2:]!=forward.X.shape or not np.isfinite(u).all():raise ValueError('Finite [...,N,3]')
    return np.einsum('btnc,tni->btci',u.reshape(-1,len(forward.X),3)[:,forward.tets],forward.gradients.numpy())


ClampedExactForward=ExactClampedForward
independentDOP853=independent_dop853
