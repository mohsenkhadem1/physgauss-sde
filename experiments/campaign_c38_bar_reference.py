"""C38 small homogeneous 3D reference: finite normal spring, known dead load.

No Gaussian boundary generalization or new cohort access. Reuses independently
validated compressible logarithmic Neo-Hookean P2 elasticity unchanged. The
law is mu/2*(tr(F.T F)-3)-mu*log(J)+lambda/2*log(J)**2. This is NOT the
polynomial stable-NH potential used by the new GPU block. The left spring uses the
consistent quadratic face Gram, not lumped corner masses or exact pinning.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from numpy.polynomial.legendre import leggauss
from experiments.campaign_static_p2_block import QuadraticBlock,EDGES,enrich,duffy,shapes
from experiments.boundary_block import block_mesh
from experiments.campaign_static_uniaxial_bar import SIZE,POISSON,analytic_solution
from experiments.campaign_nonlinear_bar_inverse import stretch

LOWER=np.log([4500.,100.]);UPPER=np.log([9000.,400.])
PREFIX_FORCE=1.;POKE_FORCE=2.;TARGET_FORCE=1.5;SIGMA_M=.0002


def prior_physical(unit):
    u=np.asarray(unit,dtype=float)
    if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all() or ((u<0)|(u>1)).any():raise ValueError('Unit log-E/log-K prior coordinates required')
    return np.exp(LOWER+u*(UPPER-LOWER))


def oracle_positions(theta,force,points):
    """Exact homogeneous 3D logarithmic NH solution, not a linearized field."""
    theta=np.asarray(theta,dtype=float);X=np.asarray(points,dtype=float)
    if theta.ndim!=2 or theta.shape[1]!=2 or not np.isfinite(theta).all() or (theta<=0).any() or X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all() or not np.isfinite(force) or force<0:raise ValueError('Positive E/K, finite reference points and nonnegative dead force required')
    s,y=stretch(theta[:,0],np.array([force]));scale=np.stack([s[:,0],np.sqrt(y[:,0]),np.sqrt(y[:,0])],axis=-1)
    q=X[None]*scale[:,None];q[:,:,0]+=force/theta[:,1,None]
    return q


def endpoint_displacements(theta,forces):
    theta=np.asarray(theta,dtype=float);F=np.asarray(forces,dtype=float)
    if theta.ndim!=2 or theta.shape[1]!=2 or not np.isfinite(theta).all() or (theta<=0).any():raise ValueError('Positive finite E/K required')
    s,_=stretch(theta[:,0],F);left=F[None]/theta[:,1,None]
    return np.stack([left,left+SIZE[0]*(s-1)],axis=-1)


def material_quadrature():
    """Positive 2x2x2 volume rule: exact affine-field squared-error/risk integral."""
    a=1/np.sqrt(3);x=SIZE[0]/2*(1+np.array([-a,a]));y=SIZE[1]/2*np.array([-a,a]);z=SIZE[2]/2*np.array([-a,a])
    return np.stack(np.meshgrid(x,y,z,indexing='ij'),axis=-1).reshape(-1,3)


class P2SpringBlock(QuadraticBlock):
    def __init__(self,*,young=6000.,spring=220.,force=2.,cells=(2,2,2)):
        if not np.isfinite(spring) or spring<=0 or not np.isfinite(force) or force<0:raise ValueError('Positive spring and nonnegative known force required')
        # Author the oracle's .24 x .12 x .08 geometry explicitly. The parent
        # block constructor has a DIFFERENT frozen width and is not called.
        if any(type(n) is not int or n<2 or n%2 for n in cells) or not np.isfinite(young) or young<=0:raise ValueError('Even positive prism cells and positive E required')
        X,self.tets,self.faces=block_mesh(SIZE,cells,(SIZE[0]/2,0,0));self.reference_local,self.elements=enrich(X,self.tets);self.rest=self.reference_local
        corners=X[self.tets];Dm=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2);det=np.linalg.det(Dm)
        if (det<=0).any():raise ValueError('Positive reference tet winding required')
        inv=np.linalg.inv(Dm);self.linear_gradients=np.concatenate([-inv.sum(1)[:,None],inv],1);self.volumes=det/6
        self.young=float(young);self.poisson=POISSON;self.density=625.;self.minimum_J=.1;self.order=3
        self.bary,self.weights=duffy(3);self.N=shapes(self.bary);self.check_bary=np.concatenate([np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES])])
        dofs=(self.elements[:,:,None]*3+np.arange(3)).reshape(-1,30);self.rows=np.repeat(dofs,30,1).ravel().astype(np.int32);self.cols=np.tile(dofs,(1,30)).ravel().astype(np.int32)
        self.rotation=np.eye(3);self.translation=np.zeros(3);self.estimated_peak_memory_bytes=int(len(self.elements)*900*32+len(self.rest)*3*20000)
        self.spring=float(spring);self.force=float(force)
        self.mask=np.zeros_like(self.rest,dtype=bool);self.mask[:,1]=abs(self.rest[:,1])<1e-13;self.mask[:,2]=abs(self.rest[:,2])<1e-13;self.free=np.flatnonzero(~self.mask.ravel())
        edges={tuple(sorted(map(int,t[[i,j]]))):int(e[4+k]) for t,e in zip(self.tets,self.elements) for k,(i,j) in enumerate(EDGES)}
        X=self.rest[:self.tets.max()+1];left=self.faces[np.all(abs(X[self.faces,0])<1e-13,axis=1)]
        z,w=leggauss(3);z=(z+1)/2;w=w/2;gram=np.zeros((6,6))
        for i,a in enumerate(z):
            for j,c in enumerate(z):
                b=np.array([(1-a)*(1-c),a,(1-a)*c]);N=np.r_[b*(2*b-1),4*b[0]*b[1],4*b[0]*b[2],4*b[1]*b[2]]
                gram+=2*w[i]*w[j]*(1-a)*np.outer(N,N)
        rows=[];cols=[];values=[]
        for f in left:
            ids=np.r_[f,[edges[tuple(sorted(map(int,f[[i,j]])))] for i,j in ((0,1),(0,2),(1,2))]];dofs=3*ids
            area=.5*np.linalg.norm(np.cross(X[f[1]]-X[f[0]],X[f[2]]-X[f[0]]))
            rows.extend(np.repeat(dofs,6));cols.extend(np.tile(dofs,6));values.extend((spring*area/(SIZE[1]*SIZE[2])*gram).ravel())
        self.spring_matrix=sparse.coo_matrix((values,(rows,cols)),shape=(self.rest.size,self.rest.size)).tocsr()
        self.load=np.zeros_like(self.rest)
        right=self.faces[np.all(abs(X[self.faces,0]-SIZE[0])<1e-13,axis=1)]
        for f in right:
            ids=np.r_[f,[edges[tuple(sorted(map(int,f[[i,j]])))] for i,j in ((0,1),(0,2),(1,2))]]
            area=.5*np.linalg.norm(np.cross(X[f[1]]-X[f[0]],X[f[2]]-X[f[0]]))
            np.add.at(self.load[:,0],ids,force*area/(SIZE[1]*SIZE[2])*gram.sum(1))

    def evaluate(self,u,tangent=False):
        U,g,H,J=self.elastic(self.rest+u,tangent);sf=(self.spring_matrix@u.ravel()).reshape(u.shape);S=.5*float(np.sum(u*sf));work=float(np.sum(self.load*u))
        return dict(potential_J=U+S-work,elastic_J=U,spring_J=S,work_J=work,residual=g+sf-self.load,tangent=None if H is None else H+self.spring_matrix,J=J,positions=self.rest+u)

    def verify_oracle(self):
        q=oracle_positions(np.array([[self.young,self.spring]]),self.force,self.rest)[0];state=self.evaluate(q-self.rest)
        independent=analytic_solution(self.force,young=self.young,spring=self.spring)
        force=(self.spring_matrix@(q-self.rest).ravel()).reshape(q.shape).sum(0)
        return dict(original_free_residual_N=float(np.linalg.norm(state['residual'].ravel()[self.free])),spring_force_N=force.tolist(),external_force_N=self.load.sum(0).tolist(),elastic_energy_error_J=abs(state['elastic_J']-independent['elastic_energy_J']),spring_energy_error_J=abs(state['spring_J']-independent['spring_energy_J']),minimum_J=float(state['J'].min()))
