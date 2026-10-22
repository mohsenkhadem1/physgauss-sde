"""Independent analytic sparse P1 Neo-Hookean/Gaussian backward Euler.

Same physical energy as the frozen Torch tiny reference; no inversion, material,
constraint or stiffness repair. Numerical descent fallback changes directions
only, never objective or stationarity. CPU NumPy/SciPy, no cohort observations.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
try:
    from . import campaign_boundary_gaussians as boundary
    from .campaign_nonlinear_excitation import geometry,command
    from .campaign_compliant_gaussian_linear import candidate_patches
except ImportError:
    import campaign_boundary_gaussians as boundary
    from campaign_nonlinear_excitation import geometry,command
    from campaign_compliant_gaussian_linear import candidate_patches


class SparseStepFailure(RuntimeError):
    def __init__(self,message,history):super().__init__(message);self.history=history


class SparseNonlinearGaussian:
    def __init__(self,points,tets,faces,*,young=5000.,poisson=.35,density=625.,fixture_scale=2.,constraint_mask=None,minimum_J=.1):
        self.rest=np.asarray(points,dtype=np.float64).copy();self.tets=np.asarray(tets,dtype=np.int64)
        self.young,self.poisson,self.density=map(float,(young,poisson,density));self.minimum_J=float(minimum_J)
        if not (young>0 and density>0 and -1<poisson<.5 and fixture_scale>0 and 0<minimum_J<1):raise ValueError('Invalid material/fixture/J settings')
        corners=self.rest[self.tets];edges=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2)
        det=np.linalg.det(edges)
        if not np.all(np.isfinite(det)) or np.any(det<=0):raise ValueError('Positive authored rest tetrahedra required')
        self.volumes=det/6;inverse=np.linalg.inv(edges)
        self.gradients=np.concatenate([-inverse.sum(axis=1)[:,None],inverse],axis=1)
        nodal_mass=np.zeros(len(points));np.add.at(nodal_mass,self.tets,(density*self.volumes[:,None]/4)*np.ones((1,4)))
        if np.any(nodal_mass<=0):raise ValueError('Every node must have positive physical solid mass')
        self.node_mass=nodal_mass;self.mass=np.repeat(nodal_mass,3)
        patches,K,D=candidate_patches();quad=boundary.SurfaceQuadrature.from_mesh(self.rest,faces)
        support=boundary.patch_bank(quad,patches,2,{'candidate_boundary_hypothesis'})
        load=[boundary.GaussianPatch(p.center_material_m,p.covariance_material_m2,'known_hardware') for p in patches]
        bank=boundary.patch_bank(quad,load,2,{'known_hardware'})
        self.weights=np.asarray(support['normalized_node_weights']);self.load_weights=bank['normalized_node_weights'];self.load_active=bank['active_mask']
        self.Kpatch=np.stack([boundary.psd(a*fixture_scale,'K') for a in K]);self.Dpatch=np.stack([boundary.psd(a,'D') for a in D])
        self.Knode=np.einsum('sn,sij->nij',self.weights,self.Kpatch);self.Dnode=np.einsum('sn,sij->nij',self.weights,self.Dpatch)
        n=len(points);indices=np.arange(n);indptr=np.arange(n+1)
        self.K=sparse.bsr_matrix((self.Knode,indices,indptr),shape=(3*n,3*n)).tocsr()
        self.C=sparse.bsr_matrix((self.Dnode,indices,indptr),shape=(3*n,3*n)).tocsr()
        self.mask=np.zeros_like(self.rest,dtype=bool) if constraint_mask is None else np.asarray(constraint_mask)
        if self.mask.dtype!=np.bool_ or self.mask.shape!=self.rest.shape:raise ValueError('Explicit node-axis boolean mask required')
        self.free=np.flatnonzero(~self.mask.ravel());dofs=(self.tets[:,:,None]*3+np.arange(3)).reshape(-1,12)
        self.rows=np.repeat(dofs,12,axis=1).ravel();self.cols=np.tile(dofs,(1,12)).ravel();self.metric_cache={}

    @classmethod
    def from_design(cls,mesh_level=0,**kwargs):return cls(*geometry(mesh_level),**kwargs)

    def elastic(self,x,*,tangent=False):
        x=np.asarray(x);G=np.einsum('eai,eaJ->eiJ',(x-self.rest)[self.tets],self.gradients)
        I=np.eye(3);F=G+I;trace=np.trace(G,axis1=-2,axis2=-1)
        second=.5*(trace**2-np.einsum('eij,eji->e',G,G));detG=np.linalg.det(G)
        jm=trace+second+detG;J=1+jm
        if not np.all(np.isfinite(J)) or np.any(J<=self.minimum_J):raise ValueError('Nonpositive/near-zero J; no repair')
        log=np.log1p(jm);mu=self.young/(2*(1+self.poisson));lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson))
        small=sum((-1.)**k*jm**k/k for k in range(2,10));xm=np.where(abs(jm)<1e-3,small,jm-log)
        U=float(np.sum(self.volumes*(mu*(.5*np.sum(G**2,axis=(1,2))-second-detG+xm)+.5*lam*log**2)))
        invT=np.swapaxes(np.linalg.inv(F),-1,-2)
        # Algebraically identical P, cancellation-resistant at rest.
        P=mu*G+(lam*log-mu)[:,None,None]*(invT-I)+lam*log[:,None,None]*I
        local=np.einsum('eiJ,eaJ,e->eai',P,self.gradients,self.volumes)
        grad=np.zeros_like(self.rest);np.add.at(grad,self.tets,local)
        H=None
        if tangent:
            A=mu*np.einsum('ik,JL->iJkL',I,I)[None]
            A=A+(mu-lam*log)[:,None,None,None,None]*np.einsum('eiL,ekJ->eiJkL',invT,invT)
            A=A+lam*np.einsum('eiJ,ekL->eiJkL',invT,invT)
            blocks=np.einsum('eiJkL,eaJ,ebL,e->eaibk',A,self.gradients,self.gradients,self.volumes,optimize=True).reshape(-1)
            H=sparse.coo_matrix((blocks,(self.rows,self.cols)),shape=(x.size,x.size)).tocsr()
        return U,grad,H,J

    def known_force(self,forces,gravity=(0,0,0)):
        F=np.asarray(forces,dtype=float);g=np.asarray(gravity,dtype=float)
        if F.shape!=(2,3) or g.shape!=(3,) or not np.all(np.isfinite(F)) or not np.all(np.isfinite(g)):raise ValueError('Explicit finite H2 force/gravity required')
        if np.any(F[~self.load_active]!=0):raise ValueError('Inactive Gaussian load carries nonzero force')
        return self.load_weights.T@F+self.node_mass[:,None]*g

    def fixture(self,x,v,target,tv):
        e=x[None]-target;rel=v[None]-tv
        spring=-self.weights[:,:,None]*np.einsum('sij,snj->sni',self.Kpatch,e)
        drag=-self.weights[:,:,None]*np.einsum('sij,snj->sni',self.Dpatch,rel)
        return -.5*np.sum(spring*e),spring+drag,-np.sum(drag*rel)

    def step(self,x,v,target_old,target_new,force,h,*,constraint_positions=None,absolute_tolerance_N=1e-10,relative_tolerance=1e-8,maximum_iterations=50):
        x,v,old,new,force=[np.asarray(a,dtype=np.float64) for a in (x,v,target_old,target_new,force)]
        if h<=0 or not np.isfinite(h) or x.shape!=self.rest.shape or v.shape!=x.shape or force.shape!=x.shape or old.shape!=(2,*x.shape) or new.shape!=old.shape:raise ValueError('Invalid explicit step fields')
        if not all(np.all(np.isfinite(a)) for a in (x,v,old,new,force)):raise ValueError('Nonfinite step fields')
        self.elastic(x);tv=(new-old)/h
        prescribed=x if constraint_positions is None else np.asarray(constraint_positions,dtype=np.float64)
        if self.mask.any() and constraint_positions is None:raise ValueError('Exact constrained positions required')
        if prescribed.shape!=x.shape or not np.all(np.isfinite(prescribed)):raise ValueError('Invalid exact target')
        dx=np.zeros_like(x);dx[self.mask]=(prescribed-x)[self.mask]
        tolerance=max(absolute_tolerance_N,relative_tolerance*np.linalg.norm(force));history=[]
        fixed=sparse.diags(self.mass/h**2)+self.K+self.C/h
        def evaluate(increment,needH=False):
            q=np.where(self.mask,prescribed,x+increment);velocity=increment/h
            U,grad,H,J=self.elastic(q,tangent=needH);Us,Fs,diss=self.fixture(q,velocity,new,tv)
            d=increment[None]-h*tv
            damping=.5/h*np.einsum('sn,sni,sij,snj->',self.weights,d,self.Dpatch,d)
            potential=.5*np.sum(self.mass*(increment-h*v).ravel()**2)/h**2+U+Us+damping-np.sum(force*increment)
            r=(self.mass*((velocity-v)/h).ravel()).reshape(x.shape)+grad-Fs.sum(0)-force
            if not np.isfinite(potential) or not np.all(np.isfinite(r)):raise SparseStepFailure('Nonfinite original potential/residual',history)
            return potential,r,H,J,U,Us,Fs,diss,grad
        for iteration in range(maximum_iterations+1):
            value,r,H,J,*_=evaluate(dx,True);rf=r.ravel()[self.free];norm=np.linalg.norm(rf)
            history.append({'iteration':iteration,'original_residual_N':float(norm),'minimum_J':float(J.min()),'algorithmic_shift_N_m':0.})
            if norm<=tolerance:break
            if iteration==maximum_iterations:raise SparseStepFailure('Original residual exceeds cap',history)
            matrix=(H+fixed)[self.free][:,self.free].tocsc()
            try:direction=splu(matrix).solve(-rf);kind='analytic_sparse_Newton'
            except RuntimeError:direction=None
            if direction is None or not np.all(np.isfinite(direction)) or rf@direction>=0:
                if h not in self.metric_cache:self.metric_cache[h]=splu(fixed[self.free][:,self.free].tocsc())
                direction=self.metric_cache[h].solve(-rf);kind='mass_fixture_metric_descent'
            slope=float(rf@direction);history[-1]['direction']=kind
            if not np.isfinite(slope) or slope>=0:raise SparseStepFailure('No descent direction',history)
            for backtrack in range(40):
                alpha=2.**(-backtrack);trial=dx.copy();trial.ravel()[self.free]+=alpha*direction
                try:v2,r2,*_=evaluate(trial)
                except ValueError:continue
                window=128*np.finfo(float).eps*max(abs(value),self.young*self.volumes.sum(),1e-30)
                if v2<=value+1e-4*alpha*slope or (abs(v2-value)<=window and np.linalg.norm(r2.ravel()[self.free])<norm):dx=trial;history[-1]['alpha']=alpha;break
            else:raise SparseStepFailure('Positive-J Armijo exhausted',history)
        q=np.where(self.mask,prescribed,x+dx);vel=dx/h
        _,r,_,J,U1,Us1,Fs,diss,grad=evaluate(dx)
        residual=np.linalg.norm(r.ravel()[self.free])
        if residual>tolerance:raise SparseStepFailure('Final original stationarity failed',history)
        U0=self.elastic(x)[0];Us0=self.fixture(x,v,old,np.zeros_like(old))[0]
        delta=.5*np.sum(self.mass*(vel.ravel()**2-v.ravel()**2))+U1+Us1-U0-Us0
        external=np.sum(force*dx);targetwork=np.sum(Fs*(new-old));reaction=np.sum(r[self.mask]*dx[self.mask])
        de=dx[None]-(new-old)
        numerical=.5*np.sum(self.mass*(vel-v).ravel()**2)+.5*np.einsum('sn,sni,sij,snj->',self.weights,de,self.Kpatch,de)+np.sum(grad*dx)-(U1-U0)
        balance=delta-external-targetwork-reaction+h*diss+numerical
        return q,vel,{'iterations':iteration,'force_residual_N':float(residual),'threshold_N':float(tolerance),'minimum_J':float(J.min()),'balance_error_J':float(balance),'free_residual_work_J':float(np.sum(r.ravel()[self.free]*dx.ravel()[self.free])),'damping_dissipation_J':float(h*diss),'external_work_J':float(external),'target_work_J':float(targetwork),'reaction_work_J':float(reaction),'constraint_max_error_m':float(np.max(abs(q[self.mask]-prescribed[self.mask]))) if self.mask.any() else 0.,'history':history}
