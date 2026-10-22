"""NEW consistent-surface P1 Neo-Hookean backward Euler mechanics.

Elasticity, physical lumped solid mass and sparse Newton/Armijo semantics inherit
from the frozen sparse reference. Only the explicitly new surface discretization
changes: normalized load integrals and full fixture Galerkin Gram matrices.
The full step override retains the reference optimizer/ledger, replacing BOTH
lumped quadratic terms coherently. Rest-central world dead loads are not follower
loads, and need not have zero torque after deformation. No observation access.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
try:
    from .campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian,SparseStepFailure
    from .campaign_quality_tet_refinement import geometry
    from .campaign_compliant_gaussian_linear import candidate_patches
    from .campaign_consistent_surface_gaussians import assemble_surface_gaussians
    from .diagnose_nonlinear_surface_quadrature import adaptive_moments
except ImportError:
    from campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian,SparseStepFailure
    from campaign_quality_tet_refinement import geometry
    from campaign_compliant_gaussian_linear import candidate_patches
    from campaign_consistent_surface_gaussians import assemble_surface_gaussians
    from diagnose_nonlinear_surface_quadrature import adaptive_moments


@lru_cache(maxsize=8)
def _canonical_reference(load_std_scale):
    if not np.isfinite(load_std_scale) or load_std_scale<=0:raise ValueError('Positive load standard deviation scale required')
    X,_,F=geometry(0);patches,_,_=candidate_patches()
    centers=np.array([p.center_material_m for p in patches])
    cov=np.array([p.covariance_material_m2 for p in patches])*load_std_scale**2
    ref=adaptive_moments(X,F,centers,cov)
    mu=ref['centroids_material_m'];axis=mu[1]-mu[0];axis=axis/np.linalg.norm(axis)
    return mu,axis,ref


def canonical_load_definition(load_std_scale=1.5):
    """Continuous six-original-triangle reference, independent of FE mesh."""
    mu,axis,ref=_canonical_reference(float(load_std_scale))
    return dict(centroids_material_m=mu.copy(),axis_world=axis.copy(),
                normalizers_m2=ref['normalizers_m2'].copy(),adaptive_checks=[dict(c) for c in ref['checks']],
                load_std_scale=float(load_std_scale),law='rest-central tension; fixed world dead loads')


def tension_command(t,amplitude,duration,axis=None):
    if not all(np.isfinite(a) for a in [t,amplitude,duration]) or amplitude<0 or duration<=0:raise ValueError('Finite time/nonnegative amplitude/positive duration required')
    a=canonical_load_definition()['axis_world'] if axis is None else np.asarray(axis,dtype=np.float64)
    if a.shape!=(3,) or not np.isfinite(a).all() or abs(np.linalg.norm(a)-1)>1e-12:raise ValueError('Explicit unit world axis required')
    pulse=np.sin(np.pi*t/duration)**2 if 0<t<duration else 0.
    return amplitude*pulse*np.stack([-a,a])


class ConsistentSurfaceNonlinearGaussian(SparseNonlinearGaussian):
    def __init__(self,points,tets,faces,*,load_std_scale=1.5,quadrature_order=12,
                 frame_rotation=None,frame_translation=None,**kwargs):
        Q=np.eye(3) if frame_rotation is None else np.asarray(frame_rotation,dtype=np.float64)
        shift=np.zeros(3) if frame_translation is None else np.asarray(frame_translation,dtype=np.float64)
        if Q.shape!=(3,3) or shift.shape!=(3,) or not np.isfinite(Q).all() or not np.isfinite(shift).all() or np.max(abs(Q.T@Q-np.eye(3)))>1e-12 or abs(np.linalg.det(Q)-1)>1e-12:raise ValueError('Proper rigid frame required')
        if not np.isfinite(load_std_scale) or load_std_scale<=0:raise ValueError('Positive load scale required')
        # Input geometry is in canonical material coordinates; transform all fields.
        X=np.asarray(points,dtype=np.float64)@Q.T+shift
        super().__init__(X,tets,faces,**kwargs)
        patches,K,D=candidate_patches(Q,shift)
        centers=np.array([p.center_material_m for p in patches]);cov=np.array([p.covariance_material_m2 for p in patches])
        self.fixture_operators=assemble_surface_gaussians(self.rest,faces,centers,cov,order=quadrature_order)
        self.load_operators=assemble_surface_gaussians(self.rest,faces,centers,cov*load_std_scale**2,order=quadrature_order)
        self.Kpatch=K*float(kwargs.get('fixture_scale',2.));self.Dpatch=D
        self.Kparts=tuple(sparse.kron(M,sparse.csr_matrix(A),format='csr') for M,A in zip(self.fixture_operators.fixture_grams,self.Kpatch))
        self.Cparts=tuple(sparse.kron(M,sparse.csr_matrix(A),format='csr') for M,A in zip(self.fixture_operators.fixture_grams,self.Dpatch))
        self.K=sum(self.Kparts);self.C=sum(self.Cparts);self.load_vectors=self.load_operators.load_vectors
        self.metric_cache={};self.load_std_scale=float(load_std_scale);self.quadrature_order=quadrature_order
        reference=canonical_load_definition(load_std_scale)
        self.canonical_centroids_material_m=reference['centroids_material_m'].copy()
        self.canonical_centroids_world_m=self.canonical_centroids_material_m@Q.T+shift
        self.load_axis_world=Q@reference['axis_world'];self.frame_rotation=Q.copy();self.frame_translation=shift.copy()
        self.canonical_load_metadata=reference
        self.boundary_semantics='consistent normalized surface Gram K/D and load integrals; rest-central dead loads'
        # No silent downstream use of inherited vertex-lumped arrays.
        for name in ['weights','load_weights','load_active','Knode','Dnode']:delattr(self,name)

    @classmethod
    def from_design(cls,mesh_level=0,*,level=None,**kwargs):
        if level is not None:
            if mesh_level!=0:raise ValueError('Specify level or mesh_level, not both')
            mesh_level=level
        return cls(*geometry(mesh_level),**kwargs)

    @staticmethod
    def patch_quadratic(vectors,operators):
        return float(sum(a.ravel()@(M@a.ravel()) for a,M in zip(vectors,operators)))

    def known_force(self,forces,gravity=(0,0,0)):
        F=np.asarray(forces,dtype=np.float64);g=np.asarray(gravity,dtype=np.float64)
        if F.shape!=(2,3) or g.shape!=(3,) or not np.isfinite(F).all() or not np.isfinite(g).all():raise ValueError('Explicit finite H2 force/gravity required')
        return self.load_vectors.T@F+self.node_mass[:,None]*g

    def fixture(self,x,v,target,tv):
        e=np.asarray(x)[None]-target;rel=np.asarray(v)[None]-tv
        spring=np.stack([-(M@a.ravel()).reshape(self.rest.shape) for M,a in zip(self.Kparts,e)])
        drag=np.stack([-(M@a.ravel()).reshape(self.rest.shape) for M,a in zip(self.Cparts,rel)])
        return .5*self.patch_quadratic(e,self.Kparts),spring+drag,self.patch_quadratic(rel,self.Cparts)

    def load_torque(self,x,forces,origin=(0,0,0)):
        """Actual deformed-configuration external torque, no zero-torque assertion."""
        return np.cross(np.asarray(x)-np.asarray(origin),self.known_force(forces)).sum(0)

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
            damping=.5/h*self.patch_quadratic(d,self.Cparts)
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
        numerical=.5*np.sum(self.mass*(vel-v).ravel()**2)+.5*self.patch_quadratic(de,self.Kparts)+np.sum(grad*dx)-(U1-U0)
        balance=delta-external-targetwork-reaction+h*diss+numerical
        return q,vel,{'iterations':iteration,'force_residual_N':float(residual),'threshold_N':float(tolerance),'minimum_J':float(J.min()),'balance_error_J':float(balance),'free_residual_work_J':float(np.sum(r.ravel()[self.free]*dx.ravel()[self.free])),'damping_dissipation_J':float(h*diss),'external_work_J':float(external),'target_work_J':float(targetwork),'reaction_work_J':float(reaction),'constraint_max_error_m':float(np.max(abs(q[self.mask]-prescribed[self.mask]))) if self.mask.any() else 0.,'history':history}
