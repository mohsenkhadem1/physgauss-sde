"""Independent static uniaxial oracle for sparse compressible Neo-Hookean P1.

Uniform right dead traction, finite distributed LEFT axial spring, and exact
transverse symmetry constraints. No Gaussian fixture, mass, inertia, fictitious
time step, inverse fit or constitutive/tangent repair enters the static solve.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import splu
from experiments.boundary_block import block_mesh
from experiments.campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian

SIZE=(.24,.12,.08)
YOUNG=5000.
POISSON=.35
SPRING=200.


def lame(young=YOUNG,poisson=POISSON):
    if not np.isfinite(young) or not np.isfinite(poisson) or young<=0 or not 0<poisson<.5:
        raise ValueError('Positive finite E and0<nu<.5 required')
    return young/(2*(1+poisson)),young*poisson/((1+poisson)*(1-2*poisson))


def analytic_solution(force,*,young=YOUNG,poisson=POISSON,spring=SPRING,size=SIZE):
    """Independent homogeneous traction solution; totalforce in reference area."""
    if not np.isfinite(force) or force<0 or not np.isfinite(spring) or spring<=0:raise ValueError('Known nonnegative force/finite spring required')
    mu,lam=lame(young,poisson);L,W,H=size;area=W*H
    def lateral(s):
        return brentq(lambda y:y-1+(lam/mu)*np.log(s*y),1e-4,1.,xtol=5e-15,rtol=1e-14)
    def axial(s):return mu*(s-lateral(s)/s)
    stretch=1. if force==0 else brentq(lambda s:area*axial(s)-force,1.,1.5,xtol=5e-15,rtol=1e-14)
    y=lateral(stretch);t=np.sqrt(y);J=stretch*y;P=mu*(stretch-y/stretch)
    Wdensity=.5*mu*(stretch**2+2*y-3)-mu*np.log(J)+.5*lam*np.log(J)**2
    shift=force/spring
    return dict(stretch=stretch,lateral_stretch=t,lateral_stretch_squared=y,J=J,
        first_Piola_axial_Pa=P,lateral_Piola_Pa=float(mu*(t-1/t)+lam*np.log(J)/t),
        left_translation_m=shift,right_translation_m=shift+(stretch-1)*L,
        elastic_energy_J=L*area*Wdensity,spring_energy_J=.5*spring*shift**2,
        external_work_J=force*(shift+(stretch-1)*L),force_N=float(force),spring_N_m=float(spring))


def face_area_weights(points,faces,axis,bound):
    selected=faces[np.all(np.abs(points[faces,axis]-bound)<=1e-13,axis=1)]
    if not len(selected):raise ValueError('Declared prism end has no triangles')
    p=points[selected];area=.5*np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    weights=np.zeros(len(points))
    for column in range(3):np.add.at(weights,selected[:,column],area/3)
    return weights,selected


class StaticFailure(RuntimeError):
    def __init__(self,message,history):super().__init__(message);self.history=history


class PrismaticBar:
    """Reuse ONLY immutable analytic elastic energy/gradient/sparse tangent.

    All boundary/force/static assembly below is new and independent of the
    Gaussian dynamic boundary system. Local DOFs allow exactly transformed
    symmetry constraints under a passive rigid frame change.
    """
    def __init__(self,cells=(2,2,2),*,young=YOUNG,poisson=POISSON,spring=SPRING,force=1.,rotation=None,translation=None):
        lame(young,poisson)
        if not np.isfinite(spring) or spring<=0 or not np.isfinite(force) or force<0:raise ValueError('Finite positiveK/nonnegative knownF required')
        self.rotation=np.eye(3) if rotation is None else np.asarray(rotation,dtype=float)
        self.translation=np.zeros(3) if translation is None else np.asarray(translation,dtype=float)
        if self.rotation.shape!=(3,3) or not np.isfinite(self.rotation).all() or not np.allclose(self.rotation.T@self.rotation,np.eye(3),rtol=0,atol=1e-13) or abs(np.linalg.det(self.rotation)-1)>1e-13:raise ValueError('Proper finite rotation required')
        if self.translation.shape!=(3,) or not np.isfinite(self.translation).all():raise ValueError('Finite translation required')
        if any(type(n) is not int or n<2 or n%2 for n in cells):raise ValueError('Even prism cell counts>=2 for exact symmetry planes')
        self.reference_local,self.tets,self.faces=block_mesh(SIZE,cells,(SIZE[0]/2,0,0))
        self.rest=self.reference_local@self.rotation.T+self.translation
        self.young,self.poisson,self.spring,self.force=map(float,(young,poisson,spring,force));self.minimum_J=.1
        corners=self.rest[self.tets];edges=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2);det=np.linalg.det(edges)
        if np.any(det<=0):raise ValueError('Positive authored rest winding required')
        self.volumes=det/6;inverse=np.linalg.inv(edges)
        self.gradients=np.concatenate([-inverse.sum(axis=1)[:,None],inverse],axis=1)
        dofs=(self.tets[:,:,None]*3+np.arange(3)).reshape(-1,12)
        self.rows=np.repeat(dofs,12,axis=1).ravel();self.cols=np.tile(dofs,(1,12)).ravel()
        self.left_area,_=face_area_weights(self.reference_local,self.faces,0,0.)
        self.right_area,_=face_area_weights(self.reference_local,self.faces,0,SIZE[0]);area=SIZE[1]*SIZE[2]
        if abs(self.left_area.sum()-area)>1e-15 or abs(self.right_area.sum()-area)>1e-15:raise ValueError('Exact prism endarea mismatch')
        self.left_weights=self.left_area/area;self.right_weights=self.right_area/area
        self.Kdiagonal=np.zeros_like(self.rest);self.Kdiagonal[:,0]=spring*self.left_weights
        self.load=np.zeros_like(self.rest);self.load[:,0]=force*self.right_weights
        self.mask=np.zeros_like(self.rest,dtype=bool)
        self.mask[:,1]=abs(self.reference_local[:,1])<1e-13;self.mask[:,2]=abs(self.reference_local[:,2])<1e-13
        self.free=np.flatnonzero(~self.mask.ravel())
        self.T=sparse.kron(sparse.eye(len(self.rest),format='csr'),self.rotation,format='csr')
        self.spring_matrix=sparse.diags(self.Kdiagonal.ravel(),format='csr')

    def elastic(self,positions,tangent=False):return SparseNonlinearGaussian.elastic(self,positions,tangent=tangent)

    def evaluate(self,u,tangent=False):
        u=np.asarray(u,dtype=float)
        if u.shape!=self.rest.shape or not np.isfinite(u).all():raise ValueError('Finite local displacement field required')
        positions=self.rest+u@self.rotation.T;elastic,grad,H,J=self.elastic(positions,tangent)
        spring_energy=.5*np.sum(self.Kdiagonal*u**2);work=np.sum(self.load*u)
        residual=grad@self.rotation+self.Kdiagonal*u-self.load
        tangent_local=None if H is None else self.T.T@H@self.T+self.spring_matrix
        return dict(potential_J=float(elastic+spring_energy-work),elastic_J=elastic,spring_J=float(spring_energy),work_J=float(work),
            residual=residual,tangent=tangent_local,J=J,positions=positions)

    def solve(self,initial=None,maximum_iterations=40):
        u=np.zeros_like(self.rest) if initial is None else np.asarray(initial,dtype=float).copy()
        if u.shape!=self.rest.shape or not np.isfinite(u).all() or np.any(u[self.mask]!=0):raise ValueError('Initial state must satisfy exact local symmetry constraints')
        tolerance=max(1e-10,1e-10*self.force);history=[]
        for iteration in range(maximum_iterations+1):
            state=self.evaluate(u,True);r=state['residual'].ravel()[self.free];norm=float(np.linalg.norm(r))
            history.append(dict(iteration=iteration,residual_N=norm,potential_J=state['potential_J'],minimum_J=float(state['J'].min())))
            if norm<=tolerance:
                state.update(displacement_local_m=u,iterations=iteration,history=history,tolerance_N=tolerance)
                return state
            if iteration==maximum_iterations:raise StaticFailure('Static original-force residual exceeds cap',history)
            matrix=state['tangent'][self.free][:,self.free].tocsc()
            try:direction=splu(matrix).solve(-r)
            except RuntimeError as error:raise StaticFailure('Unmodified static tangent factorization failed',history) from error
            slope=float(r@direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope>=0:raise StaticFailure('Unmodified Newton direction not descent',history)
            for backtrack in range(40):
                alpha=2.**(-backtrack);trial=u.copy();trial.ravel()[self.free]+=alpha*direction
                try:candidate=self.evaluate(trial)
                except ValueError:continue
                # Near the float64 energy floor, also require decreasing FORCE
                # residual; this is acceptance arithmetic, not a changed potential.
                roundoff=128*np.finfo(float).eps*max(abs(state['potential_J']),self.young*self.volumes.sum(),1e-30)
                if candidate['potential_J']<=state['potential_J']+1e-4*alpha*slope or (abs(candidate['potential_J']-state['potential_J'])<=roundoff and np.linalg.norm(candidate['residual'].ravel()[self.free])<norm):
                    u=trial;history[-1]['alpha']=alpha;break
            else:raise StaticFailure('PositiveJ static Armijo exhausted',history)
        raise AssertionError('Unreachable static solve')

    def oracle_field(self):
        reference=analytic_solution(self.force,young=self.young,poisson=self.poisson,spring=self.spring)
        u=self.reference_local*(np.array([reference['stretch'],reference['lateral_stretch'],reference['lateral_stretch']])-1)
        u[:,0]+=reference['left_translation_m']
        return u,reference

    def independent_stress(self,positions):
        F=np.eye(3)+np.einsum('eai,eaJ->eiJ',(positions-self.rest)[self.tets],self.gradients)
        mu,lam=lame(self.young,self.poisson);J=np.linalg.det(F)
        # Direct textbook P, independently evaluated from the cancellation-
        # resistant constitutive expression under test.
        P=mu*F+(lam*np.log(J)-mu)[:,None,None]*np.swapaxes(np.linalg.inv(F),-1,-2)
        return F,P
