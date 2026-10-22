"""Independent sparse P2 consistent-mass BE-started BDF2, fixed exact clamp.

Full analytic T10 mass; no row lumping or stiffness repair. Stable-NH material
law is reused unchanged. Coordinates/velocities are displacement based.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.campaign_static_p2_block import EDGES,shape_gradients
from experiments.campaign_newton_stable_material import StableP2Block,stable_material
from experiments.campaign_stable_transient import ImplicitFailure


def scalar_mass(model):
    B=np.ones((10,10));B[:4,:4]=1.;np.fill_diagonal(B[:4,:4],6.)
    for i in range(4):
        for j,e in enumerate(EDGES):B[i,4+j]=B[4+j,i]=-4. if i in e else -6.
    for i,a in enumerate(EDGES):
        for j,b in enumerate(EDGES):B[4+i,4+j]=32. if i==j else 16. if set(a)&set(b) else 8.
    rows=np.repeat(model.elements,10,axis=1).ravel();cols=np.tile(model.elements,(1,10)).ravel()
    return sparse.coo_matrix(((model.density*model.volumes[:,None,None]*B[None]/420).ravel(),(rows,cols)),shape=(len(model.rest),len(model.rest))).tocsr()


@dataclass
class P2State:
    u:np.ndarray
    v:np.ndarray
    delta:np.ndarray


def advance_consistent(elastic,mass,mask,current,previous,h,force,energy_scale_J,force_tolerance_N=1e-10,maximum_iterations=20):
    """Solve the original full-mass objective; BE startup, then physical BDF2."""
    u=current.u;force=np.asarray(force,float);mask=np.asarray(mask)
    if h<=0 or not np.isfinite(h) or u.shape!=force.shape or mask.shape!=u.shape or mask.dtype!=np.bool_ or mass.shape!=(len(u),len(u)) or not all(np.isfinite(x).all() for x in [u,current.v,current.delta,force]) or np.any(u[mask]!=0) or np.any(current.v[mask]!=0):raise ValueError('Finite full-mass state and exact fixed clamp required')
    heff=h if previous is None else 2*h/3
    pred=h*current.v if previous is None else current.delta/3+(2*h/9)*(4*current.v-previous.v)
    pred=pred.copy();pred[mask]=0.;factor=1/heff**2;free=np.flatnonzero(~mask.ravel());M=sparse.kron(mass,sparse.eye(3),format='csr');z=np.zeros_like(u);history=[]
    def evaluate(d,tangent):
        E,g,H,J=elastic(u+d,tangent=tangent)
        if not np.isfinite(J).all() or np.any(J<=.1):raise ValueError('Positive-J guard failed; no repair')
        error=d-pred;Me=mass@error;r=g+factor*Me-force
        potential=float(E+.5*factor*np.sum(error*Me)-np.sum(force*d))
        return potential,r,None if H is None else H+factor*M,J,E,g
    for it in range(maximum_iterations+1):
        V,r,H,J,E,g=evaluate(z,True);norm=float(np.linalg.norm(r.ravel()[free]));history.append({'iteration':it,'objective_free_R_N':norm,'minimum_J':float(np.min(J))})
        if norm<=force_tolerance_N:break
        if it==maximum_iterations:raise ImplicitFailure('Consistent P2 original objective iteration cap',history)
        direction=splu(H[free][:,free].tocsc()).solve(-r.ravel()[free]);slope=float(direction@r.ravel()[free])
        if not np.isfinite(direction).all() or not np.isfinite(slope) or slope>=0:raise ImplicitFailure('Original tangent direction not descent',history)
        for k in range(30):
            alpha=2.**(-k);trial=z.copy();trial.ravel()[free]+=alpha*direction
            try:VV,rr,_,_,_,_=evaluate(trial,False)
            except ValueError:continue
            roundoff=128*np.finfo(float).eps*max(abs(V),abs(E),energy_scale_J)
            if VV<=V+1e-4*alpha*slope or(abs(VV-V)<=roundoff and np.linalg.norm(rr.ravel()[free])<norm):z=trial;history[-1]['alpha']=alpha;break
        else:raise ImplicitFailure('Positive-J original-objective Armijo failed',history)
    v=z/h if previous is None else (3*z-current.delta)/(2*h);v[mask]=0
    acc=(v-current.v)/h if previous is None else (3*v-4*current.v+previous.v)/(2*h)
    physical=mass@acc+g-force;new=P2State(u+z,v,z)
    details={'actual_method':'BE_STARTUP' if previous is None else 'BDF2','h_eff_s':heff,'iterations':it,'objective_free_R_N':norm,'physical_original_free_R_N':float(np.linalg.norm(physical[~mask])),'minimum_J':float(np.min(J)),'constraint_error_m':float(np.max(abs(new.u[mask]))) if mask.any() else 0.,'constraint_velocity_error_m_s':float(np.max(abs(v[mask]))) if mask.any() else 0.,'history':history,'elastic_energy_J':float(E)}
    return new,details


class ConsistentP2Transient:
    def __init__(self,cells=(4,2,2),order=3):
        self.model=StableP2Block(cells,young=5000.,poisson=.35,density=625.,gravity=0.,traction=1.,order=order)
        m=self.model;self.mass=scalar_mass(m)
        self.grad=np.stack([shape_gradients(b[None],m.linear_gradients) for b in m.bary]);self.check_grad=np.stack([shape_gradients(b[None],m.linear_gradients) for b in m.check_bary]);self.weights=m.weights[:,None]*m.volumes[None]
    def initial(self):
        z=np.zeros_like(self.model.rest);return P2State(z.copy(),z.copy(),z.copy())
    def elastic_u(self,u,tangent=False):
        m=self.model;ue=np.asarray(u)[m.elements]
        if u.shape!=m.rest.shape or not np.isfinite(u).all():raise ValueError('Finite displacement field required')
        F=np.eye(3)+np.einsum('eai,qeaJ->qeiJ',ue,self.grad);check=np.eye(3)+np.einsum('eai,qeaJ->qeiJ',ue,self.check_grad);J=np.r_[np.linalg.det(F).ravel(),np.linalg.det(check).ravel()]
        if not np.isfinite(J).all() or np.any(J<=m.minimum_J):raise ValueError('Positive-J guard failed; no repair')
        W,P,A=stable_material(F,m.young,m.poisson,tangent)
        ge=np.einsum('qe,qeiJ,qeaJ->eai',self.weights,P,self.grad);g=np.zeros_like(u);np.add.at(g,m.elements,ge);H=None
        if tangent:
            blocks=np.einsum('qe,qeaJ,qeiJkL,qebL->eaibk',self.weights,self.grad,A,self.grad,optimize=True)
            H=sparse.coo_matrix((blocks.ravel(),(m.rows,m.cols)),shape=(u.size,u.size)).tocsr()
        return float(np.sum(self.weights*W)),g,H,J
    def step(self,current,previous,h,force):
        return advance_consistent(self.elastic_u,self.mass,self.model.mask,current,previous,h,force,self.model.young*self.model.volumes.sum())


def force_coefficient(program,t):
    if not np.isfinite(t) or t<0:raise ValueError('Nonnegative physical time required')
    if program=='step':return 1.
    if program=='smooth_ramp':return float(np.sin((np.pi/2)*min(t/.05,1.))**2)
    raise ValueError('Declared program required')
