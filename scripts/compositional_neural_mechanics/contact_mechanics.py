"""Stage B: separate unilateral normal penalty contact for mapped Q2 bodies.

Known rigid planes/spheres, prescribed geometry, no friction/adhesion/dynamics.
Energy integrates over the actual mapped REFERENCE boundary area. The current
surface is evaluated at material quadrature points. Finite penalty permits
penetration; sampled maximum penetration is not a bound over the full surface.
The tangent is piecewise smooth and undefined classically exactly at gap zero.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy import sparse
from numpy.polynomial.legendre import leggauss
from scripts.phystwin.tensor_fixture import basis
from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics


def vector(value,like):
    x=torch.as_tensor(value,dtype=like.dtype,device=like.device)
    if x.shape!=(3,) or not bool(torch.isfinite(x).all()):raise ValueError('Finite world3-vector required')
    return x


@dataclass(frozen=True)
class RigidPlane:
    point: object=(0.,0.,0.)
    normal: object=(0.,0.,1.)
    penalty_N_m3: float=1e7
    name: str='plane'

    def origin(self,x):return vector(self.point,x)
    def signed_distance(self,x):
        n=vector(self.normal,x);norm=torch.linalg.vector_norm(n)
        if float(norm.detach())<=1e-12:raise ValueError('Nonzero plane normal required')
        return ((x-self.origin(x))*(n/norm)).sum(-1)
    def transformed(self,R,t):
        R=np.asarray(R,float);t=np.asarray(t,float)
        return RigidPlane(R@np.asarray(self.point)+t,R@np.asarray(self.normal),self.penalty_N_m3,self.name)


@dataclass(frozen=True)
class RigidSphere:
    center: object=(0.,0.,0.)
    radius: float=.03
    penalty_N_m3: float=1e7
    name: str='sphere'

    def origin(self,x):return vector(self.center,x)
    def signed_distance(self,x):
        if not np.isfinite(self.radius) or self.radius<=0:raise ValueError('Positive finite sphere radius required')
        distance=torch.linalg.vector_norm(x-self.origin(x),dim=-1)
        if bool((distance<=1e-12).any()):raise ValueError('Sphere-center distance singularity is not a valid contact sample')
        return distance-self.radius
    def transformed(self,R,t):return RigidSphere(np.asarray(R)@np.asarray(self.center)+np.asarray(t),self.radius,self.penalty_N_m3,self.name)


def normal_contact_response(collider,positions,*,tangent=False,create_graph=False):
    """Return energy/area, dE/dx, d²E/dx² and signed gap pointwise.

    dE/dx is force density on the rigid collider; the soft body receives its
    negative. No normal tensile force appears on separated samples.
    """
    if not np.isfinite(collider.penalty_N_m3) or collider.penalty_N_m3<=0:raise ValueError('Positive numerical penalty density required')
    with torch.enable_grad():
        x=positions if positions.requires_grad else positions.detach().requires_grad_(True)
        if x.shape[-1]!=3 or not bool(torch.isfinite(x).all()):raise ValueError('Finite contact positions required')
        gap=collider.signed_distance(x);penetration=torch.clamp_max(gap,0.)
        W=.5*collider.penalty_N_m3*penetration.square()
        traction=torch.autograd.grad(W.sum(),x,create_graph=create_graph or tangent,retain_graph=True)[0]
        C=None
        if tangent:C=torch.stack([torch.autograd.grad(traction[...,i].sum(),x,create_graph=create_graph,retain_graph=True)[0] for i in range(3)],dim=-2)
        if not bool(torch.isfinite(W).all()) or not bool(torch.isfinite(traction).all()) or (C is not None and not bool(torch.isfinite(C).all())):raise ValueError('Nonfinite contact response')
        if not create_graph:W=W.detach();traction=traction.detach();gap=gap.detach();C=None if C is None else C.detach()
        return W,traction,C,gap


class ContactQ2Mechanics(Q2Mechanics):
    """Compose contact with existing bulk, bilateral attachments and gravity."""
    def __init__(self,*args,colliders=(),surface_order=4,**kwargs):
        if surface_order not in (3,4,6,8,16):raise ValueError('Explicit supported surface quadrature order required')
        super().__init__(*args,**kwargs)
        self.colliders=tuple(colliders);self.surface_order=surface_order
        if any(not isinstance(c,(RigidPlane,RigidSphere)) for c in self.colliders):raise ValueError('Only explicit known plane/sphere contact is supported')
        if len({c.name for c in self.colliders})!=len(self.colliders):raise ValueError('Distinct collider names required')
        self.contact_patches=self._surface_quadrature(surface_order)

    def _surface_quadrature(self,order):
        q,w=leggauss(order);local2=np.stack(np.meshgrid(q,q,indexing='ij'),-1).reshape(-1,2);qw=np.outer(w,w).ravel();patches=[]
        for axis in range(3):
            others=[k for k in range(3) if k!=axis]
            for side in (0,1):
                cells=np.stack(np.meshgrid(*[np.arange(self.cells[k]) for k in others],indexing='ij'),-1).reshape(-1,2)
                ijk=np.zeros((len(cells),3),dtype=int);ijk[:,axis]=0 if side==0 else self.cells[axis]-1;ijk[:,others]=cells
                ids=self.elements[np.ravel_multi_index(ijk.T,tuple(self.cells))]
                points=np.zeros((len(local2),3));points[:,axis]=-1 if side==0 else 1;points[:,others]=local2;N,G=basis(points)
                jac=np.einsum('eai,qaj->qeij',self.rest[ids],G)
                area=np.linalg.norm(np.cross(jac[...,others[0]],jac[...,others[1]]),axis=-1)
                weights=qw[:,None]*area
                if not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Positive finite reference surface measure required')
                patches.append(dict(ids=ids,N=N,weights=weights,axis=axis,side=side))
        return patches

    def contact_response(self,u,*,tangent=False,create_graph=False):
        energy=u.new_zeros(());residual=torch.zeros_like(u);blocks=[];rows=[];cols=[];records=[]
        for collider in self.colliders:
            force=u.new_zeros(3);torque=u.new_zeros(3);collider_energy=u.new_zeros(());active=0;max_penetration=0.;minimum_gap=float('inf')
            for patch in self.contact_patches:
                ids=torch.as_tensor(patch['ids'],dtype=torch.long,device=u.device);N=self.engine.tensor(patch['N']);weights=self.engine.tensor(patch['weights'])
                x=torch.einsum('qa,eai->qei',N,(self.engine.rest+u)[ids])
                W,p,C,gap=normal_contact_response(collider,x,tangent=tangent,create_graph=create_graph)
                energy=energy+(weights*W).sum();collider_energy=collider_energy+(weights*W).sum()
                local=torch.einsum('qe,qei,qa->eai',weights,p,N);residual=residual.index_add(0,ids.reshape(-1),local.reshape(-1,3))
                force=force+torch.einsum('qe,qei->i',weights,p)
                torque=torque+torch.einsum('qe,qei->i',weights,torch.linalg.cross(x-collider.origin(x),p,dim=-1))
                active+=int((gap.detach()<0).sum());minimum_gap=min(minimum_gap,float(gap.detach().min()));max_penetration=max(max_penetration,float(torch.clamp_min(-gap.detach(),0).max()))
                if tangent:
                    h=torch.einsum('qe,qeik,qa,qb->eaibk',weights,C,N,N);dofs=(patch['ids'][:,:,None]*3+np.arange(3)).reshape(-1,81)
                    blocks.append(h.detach().cpu().numpy().ravel());rows.append(np.repeat(dofs,81,1).ravel());cols.append(np.tile(dofs,(1,81)).ravel())
            records.append(dict(name=collider.name,mode='unilateral_normal_penalty',force_on_rigid_N=force,torque_on_rigid_Nm=torque,origin_world_m=collider.origin(u),
                                contact_energy_J=collider_energy,active_quadrature_samples=active,minimum_sampled_gap_m=minimum_gap,max_sampled_penetration_m=max_penetration,penalty_N_m3=collider.penalty_N_m3))
        H=None
        if tangent:
            H=sparse.csr_matrix((self.rest.size,self.rest.size)) if not blocks else sparse.coo_matrix((np.concatenate(blocks),(np.concatenate(rows),np.concatenate(cols))),shape=(self.rest.size,self.rest.size)).tocsr()
        return energy,residual,H,records

    def evaluate(self,u,tangent=False):
        state=super().evaluate(u,tangent)
        energy,r,H,records=self.contact_response(self.engine.tensor(u),tangent=tangent)
        state['potential_J']+=float(energy.detach());state['residual']+=r.detach().cpu().numpy()
        if tangent:state['tangent']+=H
        state['contact_J']=float(energy.detach());state['max_sampled_penetration_m']=max([r['max_sampled_penetration_m'] for r in records],default=0.)
        state['contact_active_samples']=sum(r['active_quadrature_samples'] for r in records)
        return state

    def torch_residual(self,u,attachment_stiffness=None):
        r=super().torch_residual(u,attachment_stiffness);_,contact,_,_=self.contact_response(u,create_graph=True)
        return r+contact

    def reaction_wrenches(self,state):
        attachment=super().reaction_wrenches(state);u=state['displacement_local_m'];_,_,_,records=self.contact_response(self.engine.tensor(u))
        contacts=[]
        for row in records:
            contacts.append({key:value.detach().cpu().numpy() if torch.is_tensor(value) and value.ndim else float(value.detach()) if torch.is_tensor(value) else value for key,value in row.items()})
        forces=attachment['forces_N'].sum(0)+sum((r['force_on_rigid_N'] for r in contacts),np.zeros(3))
        torque=(attachment['torques_Nm']+np.cross(attachment['origins_world_m'],attachment['forces_N'])).sum(0)
        torque+=sum((r['torque_on_rigid_Nm']+np.cross(r['origin_world_m'],r['force_on_rigid_N']) for r in contacts),np.zeros(3))
        body_force=self.density*self.nodal_volumes[:,None]*self.gravity
        body_torque=np.cross(self.rest+u,body_force).sum(0)
        return dict(**{k:v for k,v in attachment.items() if k!='force_balance_N'},contact_reactions=contacts,
                    force_balance_N=float(np.linalg.norm(forces-body_force.sum(0))),world_torque_balance_Nm=float(np.linalg.norm(torque-body_torque)),
                    measure='Reference area of mapped boundary; contact positions deform with material',
                    limitation='Sampled finite-penalty penetration; no friction, impact, exact nonpenetration or freely moving rigid dynamics')


def dense_surface_gap_check(model,u,points_per_axis=65):
    """Independent material-face grid, not contact integration samples.

    Includes face edges/centers; flags known missed contact but is not a certified
    global minimum, collision detector, or geometric nonpenetration guarantee.
    """
    if points_per_axis<3 or points_per_axis%2!=1:raise ValueError('Odd dense grid >=3 required')
    axis=np.linspace(0.,1.,points_per_axis)
    uv=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    faces=[]
    for a in range(3):
        for side in (0.,1.):
            q=np.empty((len(uv),3));q[:,a]=side;q[:,[i for i in range(3) if i!=a]]=uv;faces.append(q)
    q=np.unique(np.concatenate(faces),axis=0)
    x=model.geometry.map(q)+model.sample_material(u,q)
    _,_,_,integration=model.contact_response(model.engine.tensor(u))
    records=[]
    for collider,integrated in zip(model.colliders,integration):
        gap=collider.signed_distance(model.engine.tensor(x)).detach().cpu().numpy()
        if not np.isfinite(gap).all():raise ValueError('Nonfinite independent gaps')
        i=int(np.argmin(gap));depth=max(0.,-float(gap[i]))
        records.append(dict(name=collider.name,dense_sample_count=len(q),minimum_dense_gap_m=float(gap[i]),
                            max_dense_penetration_m=depth,minimum_material_coordinate=q[i].tolist(),
                            missed_contact=bool(depth>1e-10 and integrated['active_quadrature_samples']==0),
                            dense_minus_quadrature_depth_m=depth-integrated['max_sampled_penetration_m']))
    return records
