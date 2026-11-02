"""Stage-A compositional Q2 mechanics using the existing qualified Newton solver.

New isoparametric reference maps and pointwise torch energies compose with
bilateral Robin attachments. This module does not implement contact or dynamics.
The solver returns numerical qualification data, never an unconditional promise
of equilibrium or stability for arbitrary learned energies.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy import sparse
from scipy.spatial.transform import Rotation
from numpy.polynomial.legendre import leggauss
from scripts.phystwin.tensor_fixture import TensorFixture,basis


@dataclass(frozen=True)
class GeometrySpec:
    dimensions: tuple=(.24,.08,.08)
    taper: float=0.
    arch: float=0. # middle-axis rise in metres; endpoints unchanged

    def map(self,coordinates):
        q=np.asarray(coordinates,float);L,H,D=self.dimensions
        if q.shape[-1]!=3 or not np.isfinite(q).all() or min(self.dimensions)<=0 or not np.isfinite([*self.dimensions,self.taper,self.arch]).all() or abs(self.taper)>=.75:
            raise ValueError('Finite positive geometry and |taper|<.75 required')
        x,y,z=np.moveaxis(q,-1,0);width=1+self.taper*(2*x-1)
        return np.stack([L*x,H*(y-.5)*width,D*(z-.5)*width+4*self.arch*x*(1-x)],axis=-1)


@dataclass(frozen=True)
class AttachmentSpec:
    stiffness_N_per_m: float=6000.
    center: tuple=(.5,.1875) # material-face coordinates y,z in [0,1]
    width: tuple=(.375,.375)

    def weight(self,yz):
        center=np.asarray(self.center);width=np.asarray(self.width)
        if center.shape!=(2,) or width.shape!=(2,) or not np.isfinite(np.r_[center,width,self.stiffness_N_per_m]).all() or min(width)<=0 or self.stiffness_N_per_m<=0:
            raise ValueError('Positive finite bilateral compliance parameters required')
        return np.prod(4*yz*(1-yz),axis=-1)*np.exp(-.5*np.sum(((yz-center)/width)**2,axis=-1))


def constitutive_response(energy,F,z,*,tangent=False,create_graph=False):
    """Pointwise energy -> first Piola stress and dP_iJ/dF_kL.

    The energy MUST NOT couple batch entries (no batchnorm/pooling). Stress and
    tangent retain parameter dependence when create_graph=True. Stability is a
    separate property of the supplied architecture and tested deformation range.
    """
    with torch.enable_grad():
        F=F if F.requires_grad else F.detach().requires_grad_(True)
        if F.shape[-2:]!=(3,3) or not bool(torch.isfinite(F).all()) or bool((torch.linalg.det(F)<=0).any()):raise ValueError('Finite orientation-preserving deformation required')
        W=energy(F,z)
        if W.shape!=F.shape[:-2] or not bool(torch.isfinite(W).all()):raise ValueError('Pointwise finite energy density required')
        P=torch.autograd.grad(W.sum(),F,create_graph=tangent or create_graph,retain_graph=True)[0]
        A=None
        if tangent:
            rows=[torch.autograd.grad(P[...,i,j].sum(),F,retain_graph=True,create_graph=create_graph)[0] for i in range(3) for j in range(3)]
            A=torch.stack(rows,dim=-3).reshape(*F.shape[:-2],3,3,3,3)
        if not bool(torch.isfinite(P).all()) or (A is not None and not bool(torch.isfinite(A).all())):raise ValueError('Nonfinite constitutive derivatives')
        if not create_graph:W=W.detach();P=P.detach();A=None if A is None else A.detach()
        return W,P,A


class AnalyticNeoHookean:
    """Reference-normalized compressible log-J Neo-Hookean, Pa. z[...,0]=logE."""
    def __init__(self,poisson=.35):
        if not -1<poisson<.5:raise ValueError('Admissible Poisson ratio required')
        self.poisson=poisson
    def __call__(self,F,z):
        E=torch.exp(z[...,0]);mu=E/(2*(1+self.poisson));lam=E*self.poisson/((1+self.poisson)*(1-2*self.poisson))
        log=torch.log(torch.linalg.det(F))
        return .5*mu*((F*F).sum((-1,-2))-3)-mu*log+.5*lam*log**2


class Q2Mechanics(TensorFixture):
    def __init__(self,geometry=GeometrySpec(),*,n=2,order=3,bulk_energy=None,material_latent_field=None,
                 attachments=(AttachmentSpec(),AttachmentSpec()),gravity=(0.,0.,0.),density=625.,device='cpu',
                 reference_modulus=12000.,minimum_J=.1,linear_solver='superlu',interface_energy=None,interface_latent_fields=None,interface_geometry_parameters=None):
        if device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Only CPU/GPU0–2 allowed')
        if len(attachments)!=2 or not 0<minimum_J<1 or reference_modulus<=0:raise ValueError('Two attachments and valid reference/J bounds required')
        super().__init__(n,reference_modulus,1.,'left_full',order=order,device=device,x_power=1.)
        self.geometry=geometry;self.material_coordinates=(self.rest-np.array([0.,-.04,-.04]))/np.array([.24,.08,.08])
        self.rest=geometry.map(self.material_coordinates);self.minimum_J=minimum_J
        self.bulk_energy=bulk_energy or AnalyticNeoHookean()
        self.material_latent_field=material_latent_field or (lambda q:torch.full((*q.shape[:-1],1),np.log(reference_modulus),dtype=q.dtype,device=q.device))
        self.device=device;self.linear_solver=linear_solver;self.attachments=attachments
        self.interface_energy=interface_energy;self.interface_latent_fields=interface_latent_fields
        self.interface_geometry_parameters=interface_geometry_parameters
        if interface_geometry_parameters is not None and interface_energy is None:raise ValueError('Differentiable footprint requires explicit interface energy')
        if interface_energy is not None and (interface_latent_fields is None or len(interface_latent_fields)!=2):raise ValueError('Explicit per-interface latent fields required')
        self.gravity=np.asarray(gravity,float);self.density=float(density)
        if self.gravity.shape!=(3,) or not np.isfinite(self.gravity).all() or not np.isfinite(density) or density<=0:raise ValueError('Finite gravity and positive density required')
        self.engine=EnergyEngine(self,device)
        self.volumes=self.engine.weights.detach().cpu().numpy().sum(0)
        N,_=basis(self.quad);integral=np.einsum('qe,qa->ea',self.engine.weights.detach().cpu().numpy(),N)
        self.nodal_volumes=np.zeros(len(self.rest));np.add.at(self.nodal_volumes,self.elements,integral)
        self.total_mass=float(self.volumes.sum()*density);self.volume_weights=self.nodal_volumes/self.volumes.sum()
        self.patch_data=[]
        self.attachment_unit_matrices=[self.patch_matrix(i,spec) for i,spec in enumerate(attachments)]
        self.anchor_matrices=[spec.stiffness_N_per_m*M for spec,M in zip(attachments,self.attachment_unit_matrices)]
        self.fixture_scalar=sum(self.anchor_matrices) if interface_energy is None else sparse.csr_matrix((len(self.rest),len(self.rest)));self.spring_matrix=sparse.kron(self.fixture_scalar,sparse.eye(3),format='csr')
        self.tool_centers=geometry.map(np.array([[0,.5,.5],[1,.5,.5]]));self.set_controls(np.zeros((2,6)))

    def patch_matrix(self,side,spec):
        q,w=leggauss(6);local=np.stack(np.meshgrid(q,q,indexing='ij'),-1).reshape(-1,2)
        cells=np.stack(np.meshgrid(np.arange(self.cells[1]),np.arange(self.cells[2]),indexing='ij'),-1).reshape(-1,2)
        eid=np.ravel_multi_index(np.c_[np.full(len(cells),0 if side==0 else self.cells[0]-1),cells].T,tuple(self.cells));ids=self.elements[eid]
        N,G=basis(np.c_[np.full(len(local),-1. if side==0 else 1.),local])
        X=self.rest[ids];der=np.einsum('eai,qaj->qeij',X,G)
        area=np.linalg.norm(np.cross(der[...,1],der[...,2]),axis=-1)
        material=np.einsum('qa,eai->qei',N,self.material_coordinates[ids])[...,1:]
        weights=np.outer(w,w).ravel()[:,None]*area*spec.weight(material)
        total=weights.sum()
        if not np.isfinite(total) or total<=0:raise ValueError('Nonzero attachment footprint required')
        self.patch_data.append(dict(ids=ids,N=N,weights=weights/total,reference_area_weights=np.outer(w,w).ravel()[:,None]*area,material=np.einsum('qa,eai->qei',N,self.material_coordinates[ids])))
        M=np.einsum('qa,qe,qb->eab',N,weights/total,N)
        rows=np.broadcast_to(ids[:,:,None],M.shape);cols=np.broadcast_to(ids[:,None,:],M.shape)
        return sparse.coo_matrix((M.ravel(),(rows.ravel(),cols.ravel())),shape=(len(self.rest),len(self.rest))).tocsr()

    def set_controls(self,poses):
        poses=np.asarray(poses,float)
        if poses.shape!=(2,6) or not np.isfinite(poses).all():raise ValueError('Two finite prescribed6D tool poses required')
        self.poses=poses.copy();self.anchor_targets=[];self.target_constant=0.
        self.load=self.density*self.nodal_volumes[:,None]*self.gravity
        for center,pose,M in zip(self.tool_centers,poses,self.anchor_matrices):
            R=Rotation.from_rotvec(pose[3:]).as_matrix();d=(self.rest-center)@R.T+center+pose[:3]-self.rest
            self.anchor_targets.append(d)
            if self.interface_energy is None:self.load+=M@d;self.target_constant+=.5*np.sum(d*(M@d))
        self.clear_linear_cache()

    def evaluate(self,u,tangent=False):
        state=super().evaluate(u,tangent)
        state['potential_J']+=self.target_constant
        if self.interface_energy is None:
            state['spring_J']=float(sum(.5*np.sum((u-d)*(M@(u-d))) for M,d in zip(self.anchor_matrices,self.anchor_targets)))
        else:
            U,r,H,_=self.interface_response(self.engine.tensor(u),tangent=tangent)
            state['spring_J']=float(U.detach());state['potential_J']+=state['spring_J'];state['residual']+=r.detach().cpu().numpy()
            if tangent:state['tangent']+=H
        if not np.isfinite(state['potential_J']) or not np.isfinite(state['residual']).all():raise ValueError('Nonfinite equilibrium state')
        return state

    def query_field(self,*args,**kwargs):
        raise ValueError('Mapped geometry requires sample_material with explicit unit-cube material coordinates')

    def sample_material(self,u,coordinates,gradient=False):
        q=np.asarray(coordinates,float)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('Material query points must lie in unit cube')
        old=q*np.array([.24,.08,.08])+np.array([0.,-.04,-.04])
        cell=np.stack([np.clip(np.searchsorted(a,old[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(self.axes)],axis=1)
        lo=np.stack([a[cell[:,i]] for i,a in enumerate(self.axes)],1);width=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(self.axes)],1)
        local=2*(old-lo)/width-1;N,G=basis(local);ids=self.elements[np.ravel_multi_index(cell.T,tuple(self.cells))]
        if gradient:
            jac=np.einsum('pai,paJ->piJ',self.rest[ids],G)
            physical=np.einsum('paJ,pJi->pai',G,np.linalg.inv(jac))
            return np.einsum('pai,paJ->piJ',np.asarray(u)[ids],physical)
        return np.einsum('pa,pai->pi',N,np.asarray(u)[ids])

    def reaction_wrenches(self,state):
        u=state['displacement_local_m'];forces=[];torques=[];origins=self.tool_centers+self.poses[:,:3]
        if self.interface_energy is not None:
            _,_,_,wrenches=self.interface_response(self.engine.tensor(u))
            forces,torques=wrenches
            forces=np.asarray([f.detach().cpu().numpy() for f in forces]);torques=np.asarray([t.detach().cpu().numpy() for t in torques])
            return dict(forces_N=forces,torques_Nm=torques,origins_world_m=origins,force_balance_N=float(np.linalg.norm(forces.sum(0)-self.total_mass*self.gravity)))
        for M,d,origin in zip(self.anchor_matrices,self.anchor_targets,origins):
            force=M@(u-d);forces.append(force.sum(0));torques.append(np.cross(self.rest+d-origin,force).sum(0))
        return dict(forces_N=np.asarray(forces),torques_Nm=np.asarray(torques),origins_world_m=origins,
                    force_balance_N=float(np.linalg.norm(np.sum(forces,axis=0)-self.total_mass*self.gravity)))

    def torch_residual(self,u,attachment_stiffness=None):
        """Differentiable residual for implicit law/latent/support gradients."""
        r=self.engine.residual_tensor(u,create_graph=True)
        if self.interface_energy is not None:
            if attachment_stiffness is not None:raise ValueError('Use interface latent fields for nonlinear interface parameters')
            _,ir,_,_=self.interface_response(u,create_graph=True)
            return r+ir-torch.as_tensor(self.density*self.nodal_volumes[:,None]*self.gravity,dtype=u.dtype,device=u.device)
        stiffness=attachment_stiffness if attachment_stiffness is not None else torch.tensor([s.stiffness_N_per_m for s in self.attachments],dtype=u.dtype,device=u.device)
        for K,M,d in zip(stiffness,self.attachment_unit_matrices,self.anchor_targets):
            coo=M.tocoo();idx=torch.tensor(np.stack([coo.row,coo.col]),device=u.device)
            matrix=torch.sparse_coo_tensor(idx,torch.as_tensor(coo.data,dtype=u.dtype,device=u.device),M.shape,check_invariants=True)
            r=r+K*torch.sparse.mm(matrix,u-torch.as_tensor(d,dtype=u.dtype,device=u.device))
        return r-torch.as_tensor(self.density*self.nodal_volumes[:,None]*self.gravity,dtype=u.dtype,device=u.device)


    def interface_response(self,u,*,tangent=False,create_graph=False):
        """Normalized reference-area quadrature of bilateral shared interface law.

        Interface energy(delta,z) returns J at each normalized patch point;
        integration weights sum to one per tool. Delta is world relative motion.
        """
        total=u.new_zeros(());residual=torch.zeros_like(u);forces=[];torques=[];blocks=[];row_ids=[];col_ids=[]
        for side,data in enumerate(self.patch_data):
            ids=torch.as_tensor(data['ids'],dtype=torch.long,device=u.device);N=self.engine.tensor(data['N']);weights=self.engine.tensor(data['weights'])
            target=self.engine.tensor(self.anchor_targets[side]);material=self.engine.tensor(data['material'])
            if self.interface_geometry_parameters is not None:
                parameters=self.interface_geometry_parameters() if callable(self.interface_geometry_parameters) else self.interface_geometry_parameters
                parameters=torch.as_tensor(parameters,dtype=u.dtype,device=u.device)
                if parameters.shape!=(2,4) or not bool(torch.isfinite(parameters).all()):raise ValueError('Two finite [center2,logwidth2] interface geometry rows required')
                center=parameters[side,:2];width=torch.exp(parameters[side,2:])
                if bool(((center<0)|(center>1)).any()) or not bool(torch.isfinite(width).all()) or bool((width<=0).any()):raise ValueError('Interface centers in unitface and positive finite widths required')
                yz=material[...,1:];profile=(4*yz*(1-yz)).prod(-1)*torch.exp(-.5*(((yz-center)/width)**2).sum(-1))
                raw=self.engine.tensor(data['reference_area_weights'])*profile;normalizer=raw.sum()
                if not bool(torch.isfinite(normalizer)) or float(normalizer.detach())<=0:raise ValueError('Nonzero finite differentiable footprint required')
                weights=raw/normalizer
            delta=torch.einsum('qa,eai->qei',N,(u-target)[ids]);delta=delta if delta.requires_grad else delta.detach().requires_grad_(True)
            z=self.interface_latent_fields[side](material)
            with torch.enable_grad():
                W=self.interface_energy(delta,z)
                if W.shape!=delta.shape[:-1] or not bool(torch.isfinite(W).all()):raise ValueError('Finite pointwise interface energy required')
                traction=torch.autograd.grad(W.sum(),delta,create_graph=create_graph or tangent,retain_graph=True)[0]
                if not bool(torch.isfinite(traction).all()):raise ValueError('Nonfinite interface traction')
                if tangent:
                    C=torch.stack([torch.autograd.grad(traction[...,i].sum(),delta,retain_graph=True,create_graph=create_graph)[0] for i in range(3)],dim=-2)
                    local=torch.einsum('qe,qeik,qa,qb->eaibk',weights,C,N,N)
                    if not bool(torch.isfinite(local).all()):raise ValueError('Nonfinite interface tangent')
                    dofs=(data['ids'][:,:,None]*3+np.arange(3)).reshape(-1,81)
                    blocks.append(local.detach().cpu().numpy().ravel());row_ids.append(np.repeat(dofs,81,1).ravel());col_ids.append(np.tile(dofs,(1,81)).ravel())
                if not create_graph:W=W.detach();traction=traction.detach()
            total=total+(weights*W).sum();local_r=torch.einsum('qe,qei,qa->eai',weights,traction,N)
            residual=residual.index_add(0,ids.reshape(-1),local_r.reshape(-1,3))
            force=torch.einsum('qe,qei->i',weights,traction)
            locations=torch.einsum('qa,eai->qei',N,(self.engine.rest+target)[ids]);origin=self.engine.tensor(self.tool_centers[side]+self.poses[side,:3])
            torque=torch.einsum('qe,qei->i',weights,torch.linalg.cross(locations-origin,traction,dim=-1))
            forces.append(force);torques.append(torque)
        H=None if not tangent else sparse.coo_matrix((np.concatenate(blocks),(np.concatenate(row_ids),np.concatenate(col_ids))),shape=(self.rest.size,self.rest.size)).tocsr()
        return total,residual,H,(forces,torques)


class EnergyEngine:
    def __init__(self,model,device):
        self.model=model;self.device=device;self.dtype=torch.float64
        self.rest=self.tensor(model.rest);self.elements=torch.as_tensor(model.elements,dtype=torch.long,device=device);self.I=torch.eye(3,dtype=self.dtype,device=device)
        N,G=basis(model.quad);_,checkG=basis(model.check)
        def transform(g):
            jac=np.einsum('eai,qaj->qeij',model.rest[model.elements],g);det=np.linalg.det(jac)
            if not np.isfinite(det).all() or (det<=0).any():raise ValueError('Invalid reference geometry Jacobian')
            return np.einsum('qaJ,qeJi->qeai',g,np.linalg.inv(jac)),det
        g,det=transform(G);cg,_=transform(checkG);self.grad=self.tensor(g);self.check_grad=self.tensor(cg)
        self.weights=self.tensor(model.weights[:,None]*8*det);self.reference_points=self.tensor(np.einsum('qa,eai->qei',N,model.material_coordinates[model.elements]))
        self.chunk=4
    def tensor(self,value):return torch.as_tensor(value,dtype=self.dtype,device=self.device)
    def residual_tensor(self,u,create_graph=False):
        ue=u[self.elements];F=self.I+torch.einsum('eai,qeaj->qeij',ue,self.grad)
        z=self.model.material_latent_field(self.reference_points);_,P,_=constitutive_response(self.model.bulk_energy,F,z,create_graph=create_graph)
        ge=torch.einsum('qe,qeiJ,qeaJ->eai',self.weights,P,self.grad)
        return torch.zeros_like(u).index_add(0,self.elements.reshape(-1),ge.reshape(-1,3))
    def elastic(self,positions,tangent=False):
        m=self.model;ue=(self.tensor(positions)-self.rest)[self.elements]
        Jcheck=torch.linalg.det(self.I+torch.einsum('eai,qeaj->qeij',ue,self.check_grad));Jquad=torch.linalg.det(self.I+torch.einsum('eai,qeaj->qeij',ue,self.grad));J=torch.cat([Jcheck.reshape(-1),Jquad.reshape(-1)])
        if not bool(torch.isfinite(J).all()) or float(J.min())<=m.minimum_J:raise ValueError('Nodal/quadrature Jacobian guard')
        z=m.material_latent_field(self.reference_points);energy=0.;ge=torch.zeros_like(ue)
        H=torch.zeros((len(ue),27,3,27,3),dtype=self.dtype,device=self.device) if tangent else None
        for start in range(0,len(self.grad),self.chunk):
            q=slice(start,start+self.chunk);g=self.grad[q];F=self.I+torch.einsum('eai,qeaj->qeij',ue,g)
            W,P,A=constitutive_response(m.bulk_energy,F,z[q],tangent=tangent)
            energy+=float((self.weights[q]*W).sum());ge+=torch.einsum('qe,qeiJ,qeaJ->eai',self.weights[q],P,g)
            if tangent:H+=torch.einsum('qe,qeiJkL,qeaJ,qebL->eaibk',self.weights[q],A,g,g)
        grad=torch.zeros_like(self.rest).index_add(0,self.elements.reshape(-1),ge.reshape(-1,3))
        matrix=None if H is None else sparse.coo_matrix((H.cpu().numpy().ravel(),(m.rows,m.cols)),shape=(m.rest.size,m.rest.size)).tocsr()
        return energy,grad.cpu().numpy(),matrix,J.cpu().numpy()
