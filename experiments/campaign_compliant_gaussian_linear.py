"""Implicit small-strain mechanics with explicit compliant Gaussian fixtures.

New boundary hypothesis only: no hard-mask replacement, contact, inference,
native/continuum claim, added Gaussian masses or pinned-source modification.
Material patches attach to X; constant K/D and prescribed targets are world-frame.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
try:
    import campaign_boundary_gaussians as boundary
    from campaign_material_gaussians import MaterialGaussians
    from campaign_linear_fem import assemble_tetrahedra
except ImportError:
    from experiments import campaign_boundary_gaussians as boundary
    from experiments.campaign_material_gaussians import MaterialGaussians
    from experiments.campaign_linear_fem import assemble_tetrahedra


@dataclass(frozen=True)
class GaussianFixtureMatrix:
    weights:torch.Tensor
    stiffness:torch.Tensor
    damping:torch.Tensor
    K:torch.Tensor
    C:torch.Tensor

    @classmethod
    def from_patches(cls,quadrature,patches,stiffness_world_N_m,damping_world_Ns_m,*,device='cpu'):
        bank=boundary.patch_bank(quadrature,patches,2,{'candidate_boundary_hypothesis'})
        K=np.stack([boundary.psd(k,'fixture stiffness') for k in stiffness_world_N_m])
        D=np.stack([boundary.psd(d,'fixture damping') for d in damping_world_Ns_m])
        if K.shape!=(2,3,3) or D.shape!=(2,3,3):raise ValueError('Exactly two explicit K/D tensors required')
        values=[torch.tensor(v,dtype=torch.float64,device=device) for v in [bank['normalized_node_weights'],K,D]]
        w,k,d=values
        return cls(w,k,d,torch.block_diag(*torch.einsum('sn,sij->nij',w,k)),torch.block_diag(*torch.einsum('sn,sij->nij',w,d)))

    def response(self,u,v,target,target_velocity):
        shape=(self.weights.shape[1],3)
        if (u.shape!=shape or v.shape!=shape or target.shape!=(2,*shape) or target_velocity.shape!=target.shape
                or any(x.dtype!=torch.float64 or x.device!=self.weights.device or not bool(torch.isfinite(x).all()) for x in [u,v,target,target_velocity])):
            raise ValueError('Finite float64 matching-device displacement/target fields required')
        e=u[None]-target;relative=v[None]-target_velocity
        elastic=-self.weights[:,:,None]*torch.einsum('sij,snj->sni',self.stiffness,e)
        damping=-self.weights[:,:,None]*torch.einsum('sij,snj->sni',self.damping,relative)
        forces=elastic+damping
        return dict(forces_each=forces,forces=forces.sum(0),energy=-.5*(elastic*e).sum(),
            dissipation=-(damping*relative).sum(),target_power=(forces*target_velocity).sum())

    def target_force(self,target,target_velocity):
        return torch.einsum('sn,sij,snj->ni',self.weights,self.stiffness,target)+torch.einsum('sn,sij,snj->ni',self.weights,self.damping,target_velocity)


class LinearCompliantSystem:
    """All nodal DOFs free. Dynamic M is positive; no stiffness repair or pins."""
    def __init__(self,elastic_K,mass_diagonal,fixture):
        K=elastic_K;m=mass_diagonal
        if (K.dtype!=torch.float64 or m.dtype!=torch.float64 or K.device!=fixture.K.device or m.device!=K.device
                or K.shape!=fixture.K.shape or m.shape!=(len(K),) or not bool(torch.isfinite(K).all())
                or not bool(torch.isfinite(m).all()) or bool((m<=0).any()) or not torch.allclose(K,K.T,rtol=0,atol=1e-12)):
            raise ValueError('Finite symmetric elastic K and positive diagonal mass required')
        self.K,self.mass,self.fixture=K,m,fixture;self.factors={}

    def energy(self,u,v,target):
        elastic=.5*u.flatten()@(self.K@u.flatten())
        support=self.fixture.response(u,v,target,torch.zeros_like(target))['energy']
        kinetic=.5*(self.mass*v.flatten().square()).sum()
        return elastic+support+kinetic

    def step(self,u,v,target_old,target_new,force,h):
        """Endpoint held load; target velocity is its explicit backward chord."""
        if not np.isfinite(h) or h<=0:raise ValueError('Positive finite dt required')
        target_velocity=(target_new-target_old)/h
        self.fixture.response(u,v,target_new,target_velocity) # shape/finite checks
        if force.shape!=u.shape or force.dtype!=torch.float64 or force.device!=u.device or not bool(torch.isfinite(force).all()):raise ValueError('Explicit finite nodal force required')
        if h not in self.factors:
            matrix=torch.diag(self.mass/(h*h))+self.fixture.C/h+self.K+self.fixture.K
            factor,info=torch.linalg.cholesky_ex(matrix)
            if int(info)!=0:raise ValueError('Dynamic matrix not SPD; no regularizer')
            self.factors[h]=factor
        rhs=force.flatten()+self.mass*(u.flatten()/(h*h)+v.flatten()/h)
        rhs+=self.fixture.C@u.flatten()/h+self.fixture.target_force(target_new,target_velocity).flatten()
        new_u=torch.cholesky_solve(rhs[:,None],self.factors[h])[:,0].reshape_as(u)
        dx=new_u-u;new_v=dx/h;dv=new_v-v
        response=self.fixture.response(new_u,new_v,target_new,target_velocity)
        residual=self.mass*(dv/h).flatten()+self.K@new_u.flatten()-response['forces'].flatten()-force.flatten()
        de=dx[None]-(target_new-target_old)
        numerical=.5*(self.mass*dv.flatten().square()).sum()+.5*dx.flatten()@(self.K@dx.flatten())
        numerical+=.5*torch.einsum('sn,sni,sij,snj->',self.fixture.weights,de,self.fixture.stiffness,de)
        work_external=(force*dx).sum();work_target=(response['forces_each']*(target_new-target_old)).sum()
        delta_energy=self.energy(new_u,new_v,target_new)-self.energy(u,v,target_old)
        balance=delta_energy-work_external-work_target+h*response['dissipation']+numerical
        ledger=dict(force_residual_N=float(torch.linalg.vector_norm(residual)),
            delta_energy_J=float(delta_energy),external_work_J=float(work_external),target_work_J=float(work_target),
            damping_dissipation_J=float(h*response['dissipation']),implicit_euler_numerical_dissipation_J=float(numerical),
            balance_error_J=float(balance),equilibrium_residual_work_J=float(residual@dx.flatten()))
        return new_u,new_v,ledger


def tiny_geometry():
    X=np.array([[0.,0.,0.],[.1,0,0],[0,.1,0],[0,0,.1],[.1,.1,.1]])
    # Declare positive rest winding for both elements; never repair a solved F.
    tets=np.array([[0,1,2,3],[4,2,1,3]],dtype=np.int64)
    faces=np.array([[0,1,2],[0,1,3],[0,2,3],[4,1,2],[4,1,3],[4,2,3]],dtype=np.int64)
    return X,tets,faces


def candidate_patches(Q=np.eye(3),shift=np.zeros(3)):
    centers=np.array([[.01,.02,.01],[.095,.09,.085]])@Q.T+shift
    cov=Q@np.diag(np.array([.035,.045,.03])**2)@Q.T
    patches=[boundary.GaussianPatch(c,cov,'candidate_boundary_hypothesis') for c in centers]
    n=np.array([1.,2.,-1.])/np.sqrt(6)
    K=np.stack([np.diag([20.,35.,50.]),10*np.outer(n,n)])
    D=np.stack([np.diag([.1,.15,.2]),.05*np.outer(n,n)])
    return patches,Q@K@Q.T,Q@D@Q.T


def independent_elastic_force(X,tets,u,E=5000.,nu=.35):
    """Stress/shape-function assembly independent of B^T D B under test."""
    mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));gradient=np.zeros_like(X)
    for tet in tets:
        corners=X[tet];g=np.linalg.inv(np.column_stack([np.ones(4),corners]))[1:,:].T
        volume=abs(np.linalg.det((corners[1:]-corners[0]).T))/6
        du=u[tet].T@g;strain=.5*(du+du.T)
        stress=2*mu*strain+lam*np.trace(strain)*np.eye(3)
        gradient[tet]+=volume*(g@stress.T)
    return gradient


def manufactured(hz=120,*,device='cpu',Q=None,shift=None):
    """Rate-specific discrete manufactured loads: consistency, NOT convergence."""
    Q=np.eye(3) if Q is None else Q;shift=np.zeros(3) if shift is None else shift
    X0,tets,faces=tiny_geometry();X=X0@Q.T+shift
    quadrature=boundary.SurfaceQuadrature.from_mesh(X,faces);patches,Ks,Ds=candidate_patches(Q,shift)
    fixture=GaussianFixtureMatrix.from_patches(quadrature,patches,Ks,Ds,device=device)
    K,mass,_,_=assemble_tetrahedra(X,tets,5000.,.35,625.)
    model=LinearCompliantSystem(K.to(device),mass.to(device),fixture)
    as_tensor=lambda a:torch.tensor(a,dtype=torch.float64,device=device)
    h=1/hz;times=np.arange(round(.25*hz)+1)/hz;s=np.sin(np.pi*times/.25)**2
    B=np.array([[.02,.03,0],[0,-.01,.01],[0,0,.015]])
    exact=(s[:,None,None]*(X0@B.T+.002*np.array([1.,-.5,.25]))[None])@Q.T
    target=s[:,None,None,None]*np.array([[.0007,.00014,-.00021],[-.00004,-.0004,-.00008]])[None,:,None,:]
    target=np.broadcast_to(target,(len(times),2,len(X),3)).copy()@Q.T
    velocities=np.zeros_like(exact);velocities[1:]=np.diff(exact,axis=0)/h
    # Independent masses from simplex volumes, not the assembled model vector.
    masses=np.zeros(len(X))
    for tet in tets:masses[tet]+=625*abs(np.linalg.det((X[tet[1:]]-X[tet[0]]).T))/24
    u=as_tensor(exact[0]);v=as_tensor(velocities[0]);states=[u.clone()];ledgers=[];loads=[]
    for k in range(1,len(times)):
        tv=(target[k]-target[k-1])/h
        spring=boundary.compliant_support_response(quadrature,patches,positions_world_m=X+exact[k],
            velocities_world_m_s=velocities[k],target_positions_world_m=X[None]+target[k],target_velocities_world_m_s=tv,
            stiffness_world_N_m=Ks,damping_world_Ns_m=Ds)
        force=masses[:,None]*(velocities[k]-velocities[k-1])/h+independent_elastic_force(X,tets,exact[k])-spring['forces_world_N']
        u,v,row=model.step(u,v,as_tensor(target[k-1]),as_tensor(target[k]),as_tensor(force),h)
        row['position_error_m']=float((u-as_tensor(exact[k])).abs().max())
        row['velocity_error_m_s']=float((v-as_tensor(velocities[k])).abs().max())
        states.append(u.clone());ledgers.append(row);loads.append(force)
    positions=as_tensor(X)+torch.stack(states)
    kernels=MaterialGaussians.from_tetrahedra(as_tensor(X),torch.tensor(tets,device=device),625.)
    transported=kernels.transport(positions)
    expectedF=np.array([Q@(np.eye(3)+a*B)@Q.T for a in s])
    expected_cov=as_tensor(expectedF)[:,None]@kernels.reference_covariances@as_tensor(expectedF).transpose(-1,-2)[:,None]
    summary=dict(hz=hz,steps=len(times)-1,device=device,
        maximum_position_error_m=max(r['position_error_m'] for r in ledgers),
        maximum_velocity_error_m_s=max(r['velocity_error_m_s'] for r in ledgers),
        maximum_force_residual_N=max(r['force_residual_N'] for r in ledgers),
        maximum_energy_balance_error_J=max(abs(r['balance_error_J']) for r in ledgers),
        maximum_balance_minus_residual_work_J=max(abs(r['balance_error_J']-r['equilibrium_residual_work_J']) for r in ledgers),
        total_physical_dissipation_J=sum(r['damping_dissipation_J'] for r in ledgers),
        total_numerical_dissipation_J=sum(r['implicit_euler_numerical_dissipation_J'] for r in ledgers),
        total_external_work_J=sum(r['external_work_J'] for r in ledgers),total_target_work_J=sum(r['target_work_J'] for r in ledgers),
        minimum_J=float(transported.jacobians.min()),
        maximum_F_error=float((transported.deformation_gradients-as_tensor(expectedF)[:,None]).abs().max()),
        maximum_covariance_error_m2=float((transported.covariances-expected_cov).abs().max()),
        maximum_mass_conservation_error_kg=float((transported.volumes*transported.bulk_density-kernels.masses).abs().max()),
        maximum_weight_normalization_error=float((fixture.weights.sum(1)-1).abs().max()),
        material_mass_kg=float(kernels.total_mass),all_DOFs_free=True,hard_constraints_replaced=False,
        independent_forces='Linear Cauchy stress + simplex shape-function gradients + NumPy analytic fixture law + independent lumped masses',
        time_refinement_claim=False)
    return summary,positions.detach().cpu().numpy(),np.asarray(loads),fixture


def run(output,gpu_device=None):
    torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    pins={p:sha(root/p) for p in ['experiments/campaign_compliant_gaussian_linear.py','experiments/campaign_boundary_gaussians.py',
        'experiments/campaign_material_gaussians.py','experiments/campaign_linear_fem.py']}
    protocol=dict(status='FROZEN_BEFORE_MANUFACTURED_SOLVES',source_sha256=pins,rates_Hz=[120,240,480],CPU_threads=2,
        optional_GPU_device=gpu_device,mesh='two tetrahedra/5nodes; no rigid hardware/contact/hardconstraints',
        scope='Rate-specific exact discrete manufactured small-strain consistency; not nonlinear temporal convergence, calibration, learned performance, native or continuum equivalence',
        proposed_parameters='E5000Pa,nu.35,rho625;2materialGaussian supports, fullPSD+rank1K/D; prescribed affine target/loads in source')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');started=time.monotonic()
    try:
        rows=[];positions={}
        for hz in protocol['rates_Hz']:
            row,q,_,_=manufactured(hz);rows.append(row);positions[hz]=q
        angle=.63;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        shifted,q,_,_=manufactured(120,Q=Q,shift=np.array([.3,-.2,.7]))
        frame_error=float(abs(q-(positions[120]@Q.T+[.3,-.2,.7])).max())
        gpu=None
        if gpu_device:
            gpu,q,_,_=manufactured(120,device=gpu_device);gpu['maximum_CPU_GPU_position_difference_m']=float(abs(q-positions[120]).max())
        result=dict(status='COMPLETED_MANUFACTURED_COMPLIANT_GAUSSIAN_INTEGRATION',CPU=rows,passive_frame=dict(metrics=shifted,maximum_position_transform_error_m=frame_error),
            GPU=gpu,elapsed_s=time.monotonic()-started,source_sha256=pins,
            physical_shape_covariance_is_not_epistemic=True,no_cohort_or_observations_read=True,
            no_prior_or_posterior_inference=True,no_pinned_source_changes=True,
            limitations=protocol['scope'])
        if (max(r['maximum_position_error_m'] for r in rows)>2e-11 or max(r['maximum_force_residual_N'] for r in rows)>1e-9
                or max(r['maximum_energy_balance_error_J'] for r in rows)>1e-12 or frame_error>2e-11
                or (gpu and gpu['maximum_CPU_GPU_position_difference_m']>2e-11)):
            raise ValueError('Manufactured tolerance failed; preserve output and do not qualify')
        (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error),source_sha256=pins),indent=2)+'\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--gpu-device',choices=['cuda:2']);args=parser.parse_args()
    result=run(args.output,args.gpu_device);print(json.dumps(result,indent=2))
