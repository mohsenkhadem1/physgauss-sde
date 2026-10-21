"""Tiny finite-strain Gaussian-interaction reference, not a dataset replacement.

Nodal compatible compressible log-J Neo-Hookean solid; rest-material surface
quadrature, finite world-frame PSD springs/dampers, exact optional node-axis
constraints. No contact, rigid handle model, inversion/stiffness repair or Gaussian
extra mass. Shape covariance is physical extent, never posterior uncertainty.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
try:
    from . import campaign_boundary_gaussians as boundary
    from .campaign_compliant_gaussian_linear import GaussianFixtureMatrix, tiny_geometry, candidate_patches
    from .campaign_linear_fem import assemble_tetrahedra
    from .campaign_nonlinear_fem import neo_hookean_energy, InvalidConfiguration
    from .campaign_material_gaussians import MaterialGaussians
except ImportError:
    import campaign_boundary_gaussians as boundary
    from campaign_compliant_gaussian_linear import GaussianFixtureMatrix, tiny_geometry, candidate_patches
    from campaign_linear_fem import assemble_tetrahedra
    from campaign_nonlinear_fem import neo_hookean_energy, InvalidConfiguration
    from campaign_material_gaussians import MaterialGaussians


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


class StepFailure(RuntimeError):
    """Explicit failed solve; input state is never modified."""
    def __init__(self, reason, history):
        super().__init__(reason)
        self.history = history


class NonlinearGaussianReference:
    def __init__(self, rest, tets, faces, *, young, poisson, density,
                 support_patches, stiffness, damping, load_patches,
                 constraint_mask=None, minimum_J=1e-8):
        self.rest=tensor(rest).clone(); self.tets=torch.as_tensor(tets,dtype=torch.long)
        self.young,self.poisson=float(young),float(poisson)
        if not np.isfinite(minimum_J) or not 0 < minimum_J < 1: raise ValueError('Declare 0 < minimum_J < 1')
        self.minimum_J=float(minimum_J)
        self.K0,self.mass,self.gradients,self.volumes=assemble_tetrahedra(rest,tets,young,poisson,density)
        q=boundary.SurfaceQuadrature.from_mesh(rest,faces)
        self.fixture=GaussianFixtureMatrix.from_patches(q,support_patches,stiffness,damping)
        bank=boundary.patch_bank(q,load_patches,2,{'known_hardware','candidate_force_hypothesis'})
        self.load_weights=tensor(bank['normalized_node_weights'])
        self.load_active=torch.as_tensor(bank['active_mask'],dtype=torch.bool).clone()
        mask=np.zeros(self.rest.shape,dtype=bool) if constraint_mask is None else np.asarray(constraint_mask)
        if mask.dtype!=np.bool_ or mask.shape!=self.rest.shape: raise ValueError('Explicit boolean node-axis constraint mask required')
        self.mask=torch.tensor(mask.copy()); self.free=(~self.mask).flatten()
        self.kernels=MaterialGaussians.from_tetrahedra(self.rest,self.tets,density)

    def elastic(self, positions):
        return neo_hookean_energy(positions,self.rest,self.tets,self.gradients,self.volumes,
                                 self.young,self.poisson,minimum_J=self.minimum_J)

    def known_force(self, total_forces, *, gravity=(0.,0.,0.), nodal_force=None):
        """Held world forces distributed by normalized rest-material weights.

        This is a distributed surface load, not exact rigid-handle mechanics.
        Two Gaussian force resultants are normalized to float64 quadrature
        precision; gravity uses the sole physical lumped solid mass.
        """
        total=tensor(total_forces); g=tensor(gravity)
        if total.shape!=(2,3) or g.shape!=(3,): raise ValueError('Explicit H2 forces and world gravity required')
        if not bool(torch.isfinite(total).all()) or not bool(torch.isfinite(g).all()): raise ValueError('Nonfinite known force or gravity')
        if bool((total[~self.load_active]!=0).any()): raise ValueError('Inactive load patch cannot carry force')
        force=self.load_weights.T@total+self.mass.reshape(-1,3)*g
        if nodal_force is not None:
            extra=tensor(nodal_force)
            if extra.shape!=self.rest.shape: raise ValueError('Nodal force shape mismatch')
            force=force+extra
        if not bool(torch.isfinite(force).all()): raise ValueError('Nonfinite known force')
        return force

    def energy(self, x, v, targets):
        elastic,_=self.elastic(x)
        response=self.fixture.response(x,v,targets,torch.zeros_like(targets))
        return elastic+response['energy']+.5*(self.mass*v.flatten().square()).sum()

    def step(self, x, v, target_old, target_new, force, h, *, constraint_positions=None,
             absolute_tolerance_N=1e-10, relative_tolerance=1e-8, maximum_iterations=50):
        """Backward Euler stationary incremental potential, exact constraints.

        Newton directions use the exact Hessian when SPD; otherwise a declared
        rest dynamic metric supplies descent. Neither changes the objective.
        Positive-J Armijo search; near-roundoff acceptance additionally requires
        residual decrease. Accepted solutions satisfy the ORIGINAL free gradient.
        """
        x,v,target_old,target_new,force=map(tensor,(x,v,target_old,target_new,force))
        if (not np.isfinite(h) or h<=0 or maximum_iterations<1 or absolute_tolerance_N<=0
                or relative_tolerance<0): raise ValueError('Invalid solve settings')
        if x.shape!=self.rest.shape or v.shape!=x.shape or force.shape!=x.shape: raise ValueError('State/force shape mismatch')
        if not all(bool(torch.isfinite(a).all()) for a in (x,v,target_old,target_new,force)): raise ValueError('Nonfinite inputs')
        self.elastic(x)
        tv=(target_new-target_old)/h
        self.fixture.response(x,v,target_new,tv)
        if bool(self.mask.any()):
            if constraint_positions is None: raise ValueError('Exact prescribed constrained positions required')
            prescribed=tensor(constraint_positions)
            if prescribed.shape!=x.shape or not bool(torch.isfinite(prescribed).all()): raise ValueError('Invalid exact targets')
        else: prescribed=x
        # Solve position increments, preserving small-step arithmetic near rest.
        base=torch.zeros_like(x).flatten(); base[self.mask.flatten()]=(prescribed-x).flatten()[self.mask.flatten()]
        z=base[self.free].clone()
        def unpack(value): return base.scatter(0,torch.where(self.free)[0],value).reshape_as(x)
        def potential(value):
            dx=unpack(value); new=torch.where(self.mask,prescribed,x+dx)
            U,J=self.elastic(new)
            e=new[None]-target_new; d=dx[None]-h*tv
            spring=.5*torch.einsum('sn,sni,sij,snj->',self.fixture.weights,e,self.fixture.stiffness,e)
            drag=.5/h*torch.einsum('sn,sni,sij,snj->',self.fixture.weights,d,self.fixture.damping,d)
            inertia=.5*(self.mass*(dx-h*v).flatten().square()).sum()/h**2
            return inertia+U+spring+drag-(force*dx).sum(),J
        def evaluate(value):
            a=value.detach().requires_grad_(True); p,J=potential(a)
            if not bool(torch.isfinite(p)): raise StepFailure('Nonfinite potential',history)
            grad=torch.autograd.grad(p,a)[0]
            if not bool(torch.isfinite(grad).all()): raise StepFailure('Nonfinite gradient',history)
            return float(p.detach()),grad.detach(),float(J.min().detach())
        history=[]; threshold=max(absolute_tolerance_N,relative_tolerance*float(torch.linalg.vector_norm(force)))
        metric=torch.diag(self.mass/h**2)+self.K0+self.fixture.K+self.fixture.C/h
        factor=torch.linalg.cholesky(metric[self.free][:,self.free]) if bool(self.free.any()) else None
        for iteration in range(maximum_iterations+1):
            try: value,g,minJ=evaluate(z)
            except InvalidConfiguration as exc: raise StepFailure(str(exc),history) from exc
            norm=float(torch.linalg.vector_norm(g))
            row=dict(iteration=iteration,residual_N=norm,minimum_J=minJ,solver_regularization_N_m=0.)
            history.append(row)
            if norm<=threshold: break
            if iteration==maximum_iterations: raise StepFailure('Iteration cap, original residual not accepted',history)
            H=torch.autograd.functional.hessian(lambda a:potential(a)[0],z)
            L,info=torch.linalg.cholesky_ex(H)
            row['direction']='exact_hessian' if int(info)==0 else 'rest_metric_descent'
            direction=-torch.cholesky_solve(g[:,None],L if int(info)==0 else factor)[:,0]
            slope=float(g@direction)
            if not np.isfinite(slope) or slope>=0: raise StepFailure('Non-descent solver direction',history)
            for backtrack in range(40):
                alpha=2.**(-backtrack); candidate=z+alpha*direction
                try: val2,g2,_=evaluate(candidate)
                except InvalidConfiguration: continue
                window=128*np.finfo(float).eps*max(abs(value),self.young*float(self.volumes.sum()),1e-30)
                if val2<=value+1e-4*alpha*slope or (abs(val2-value)<=window and float(torch.linalg.vector_norm(g2))<norm):
                    z=candidate; row['alpha']=alpha; break
            else: raise StepFailure('Positive-J line search exhausted',history)
        dx=unpack(z).detach(); new=torch.where(self.mask,prescribed,x+dx).detach(); newv=(dx/h).detach()
        U0,_=self.elastic(x); U1,J=self.elastic(new)
        a=new.detach().requires_grad_(True); elastic_grad=torch.autograd.grad(self.elastic(a)[0],a)[0].detach()
        response=self.fixture.response(new,newv,target_new,tv)
        full_residual=(self.mass*((newv-v)/h).flatten()).reshape_as(x)+elastic_grad-response['forces']-force
        de=dx[None]-(target_new-target_old)
        kinetic_num=.5*(self.mass*(newv-v).flatten().square()).sum()
        spring_num=.5*torch.einsum('sn,sni,sij,snj->',self.fixture.weights,de,self.fixture.stiffness,de)
        elastic_bregman=(elastic_grad*dx).sum()-(U1-U0) # signed for nonconvex elasticity
        external=(force*dx).sum(); target_work=(response['forces_each']*(target_new-target_old)).sum()
        reaction_work=(full_residual[self.mask]*dx[self.mask]).sum()
        delta=self.energy(new,newv,target_new)-self.energy(x,v,target_old)
        diss=h*response['dissipation']; numerical=kinetic_num+spring_num+elastic_bregman
        balance=delta-external-target_work-reaction_work+diss+numerical
        ledger=dict(accepted=True,iterations=iteration,force_residual_N=float(torch.linalg.vector_norm(full_residual.flatten()[self.free])),
            threshold_N=threshold,minimum_J=float(J.min()),constraint_max_error_m=float((new[self.mask]-prescribed[self.mask]).abs().max()) if bool(self.mask.any()) else 0.,
            delta_energy_J=float(delta),external_work_J=float(external),target_work_J=float(target_work),reaction_work_J=float(reaction_work),
            damping_dissipation_J=float(diss),implicit_euler_numerical_term_J=float(numerical),
            signed_elastic_bregman_J=float(elastic_bregman),balance_error_J=float(balance),
            free_residual_work_J=float((full_residual.flatten()[self.free]*dx.flatten()[self.free]).sum()),history=history)
        return new,newv,ledger


def tiny_model(*, young=5000., mask=None, support_scale=1.):
    X,tets,faces=tiny_geometry(); patches,K,D=candidate_patches()
    loads=[boundary.GaussianPatch(p.center_material_m,p.covariance_material_m2,'known_hardware',p.active) for p in patches]
    return NonlinearGaussianReference(X,tets,faces,young=young,poisson=.35,density=625.,
        support_patches=patches,stiffness=K*support_scale,damping=D*support_scale,load_patches=loads,constraint_mask=mask)


def transient(model,hz,duration=.2):
    h=1/hz; x=model.rest.clone();v=torch.zeros_like(x);targets=model.rest[None].repeat(2,1,1)
    states=[x.clone()]; rows=[]
    for k in range(round(duration*hz)):
        t=(k+.5)*h
        f=model.known_force([[.004*np.sin(np.pi*t/duration)**2, .002,0.],[0.,0.,0.]])
        x,v,row=model.step(x,v,targets,targets,f,h)
        rows.append(row);states.append(x.clone())
    return torch.stack(states),rows


def run(output):
    torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path('experiments/campaign_nonlinear_fem.py'),Path('experiments/campaign_boundary_gaussians.py'),Path('experiments/campaign_material_gaussians.py'),Path('experiments/campaign_compliant_gaussian_linear.py'),Path('experiments/campaign_linear_fem.py')]
    protocol=dict(scope='Tiny independent nonlinear implementation checks, no cohort/native/continuum/inference claim',rates_hz=[60,120,240],duration_s=.2,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        results=[transient(tiny_model(),hz) for hz in protocol['rates_hz']]
        errors=[float(torch.sqrt(((results[i][0]-results[i+1][0][::2])**2).mean())) for i in range(2)]
        rows=[r for _,rr in results for r in rr]
        model=tiny_model(); transported=model.kernels.transport(results[-1][0]);mass_error=float((transported.bulk_density*transported.volumes-model.kernels.masses).abs().max())
        summary=dict(completed=True,temporal_full_trajectory_rmse_m=errors,adjacent_contraction=errors[1]/errors[0],minimum_J=min(r['minimum_J'] for r in rows),maximum_original_force_residual_N=max(r['force_residual_N'] for r in rows),maximum_energy_balance_error_J=max(abs(r['balance_error_J']) for r in rows),maximum_balance_minus_residual_work_J=max(abs(r['balance_error_J']-r['free_residual_work_J']) for r in rows),mass_transport_error_kg=mass_error,rest_metric_fallbacks=sum(h.get('direction')=='rest_metric_descent' for r in rows for h in r['history']),limitations=['Discrete tiny-mesh validation only','Signed nonlinear elastic Bregman term is not guaranteed nonnegative','No contact or exact rigid handles','Gaussian covariance is shape extent, not uncertainty'])
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc),history=getattr(exc,'history',[])),indent=2)+'\n');raise
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output),indent=2))
