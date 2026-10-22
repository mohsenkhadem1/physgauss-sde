"""Pure-bulk fixed-support gravity dynamics and bounded-memory DOP plumbing.

No contact/plane kernel is called, no integration runs at import, and no runtime
paths/packages are changed. Caller must provide the pinned Warp environment.
New physical trajectories require a separate reviewed, frozen protocol.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np
from scipy.integrate import DOP853
from scipy.sparse.linalg import splu
from experiments.campaign_static_p2_block import shape_gradients
from experiments.campaign_p2_consistent_transient import scalar_mass


class WarpBulkP2Force:
    """Original exported FP64 bulk/check kernels only; no dummy contact inputs."""
    def __init__(self,model,*,device='cpu'):
        import warp as wp
        from experiments.campaign_p2_warp_rhs import bulk,check_jacobian
        if wp.__version__!='1.18.0':raise ValueError('Explicit pinned Warp1.18.0 required')
        self.wp=wp;self.device=wp.get_device(device);self.bulk_kernel=bulk;self.check_kernel=check_jacobian
        self.n=len(model.rest);E=np.asarray(model.elements);V=np.asarray(model.volumes);b=np.asarray(model.bary);w=np.asarray(model.weights);checks=np.asarray(model.check_bary)
        if E.shape!=(len(V),10) or not np.issubdtype(E.dtype,np.integer) or np.any(E<0) or np.any(E>=self.n):raise ValueError('Valid actual T10 connectivity required')
        if not np.isfinite([model.young,model.poisson,model.minimum_J]).all() or model.young<=0 or not -1<model.poisson<.5 or not 0<model.minimum_J<1:raise ValueError('Finite valid material/Jguard required')
        grad=np.stack([shape_gradients(x[None],model.linear_gradients) for x in b]);check_grad=np.stack([shape_gradients(x[None],model.linear_gradients) for x in checks]);weights=w[:,None]*V[None]
        if not all(np.isfinite(a).all() for a in [grad,check_grad,weights]) or np.any(weights<=0):raise ValueError('Finite positive actual volume quadrature required')
        array=lambda x,dtype=wp.float64:wp.array(np.ascontiguousarray(x),dtype=dtype,device=self.device)
        self.elements=array(E,wp.int64);self.grad=array(grad);self.check_grad=array(check_grad);self.weights=array(weights)
        self.u=wp.empty((self.n,3),dtype=wp.float64,device=self.device);self.g=wp.zeros((self.n,3),dtype=wp.float64,device=self.device)
        self.E=wp.empty(weights.shape,dtype=wp.float64,device=self.device);self.J=wp.empty(weights.shape,dtype=wp.float64,device=self.device);self.Jcheck=wp.empty(check_grad.shape[:2],dtype=wp.float64,device=self.device)
        self.mu=model.young/(2*(1+model.poisson));self.kappa=self.mu+model.young*model.poisson/((1+model.poisson)*(1-2*model.poisson));self.minimum_J=model.minimum_J
        self.integration_points=len(b);self.additional_J_guard_points=len(checks)

    def evaluate(self,u):
        u=np.asarray(u)
        if u.shape!=(self.n,3) or u.dtype!=np.float64 or not np.isfinite(u).all():raise ValueError('Finite actual FP64 nodal displacement required')
        wp=self.wp;wp.copy(self.u,wp.array(np.ascontiguousarray(u),dtype=wp.float64,device=self.device));self.g.zero_()
        wp.launch(self.bulk_kernel,dim=self.E.shape,inputs=[self.u,self.elements,self.grad,self.weights,self.mu,self.kappa,self.E,self.J,self.g],device=self.device)
        wp.launch(self.check_kernel,dim=self.Jcheck.shape,inputs=[self.u,self.elements,self.check_grad,self.Jcheck],device=self.device)
        J,Jcheck=self.J.numpy(),self.Jcheck.numpy();minimum=float(min(J.min(),Jcheck.min()))
        if not np.isfinite(J).all() or not np.isfinite(Jcheck).all() or minimum<=self.minimum_J:raise ValueError('Original positive-J guard failed; no repair')
        energy=float(self.E.numpy().sum());g=self.g.numpy().copy()
        if not np.isfinite(energy) or not np.isfinite(g).all():raise ValueError('Nonfinite original elastic energy/force')
        return dict(elastic_energy_J=energy,gradient_N=g,minimum_J=minimum)


class SupportedBulkDynamics:
    """Fixed full-node supports, constant known loads and exact scalar M_ff LU.

    Physical scalar T10 mass is never row-lumped. M_cf contributes to the fixed
    reaction even though fixed u/v/a are identically zero. Supports with only
    selected coordinate axes or moving targets require a different interface.
    """
    def __init__(self,model,force_backend,*,mass_residual_tolerance_N=1e-9):
        self.model=model;self.backend=force_backend;self.mask=np.asarray(model.mask).copy();self.force=np.asarray(model.load,dtype=np.float64).copy()
        if self.mask.dtype!=np.bool_ or self.mask.shape!=model.rest.shape or not np.array_equal(self.mask,np.repeat(self.mask[:,:1],3,1)):raise ValueError('Explicit fixed full-vector node constraints required')
        self.free_nodes=np.flatnonzero(~self.mask[:,0]);self.fixed_nodes=np.flatnonzero(self.mask[:,0]);self.size=3*len(self.free_nodes);self.state_size=2*self.size
        if not self.size or not len(self.fixed_nodes) or not np.isfinite(self.force).all():raise ValueError('Nonempty free/fixed sets and finite known force required')
        if not np.isfinite(mass_residual_tolerance_N) or mass_residual_tolerance_N<=0:raise ValueError('Positive original mass residual tolerance required')
        self.mass=scalar_mass(model).tocsr();self.mass_ff=self.mass[self.free_nodes][:,self.free_nodes].tocsc();self.mass_lu=splu(self.mass_ff);self.mass_tolerance=mass_residual_tolerance_N
        if not np.isfinite(self.mass.data).all() or (self.mass.diagonal()<=0).any():raise ValueError('Original finite consistent mass required')
        self.rhs_calls=0;self.maximum_RHS_mass_residual_N=0.;self.minimum_RHS_J=float('inf')

    def pack(self,u,v):
        u=np.asarray(u);v=np.asarray(v)
        if any(a.shape!=self.model.rest.shape or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any(a[self.mask]!=0) for a in [u,v]):raise ValueError('Finite FP64 exact-fixed full nodal u/v required')
        return np.r_[u[self.free_nodes].ravel(),v[self.free_nodes].ravel()]

    def unpack(self,y):
        y=np.asarray(y)
        if y.dtype!=np.float64 or y.shape!=(self.state_size,) or not np.isfinite(y).all():raise ValueError('Matching finite FP64 free-coordinate state required')
        u=np.zeros_like(self.model.rest);v=np.zeros_like(u);u[self.free_nodes]=y[:self.size].reshape(-1,3);v[self.free_nodes]=y[self.size:].reshape(-1,3)
        return u,v

    def mechanics(self,y):
        u,v=self.unpack(y);elastic=self.backend.evaluate(u);net=self.force-elastic['gradient_N'];a=np.zeros_like(u);a[self.free_nodes]=self.mass_lu.solve(net[self.free_nodes]);Ma=self.mass@a
        R=float(np.linalg.norm((Ma-net)[self.free_nodes]))
        if not np.isfinite(a).all() or not np.isfinite(R) or R>self.mass_tolerance:raise ValueError('Original full consistent-M acceleration residual failed')
        kinetic=.5*float(np.sum(v*(self.mass@v)));work=float(np.sum(self.force*u));total=elastic['elastic_energy_J']+kinetic-work
        reaction=Ma+elastic['gradient_N']-self.force
        return dict(displacement_m=u,velocity_m_s=v,acceleration_m_s2=a,reaction_N=reaction,elastic_energy_J=elastic['elastic_energy_J'],kinetic_energy_J=kinetic,known_force_work_J=work,mechanical_energy_J=total,mass_residual_N=R,minimum_J=elastic['minimum_J'])

    def rhs(self,t,y):
        state=self.mechanics(y);self.rhs_calls+=1;self.maximum_RHS_mass_residual_N=max(self.maximum_RHS_mass_residual_N,state['mass_residual_N']);self.minimum_RHS_J=min(self.minimum_RHS_J,state['minimum_J'])
        return np.r_[state['velocity_m_s'][self.free_nodes].ravel(),state['acceleration_m_s2'][self.free_nodes].ravel()]

    def telemetry(self,y):
        a=self.mechanics(y)
        return {**{k:a[k] for k in ['mechanical_energy_J','elastic_energy_J','kinetic_energy_J','known_force_work_J','mass_residual_N','minimum_J']},'fixed_reaction_resultant_N':a['reaction_N'][self.fixed_nodes].sum(0).tolist(),'maximum_fixed_reaction_N':float(np.linalg.norm(a['reaction_N'][self.fixed_nodes],axis=1).max()),'maximum_fixed_u_m':float(abs(a['displacement_m'][self.mask]).max()),'maximum_fixed_v_m_s':float(abs(a['velocity_m_s'][self.mask]).max())}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AcceptedDOPCheckpoint:
    """Two alternating atomic slots keep one committed accepted prefix valid.

    Only current y/f/h, all scalar-ledger rows and common-output full states are
    retained. No all-accepted full-field trajectory is materialized or stored.
    A future resource continuation must hash/freeze this prefix in a new plan.
    """
    def __init__(self,directory,metadata):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.metadata=json.loads(json.dumps(metadata));self.commit_generation=0
        if (self.directory/'latest.json').exists():raise FileExistsError('Fresh checkpoint directory required; continuations need a new declared phase')

    def __call__(self,solver,ledger,common_times,common_states):
        count=len(ledger)-1;slot=self.commit_generation%2;file=self.directory/f'prefix-{slot}.npz';tmp=self.directory/f'prefix-{slot}.npz.tmp'
        with tmp.open('wb') as f:
            np.savez(f,current_y=solver.y,current_f=solver.f,current_time_s=solver.t,next_h_abs_s=solver.h_abs,previous_h_s=0. if solver.h_previous is None else solver.h_previous,previous_time_s=solver.t if solver.t_old is None else solver.t_old,nfev=solver.nfev,accepted_times_s=np.array([r['time_s'] for r in ledger]),accepted_telemetry_json=np.array([json.dumps(r,sort_keys=True) for r in ledger]),common_times_s=np.asarray(common_times),common_states=np.asarray(common_states),metadata_json=json.dumps(self.metadata,sort_keys=True));f.flush();os.fsync(f.fileno())
        os.replace(tmp,file);pointer=self.directory/'latest.json';temporary=self.directory/'latest.json.tmp'
        with temporary.open('w') as f:json.dump(dict(file=file.name,sha256=_sha(file),accepted_steps=count,time_s=solver.t,commit_generation=self.commit_generation),f);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temporary,pointer)
        self.commit_generation+=1


def read_dop_checkpoint(directory,expected_metadata):
    directory=Path(directory);pointer=json.loads((directory/'latest.json').read_text())
    if pointer['file'] not in ['prefix-0.npz','prefix-1.npz']:raise ValueError('Canonical committed prefix slot required')
    file=directory/pointer['file']
    if _sha(file)!=pointer['sha256']:raise ValueError('Changed committed accepted prefix')
    with np.load(file,allow_pickle=False) as a:
        if json.loads(str(a['metadata_json']))!=expected_metadata:raise ValueError('Source/model/force/protocol metadata mismatch')
        return {k:a[k].copy() for k in a.files}


def streaming_dop853(rhs,telemetry,initial,outputs_s,*,rtol,atol,max_step_s,energy_tolerance_J,checkpoint=None,checkpoint_every_accepted_steps=None):
    """Public DOP853.step, dense interpolation only at fixed shared outputs."""
    outputs=np.asarray(outputs_s,dtype=float);initial=np.asarray(initial)
    if outputs.ndim!=1 or len(outputs)<2 or not np.isfinite(outputs).all() or np.any(np.diff(outputs)<=0):raise ValueError('Finite ordered shared output clock required')
    if initial.dtype!=np.float64 or not np.isfinite(initial).all() or not np.isfinite(energy_tolerance_J) or energy_tolerance_J<=0:raise ValueError('FP64 initial state and explicit positive energy budget required')
    if checkpoint is not None and (type(checkpoint_every_accepted_steps) is not int or checkpoint_every_accepted_steps<1):raise ValueError('Explicit prospective positive checkpoint interval required')
    policy=dict(every_accepted_steps=checkpoint_every_accepted_steps,on_initial=True,on_new_common_output=True,on_final=True,max_uncommitted_accepted_steps_bound=checkpoint_every_accepted_steps)
    if isinstance(checkpoint,AcceptedDOPCheckpoint) and checkpoint.metadata.get('checkpoint_policy')!=policy:raise ValueError('Checkpoint metadata must bind actual prospective commit policy')
    solver=DOP853(rhs,outputs[0],initial.copy(),outputs[-1],rtol=rtol,atol=atol,max_step=max_step_s)
    ledger=[];common_times=[float(outputs[0])];common_states=[solver.y.copy()];common_telemetry=[];index=1;reference_energy=None;last_commit=0;commits=0;maximum_uncommitted=0
    def checked(y,t):
        row=dict(telemetry(y),time_s=float(t));E=float(row['mechanical_energy_J'])
        if not np.isfinite(E):raise ValueError('Nonfinite original mechanical energy')
        if reference_energy is not None and abs(E-reference_energy)>energy_tolerance_J:raise ValueError('Predeclared mechanical-energy guard failed')
        return row
    first=checked(solver.y,solver.t);reference_energy=first['mechanical_energy_J'];ledger.append(first);common_telemetry.append(first.copy())
    if checkpoint:checkpoint(solver,ledger,common_times,common_states);commits+=1
    while solver.status=='running':
        message=solver.step()
        if solver.status=='failed':raise RuntimeError('DOP853failed:'+str(message))
        ledger.append(checked(solver.y,solver.t));dense=None;previous_output_count=len(common_times)
        while index<len(outputs) and outputs[index]<=solver.t:
            t=float(outputs[index])
            if t<solver.t_old:raise AssertionError('Missed declared shared output')
            if t==solver.t:y=solver.y.copy()
            else:
                if dense is None:dense=solver.dense_output()
                y=dense(t).copy()
            row=checked(y,t);common_times.append(t);common_states.append(y);common_telemetry.append(row);index+=1
        if checkpoint:
            count=len(ledger)-1;maximum_uncommitted=max(maximum_uncommitted,count-last_commit)
            if count-last_commit>=checkpoint_every_accepted_steps or len(common_times)>previous_output_count or solver.status=='finished':
                checkpoint(solver,ledger,common_times,common_states);last_commit=count;commits+=1
    if index!=len(outputs) or solver.t!=outputs[-1]:raise RuntimeError('Incomplete declared output schedule')
    return dict(times_s=np.asarray(common_times),common_states=np.asarray(common_states),common_telemetry=common_telemetry,accepted_scalar_ledger=ledger,accepted_steps=len(ledger)-1,nfev=solver.nfev,initial_mechanical_energy_J=reference_energy,energy_tolerance_J=energy_tolerance_J,checkpoint_policy=policy,checkpoint_commits=commits,maximum_observed_uncommitted_accepted_steps=maximum_uncommitted)
