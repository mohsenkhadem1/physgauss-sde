"""Accepted-iterate durability for NEW static P2 drivers only.

Numerical statements follow the frozen TorchP2 static solve verbatim. The only
addition is a callback on a cloned accepted iterate, before the next CG solve.
Old backends and live jobs are never patched. A checkpoint is not convergence.
"""
from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from experiments.campaign_p2_torch import NonlinearFailure

ROOT=Path(__file__).resolve().parents[1]


class CheckpointedStaticSolveMixin:
    def solve(self,force=None,initial=None,target=None,effective_dt=None,
              force_tolerance=1e-10,maximum_iterations=25,on_accepted=None):
        if target is not None or effective_dt is not None:
            raise ValueError('Explicitly static checkpoint helper only')
        force=self.force if force is None else force
        u=torch.zeros_like(self.rest) if initial is None else initial.clone()
        if not bool(torch.isfinite(u).all()) or bool((u[self.mask]!=0).any()):raise ValueError('Finite exactly constrained initial displacement required')
        history=[]
        def objective(x):
            cache=self.evaluate(x);res=cache['gradient']-force;value=cache['energy']-(force*x).sum()
            return cache,res.masked_fill(self.mask,0.),value
        for it in range(maximum_iterations+1):
            cache,r,potential=objective(u);norm=float(torch.linalg.vector_norm(r));row=dict(iteration=it,residual_N=norm,potential_J=float(potential),minimum_J=float(cache['minimum_J']));history.append(row)
            if on_accepted is not None:
                on_accepted(dict(displacement=u.detach().clone(),iteration=it,
                    residual_N=norm,potential_J=float(potential),minimum_J=float(cache['minimum_J']),
                    history=deepcopy(history),solve_configuration=dict(force_tolerance_N=force_tolerance,maximum_iterations=maximum_iterations,maximum_CG_iterations=self.maximum_cg_iterations,quadrature_chunk_size=self.quadrature_chunk_size)))
            if norm<=force_tolerance:return dict(displacement=u,positions=self.rest+u,cache=cache,residual=r,history=history,iterations=it)
            if it==maximum_iterations:raise NonlinearFailure('Original nonlinear residual exceeds cap: '+str(history[-1]))
            direction,cg_steps=self.cg(cache,-r,0.,rtol=min(.01,max(1e-8,.1*norm**.5)));row['cg_iterations']=cg_steps
            slope=float((r*direction).sum())
            if not np.isfinite(slope) or slope>=0:raise NonlinearFailure('Original Newton direction is not descent')
            for k in range(30):
                alpha=2.**(-k);trial=u+alpha*direction
                try:cc,rr,vv=objective(trial)
                except ValueError:continue
                energy_roundoff=128*np.finfo(float).eps*max(abs(float(potential)),self.energy_scale)
                if float(vv)<=float(potential)+1e-4*alpha*slope or(abs(float(vv-potential))<=energy_roundoff and float(torch.linalg.vector_norm(rr))<norm):
                    u=trial;row['alpha']=alpha;break
            else:raise NonlinearFailure('Positive-J original-objective line search failed')
        raise AssertionError('Unreachable')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_record(value):
    a=np.ascontiguousarray(value)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())


def checkpoint_identity(model,solver,*,source_sha256,protocol_sha256,force=None):
    """Hash the actual geometry/constitutive inputs and actual nodal force."""
    sources=dict(source_sha256)
    own=str(Path(__file__).resolve().relative_to(ROOT))
    if sources.get(own)!=_sha(__file__):raise ValueError('Checkpoint helper must be explicitly source-pinned')
    for path,digest in sources.items():
        p=Path(path)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=path or _sha(ROOT/p)!=digest:raise ValueError('Canonical unchanged source pins required')
    if not isinstance(protocol_sha256,str) or len(protocol_sha256)!=64 or any(c not in '0123456789abcdef' for c in protocol_sha256):raise ValueError('Frozen protocol SHA256 required')
    fields=['rest','tets','elements','faces','linear_gradients','volumes','mask','bary','weights','check_bary']
    geometry={name:_array_record(getattr(model,name)) for name in fields}
    material={name:float(getattr(model,name)) for name in ['young','poisson','density','minimum_J']}
    force_array=solver.force.detach().cpu().numpy() if force is None else force.detach().cpu().numpy()
    return dict(schema='P2-accepted-static-identity-v0.1',protocol_sha256=protocol_sha256,source_sha256=sources,model_arrays=geometry,material=material,actual_nodal_force=_array_record(force_array))


class AcceptedStaticCheckpoint:
    """Atomically commit immutable numbered accepted states, then latest pointer.

    An interrupted write leaves the previous pointer valid. No rejected trial or
    CG iterate is saved as accepted. Full source/model/force/config identities
    must match on reload. This class does not authorize any physical resume.
    """
    def __init__(self,directory,identity):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.identity=deepcopy(identity)

    def __call__(self,state):
        iteration=state['iteration'];base=self.directory/f'accepted-{iteration:03d}'
        if base.with_suffix('.json').exists() or base.with_suffix('.npz').exists():raise FileExistsError('Preserve existing accepted checkpoint')
        data=base.with_suffix('.npz');temporary=base.with_suffix('.npz.tmp');u=state['displacement'].detach().cpu().numpy()
        with temporary.open('xb') as f:np.savez(f,displacement_m=u);f.flush();os.fsync(f.fileno())
        os.replace(temporary,data)
        metadata=dict(identity=self.identity,iteration=iteration,residual_N=state['residual_N'],potential_J=state['potential_J'],minimum_J=state['minimum_J'],history=state['history'],solve_configuration=state['solve_configuration'],data_file=data.name,data_sha256=_sha(data),accepted_not_necessarily_converged=True)
        meta=base.with_suffix('.json');tmp=base.with_suffix('.json.tmp')
        with tmp.open('x') as f:json.dump(metadata,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,meta)
        pointer=self.directory/'latest.json';tmp=self.directory/'latest.json.tmp'
        with tmp.open('w') as f:json.dump(dict(metadata_file=meta.name,metadata_sha256=_sha(meta)),f);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,pointer)


def load_accepted_checkpoint(directory,expected_identity,expected_configuration,*,device='cpu'):
    directory=Path(directory);pointer=json.loads((directory/'latest.json').read_text());name=pointer['metadata_file']
    if Path(name).name!=name or not name.startswith('accepted-'):raise ValueError('Local numbered accepted metadata required')
    path=directory/name
    if _sha(path)!=pointer['metadata_sha256']:raise ValueError('Changed accepted metadata')
    meta=json.loads(path.read_text())
    if meta['identity']!=expected_identity or meta['solve_configuration']!=expected_configuration:raise ValueError('Source/model/force/protocol/config identity mismatch')
    data=meta['data_file']
    if Path(data).name!=data or _sha(directory/data)!=meta['data_sha256']:raise ValueError('Changed accepted array')
    with np.load(directory/data,allow_pickle=False) as a:u=a['displacement_m'].copy()
    if u.dtype!=np.float64 or not np.isfinite(u).all() or list(u.shape)!=expected_identity['model_arrays']['rest']['shape']:raise ValueError('Finite original FP64 nodal displacement required')
    for p,h in expected_identity['source_sha256'].items():
        if _sha(ROOT/p)!=h:raise ValueError('Runtime source changed since accepted checkpoint')
    if not 0<=meta['iteration']<=expected_configuration['maximum_iterations']:raise ValueError('Accepted iteration outside original budget')
    return dict(initial=torch.tensor(u,dtype=torch.float64,device=device),metadata=meta,remaining_Newton_iterations=expected_configuration['maximum_iterations']-meta['iteration'])
