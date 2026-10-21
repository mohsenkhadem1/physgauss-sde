"""Frozen one-step diagnostic: original cache replay, long reference, refreshed cache.

Never edits/relabels the original failure or changes its physics/tolerances.
Only this new wrapper and private model caches are used. No trajectory replay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

try:
    import campaign_nonlinear_dynamics as core
    import campaign_authored_force_case as old
except ImportError:
    from experiments import campaign_nonlinear_dynamics as core
    from experiments import campaign_authored_force_case as old

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).resolve()
CASE='runs/gaussian-campaign-v01/authored-force-envelope-v01/stage1/cells/t0000__twist_couple__temporal__hz120'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def resolve(p):return old.resolve_repo_path(p)


def restore_state(arrays):
    """Restore saved point-chord state directly, never tangent initialization."""
    mapping={'positions':'positions_m','velocities':'velocities_point_chord_m_s',
        'auxiliary_positions':'auxiliary_positions_m','auxiliary_velocities':'auxiliary_velocities_point_chord_m_s',
        'handle_centers':'handle_centers_m','handle_rotations':'handle_rotations','reduced_position_coordinates':'reduced_position_coordinates'}
    values={}
    for field,key in mapping.items():
        value=np.asarray(arrays[key]);
        if value.dtype!=np.dtype('float64') or not np.isfinite(value).all():raise ValueError('Exact finite float64 state required: '+key)
        values[field]=torch.from_numpy(value.copy())
    return core.DynamicState(time_s=float(arrays['time_s']),velocity_semantics='Restored saved preceding-step point chords; no tangent initialization.',**values)


def step_capture(call, code):
    """Copy only this original step frame's q/history on exit; no source edits."""
    if sys.gettrace() is not None:raise RuntimeError('Existing Python tracer would be disturbed')
    captured={}
    def local(frame,event,arg):
        if event=='return':
            q=frame.f_locals.get('q')
            if isinstance(q,torch.Tensor):captured['q']=q.detach().clone()
            if isinstance(frame.f_locals.get('history'),list):captured['history']=[dict(row) for row in frame.f_locals['history']]
        return local
    def trace(frame,event,arg):
        if event=='call' and frame.f_code is code:
            frame.f_trace_lines=False;frame.f_trace_opcodes=False;return local
        return None
    result=None;error=None
    try:
        sys.settrace(trace)
        result=call()
    except RuntimeError as problem:error=str(problem)
    finally:sys.settrace(None)
    return result,error,captured


def original_objective(model,state,h,F,q):
    """Exact original step expression; used only for posthoc gradients/curvature."""
    old_q=model.coordinates(state);target=model.targets(state.time_s+h)
    x,Y,c,R=model.configuration(q,state.handle_rotations,target)
    dx,dY,dc=model.increments(q,old_q,state,target)
    U,J=model.elastic(x)
    objective=(model.body_mass[:,None]*(dx-h*state.velocities).square()).sum()/(2*h*h)
    objective+=(model.auxiliary_mass[:,None]*(dY-h*state.auxiliary_velocities).square()).sum()/(2*h*h)
    objective+=U+model.drag*(model.body_mass[:,None]*dx.square()).sum()/(2*h)
    objective-=(F*dc).sum()  # Exact failed command has identically zero nodal force.
    return objective,J


def lanczos_probe(apply,size,*,seed=521,steps=24):
    """Symmetric Lanczos Ritz diagnostics, NOT certified global eigenbounds."""
    generator=torch.Generator(device='cpu');generator.manual_seed(seed)
    vector=torch.randn(size,dtype=torch.float64,generator=generator);vector/=torch.linalg.vector_norm(vector)
    basis=[];diagonal=[];off=[];last_beta=0.
    for index in range(min(size,steps)):
        basis.append(vector);product=apply(vector);alpha=float(vector@product);diagonal.append(alpha)
        residual=product-alpha*vector
        if index:residual-=off[-1]*basis[-2]
        for _ in range(2):
            for v in basis:residual-=(v@residual)*v
        beta=float(torch.linalg.vector_norm(residual));last_beta=beta
        if beta<1e-13 or index+1==min(size,steps):break
        off.append(beta);vector=residual/beta
    matrix=torch.diag(torch.tensor(diagonal,dtype=torch.float64))
    if len(diagonal)>1:
        sub=torch.tensor(off[:len(diagonal)-1],dtype=torch.float64);matrix+=torch.diag(sub,1)+torch.diag(sub,-1)
    eig,vec=torch.linalg.eigh(matrix)
    return {'steps':len(diagonal),'minimum_Ritz':float(eig[0]),'maximum_Ritz':float(eig[-1]),
        'minimum_Ritz_residual_estimate':last_beta*abs(float(vec[-1,0])),
        'maximum_Ritz_residual_estimate':last_beta*abs(float(vec[-1,-1])),
        'not_global_eigenvalue_bounds':True,'seed':seed}


def curvature(model,state,h,F,q,factors):
    value=q.detach().clone().requires_grad_(True);objective,J=original_objective(model,state,h,F,value)
    gradient=torch.autograd.grad(objective,value,create_graph=True)[0]
    applied=torch.zeros(model.size,dtype=torch.float64)
    for j,dofs in enumerate(model.handle_dofs):applied[dofs[:3]]+=F[j]
    metrics=core.equilibrium_metrics(gradient.detach(),applied,model.handle_dofs,model.length_scale,value.detach())
    again=value.detach().clone().requires_grad_(True);obj2,_=original_objective(model,state,h,F,again);g2=torch.autograd.grad(obj2,again)[0]
    result={'incremental_potential_J':float(objective.detach()),'minimum_J':float(J.min().detach()),
        'physical_metrics':metrics,'repeat_gradient_maximum_difference':float((g2-gradient.detach()).abs().max()),'preconditioners':{}}
    for name,L in factors.items():
        def apply(v):
            w=torch.linalg.solve_triangular(L.T,v[:,None],upper=True)[:,0]
            hv=torch.autograd.grad(gradient,value,grad_outputs=w,retain_graph=True)[0]
            return torch.linalg.solve_triangular(L,hv[:,None],upper=False)[:,0].detach()
        probe=lanczos_probe(apply,model.size)
        rg=torch.linalg.solve_triangular(L,gradient.detach()[:,None],upper=False)[:,0];norm=float(torch.linalg.vector_norm(rg))
        if norm:probe['residual_direction_Rayleigh']=float((rg/norm)@apply(rg/norm))
        result['preconditioners'][name]=probe
    return result


def assemble_model(physical,arrays,translation):
    case,spec,hardware=physical['case'],physical['spec'],physical['hardware'];m=case['material']
    points=arrays['rest_points_m']+np.asarray(translation)
    if case['boundary_type']!='point_pin':raise ValueError('Only exact failed point-pin case is declared')
    return core.NonlinearDynamics(points,arrays['tetrahedra'],m['youngs_modulus_Pa'],m['poissons_ratio'],m['density_kg_m3'],
        fixture_nodes=spec['fixture_nodes'],load_patches=spec['load_patches'],load_nodes=spec['load_nodes'],
        handle_masses=[h['mass_kg'] for h in hardware],handle_inertias=[h['full_COM_inertia_kg_m2'] for h in hardware],
        handle_com_offsets=[h['COM_offset_m'] for h in hardware],drag=m['linear_damping_s_inverse'],boundary_kind=case['boundary_type'])


def check_protocol(protocol_path,expected_hash):
    p=resolve(protocol_path)
    if sha(p)!=expected_hash:raise ValueError('New diagnostic protocol hash mismatch')
    protocol=read(p)
    if protocol['protocol']!='authored-force-failed-step-ABC-v0.1' or protocol['status']!='FROZEN_AUTHORIZED_ONE_STEP_ONLY':raise ValueError('Exact new one-step scope required')
    if protocol['case']!=CASE or protocol['variants']!=[{'id':'A','preconditioner':'original_rest_cached','cap':250},{'id':'B','preconditioner':'original_rest_cached','cap':1000},{'id':'C','preconditioner':'current_state_refreshed','cap':250}]:raise ValueError('Declared ABC matrix changed')
    required={SOURCE.relative_to(ROOT).as_posix(),*old.dependencies(),*[CASE+'/'+name for name in ['last-accepted-state.npz','failure.json','protocol.json','unaccepted-core-locals.npz','steps.jsonl','private-provenance/static-reference.npz','private-provenance/physical-inputs.json']]}
    if not required.issubset(protocol['source_and_input_sha256']):raise ValueError('Incomplete frozen source/input coverage')
    for rel,digest in protocol['source_and_input_sha256'].items():
        if sha(resolve(rel))!=digest:raise ValueError('Pinned source/input changed: '+rel)
    if protocol['source_and_input_sha256'].get(SOURCE.relative_to(ROOT).as_posix())!=sha(SOURCE):raise ValueError('Wrapper source is not frozen')
    termination=protocol['terminal_guard'];proof=read(resolve(termination['path']))
    if sha(resolve(termination['path']))!=termination['sha256']:raise ValueError('Terminal proof changed')
    expected_pids=[proof['supervisor_pid']]+[row['pid'] for row in proof['workers'] if row.get('pid') is not None]
    if (proof.get('stage_terminal') is not True or proof.get('all_launched_worker_PIDs_missing') is not True
            or proof.get('supervisor_PID_missing') is not True or len(expected_pids)!=28
            or sorted(protocol['terminal_guard']['verified_absent_pids'])!=sorted(expected_pids)):
        raise ValueError('Complete publisher terminal boundary required')
    # The publisher's exact verified proof is immutable; independently reject any live PID.
    for pid in protocol['terminal_guard']['verified_absent_pids']:
        if Path('/proc',str(pid)).exists():raise ValueError('Old worker/supervisor still exists: '+str(pid))
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('Exactly two CPU threads required')
    return protocol


def run(protocol_path,protocol_sha):
    protocol=check_protocol(protocol_path,protocol_sha);torch.set_num_threads(2)
    base=resolve(CASE);output=resolve(protocol['fresh_output']);output.mkdir(parents=True,exist_ok=False)
    write(output/'execution.json',{'terminal':False,'completed':False,'pid':os.getpid(),'protocol_sha256':protocol_sha})
    (output/'protocol.json').write_bytes(resolve(protocol_path).read_bytes())
    physical=read(base/'private-provenance/physical-inputs.json');saved_protocol=read(base/'protocol.json');failure=read(base/'failure.json')
    with np.load(base/'private-provenance/static-reference.npz',allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    with np.load(base/'last-accepted-state.npz',allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
    h=failure['step_request']['requested_dt_s'];F=torch.tensor(failure['step_request']['held_handle_forces_world_N'],dtype=torch.float64)
    if h!=1/120 or bool(F.any()) or failure['completed_steps']!=240:raise ValueError('Exact declared zero-force failed step required')
    reports={};states={};qpoints={};started=time.monotonic()
    try:
        for variant in protocol['variants']:
            name=variant['id'];folder=output/name;folder.mkdir()
            model=assemble_model(physical,arrays,saved_protocol['cell']['translation_m'])
            rest=model.initial_state();L0=model.preconditioner(rest,h).clone()  # Mandatory original first-state cache.
            state=restore_state(saved);model.validate_state(state)
            if name=='C':model._factor_cache.clear();model.preconditioner(state,h)
            factor=model.preconditioner(state,h).clone()
            wall=time.monotonic()
            result,error,captured=step_capture(lambda:model.step(state,h,handle_forces=F,
                relative_tolerance=1e-7,absolute_tolerance_N=1e-10,maximum_iterations=variant['cap']),core.NonlinearDynamics.step.__code__)
            history=captured.get('history',[]);write(folder/'history.json',history)
            q=captured.get('q')
            if q is None:raise RuntimeError('Original step q was not captured')
            qpoints[name]=q;np.savez_compressed(folder/'optimizer-q.npz',q=q.numpy())
            report={'id':name,'preconditioner':variant['preconditioner'],'cap':variant['cap'],'completed':result is not None,'error':error,
                'elapsed_s':time.monotonic()-wall,'iterations':len(history)-1,'last':history[-1],
                'preconditioner_factor_sha256':hashlib.sha256(factor.numpy().tobytes()).hexdigest(),
                'rest_preconditioner_factor_sha256':hashlib.sha256(L0.numpy().tobytes()).hexdigest(),
                'original_failure_requalified':False,'full_trajectory_qualified':False}
            if result is not None:
                new,ledger=result;states[name]=new;old.save_state(folder/'state.npz',new);write(folder/'ledger.json',ledger)
            reports[name]=report;write(folder/'summary.json',report);print(json.dumps(report),flush=True)
            if name=='A':
                original=failure['core_failure']['nonlinear_history']
                with np.load(base/'unaccepted-core-locals.npz',allow_pickle=False) as a:original_q=a['q']
                replay=error is not None and len(history)==251 and history==original and np.array_equal(q.numpy(),original_q)
                report['exact_original_history_reproduced']=replay;write(folder/'summary.json',report)
                if not replay:raise RuntimeError('A failed to exactly reproduce original cached-preconditioner failure; stop')
        # Curvature/gradient re-evaluation at the same original q for both cached metrics.
        model=assemble_model(physical,arrays,saved_protocol['cell']['translation_m']);rest=model.initial_state();L0=model.preconditioner(rest,h).clone()
        state=restore_state(saved);model._factor_cache.clear();Lc=model.preconditioner(state,h).clone()
        curvature_results={name:curvature(model,state,h,F,qpoints[name],{'rest_cached':L0,'current_refreshed':Lc}) for name in qpoints}
        write(output/'curvature.json',curvature_results)
        for name in qpoints:
            close_error=abs(curvature_results[name]['incremental_potential_J']-reports[name]['last']['incremental_potential_J'])
            curvature_results[name]['original_recorded_objective_difference_J']=close_error
        write(output/'curvature.json',curvature_results)
        comparisons={}
        if 'B' in states and 'C' in states:
            comparisons={key:float((getattr(states['B'],key)-getattr(states['C'],key)).abs().max()) for key in ['positions','velocities','auxiliary_positions','auxiliary_velocities','handle_centers','handle_rotations']}
            comparisons['optimizer_q_max_abs_difference']=float((qpoints['B']-qpoints['C']).abs().max())
        summary={'completed':True,'variants':reports,'B_C_maximum_differences':comparisons,'elapsed_s':time.monotonic()-started,
            'original_failed_cell_preserved':True,'original_failure_requalified':False,'full_trajectory_qualified':False,
            'curvature_scope':'24-step symmetric Lanczos Ritz diagnostics, not certified global spectra; objective and gradient unchanged',
            'physics_changed':False,'GPU_used':False,'CPU_threads':2}
        write(output/'summary.json',summary);write(output/'execution.json',{'terminal':True,'completed':True,'pid':os.getpid(),'elapsed_s':summary['elapsed_s']})
        return summary
    except BaseException as error:
        write(output/'failure.json',{'exception':type(error).__name__,'message':str(error),'reports':reports})
        write(output/'execution.json',{'terminal':True,'completed':False,'pid':os.getpid()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    args=p.parse_args();run(args.protocol,args.protocol_sha256)
