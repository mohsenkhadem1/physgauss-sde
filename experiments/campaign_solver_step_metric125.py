"""One restored failed-step diagnostic with private optimizer metric1.25P0."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types
import numpy as np
import torch
try:
    import campaign_solver_step_diagnosis as abc
except ImportError:
    from experiments import campaign_solver_step_diagnosis as abc


def run(protocol_path,protocol_sha):
    path=abc.resolve(protocol_path)
    if abc.sha(path)!=protocol_sha:raise ValueError('Metric1.25 protocol hash mismatch')
    protocol=abc.read(path)
    if (protocol['protocol']!='restored-failed-step-metric125-v0.1'
            or protocol['status']!='FROZEN_AUTHORIZED_ONE_STEP_ONLY'
            or protocol['metric_scale']!=1.25 or protocol['maximum_iterations']!=250):
        raise ValueError('Exact one-step metric1.25 scope required')
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):
        raise ValueError('Exactly two CPU threads required')
    for rel,digest in protocol['source_and_input_sha256'].items():
        if abc.sha(abc.resolve(rel))!=digest:raise ValueError('Frozen diagnostic dependency changed: '+rel)
    if protocol['source_and_input_sha256'].get(Path(__file__).resolve().relative_to(abc.ROOT).as_posix())!=abc.sha(__file__):
        raise ValueError('Diagnostic source not pinned')
    abc_protocol=abc.check_protocol(protocol['ABC_protocol']['path'],protocol['ABC_protocol']['sha256'])
    root=abc.resolve(abc.CASE);output=abc.resolve(protocol['fresh_output'])
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
    status={'terminal':False,'completed':False,'pid':os.getpid(),'gpu_used':False,'cpu_threads':2}
    abc.write(output/'execution.json',status);(output/'protocol.json').write_bytes(path.read_bytes())
    try:
        physical=abc.read(root/'private-provenance/physical-inputs.json')
        source_protocol=abc.read(root/'protocol.json');failure=abc.read(root/'failure.json')
        with np.load(root/'private-provenance/static-reference.npz',allow_pickle=False) as raw:arrays={k:raw[k].copy() for k in raw.files}
        with np.load(root/'last-accepted-state.npz',allow_pickle=False) as raw:saved={k:raw[k].copy() for k in raw.files}
        model=abc.assemble_model(physical,arrays,source_protocol['cell']['translation_m']);h=1/120
        rest=model.initial_state();P0=model.preconditioner(rest,h).clone()
        state=abc.restore_state(saved);model.validate_state(state)
        original=model.preconditioner
        def scaled(this,value,dt):return original(value,dt)*math.sqrt(1.25)
        model.preconditioner=types.MethodType(scaled,model)
        np.testing.assert_array_equal(original(state,h).numpy(),P0.numpy())
        F=torch.tensor(failure['step_request']['held_handle_forces_world_N'],dtype=torch.float64)
        if bool(F.any()):raise ValueError('Exact failed zero force required')
        step_started=time.monotonic()
        result,error,captured=abc.step_capture(lambda:model.step(state,h,handle_forces=F,
            relative_tolerance=1e-7,absolute_tolerance_N=1e-10,maximum_iterations=250),abc.core.NonlinearDynamics.step.__code__)
        step_seconds=time.monotonic()-step_started;history=captured.get('history',[])
        abc.write(output/'history.json',history)
        if captured.get('q') is not None:np.savez_compressed(output/'optimizer-q.npz',q=captured['q'].numpy())
        if result is None:raise RuntimeError('Metric1.25 step failed: '+str(error))
        new,ledger=result;metrics=abc.old.endpoint_metrics(model,new,state)
        abc.old.endpoint_guard(model,new,metrics,ledger)
        abc.old.save_state(output/'state.npz',new);abc.write(output/'ledger.json',ledger)
        differences={};physical_differences={}
        fields={'positions':'positions_m','velocities':'velocities_point_chord_m_s','auxiliary_positions':'auxiliary_positions_m',
            'auxiliary_velocities':'auxiliary_velocities_point_chord_m_s','handle_centers':'handle_centers_m','handle_rotations':'handle_rotations',
            'reduced_position_coordinates':'reduced_position_coordinates'}
        for name,reference_root in protocol['references'].items():
            reference_root=abc.resolve(reference_root)
            with np.load(reference_root/'state.npz',allow_pickle=False) as raw:
                differences[name]={field:float(np.max(abs(getattr(new,field).numpy()-raw[key]))) for field,key in fields.items()}
            base=abc.read(reference_root/'ledger.json')
            physical_differences[name]={'objective_J':abs(history[-1]['incremental_potential_J']-base['nonlinear_history'][-1]['incremental_potential_J']),
                'minimum_J':abs(ledger['minimum_J']-base['minimum_J']),
                'handle_work_J':abs(ledger['held_handle_work_J']-base['held_handle_work_J']),
                'support_work_J':abs(ledger['support_work_J']-base['support_work_J'])}
        limits=protocol['reference_agreement_limits']
        agreement=all(v<=limits['mixed_unit_state_max_abs'] for row in differences.values() for v in row.values())
        agreement=agreement and all(v<=limits[key] for row in physical_differences.values() for key,v in row.items())
        if any(abc.sha(abc.resolve(rel))!=digest for rel,digest in protocol['source_and_input_sha256'].items()):raise ValueError('Frozen dependency changed during diagnostic')
        summary={'completed':True,'metric_scale':1.25,'iterations':len(history)-1,'residual_N':history[-1]['scaled_generalized_residual_norm_N'],
            'residual_tolerance_N':ledger['residual_tolerance_N'],'step_seconds':step_seconds,'elapsed_s':time.monotonic()-started,
            'original_P0_factor_sha256':hashlib.sha256(P0.numpy().tobytes()).hexdigest(),
            'reference_state_differences':differences,'reference_physical_differences':physical_differences,
            'reference_agreement_passed':agreement,'geometry':metrics,'original_failure_requalified':False,
            'full_trajectory_qualified':False,'physical_objective_gradient_linesearch_unchanged':True,
            'comparison_caveat':'State limit is mixed-unit numerical regression only; physical differences carry explicit units.'}
        abc.write(output/'summary.json',summary)
        status.update(terminal=True,completed=True,elapsed_s=summary['elapsed_s']);abc.write(output/'execution.json',status)
        print(json.dumps(summary),flush=True)
        if not agreement:raise RuntimeError('Predeclared B/D agreement failed')
        return summary
    except BaseException as error:
        abc.write(output/'failure.json',{'exception':type(error).__name__,'message':str(error),'original_failure_requalified':False})
        status.update(terminal=True,completed=False);abc.write(output/'execution.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    a=p.parse_args();run(a.protocol,a.protocol_sha256)
