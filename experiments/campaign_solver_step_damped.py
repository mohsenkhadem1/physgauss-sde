"""One declared D diagnostic: half original cached-preconditioner direction.

Only a private model instance's returned Cholesky factor is scaled sqrt(2).
Physical objective, gradient, residuals, line search and ledgers stay original.
"""
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

SOURCE=Path(__file__).resolve()


def install_half_direction(model):
    """Instance-only optimizer metric 2P0: direction approximately .5 P0^-1(-g)."""
    original=model.preconditioner
    def scaled(this,state,h):
        return original(state,h)*math.sqrt(2.)
    model.preconditioner=types.MethodType(scaled,model)
    return original


def run(protocol_path,protocol_sha):
    path=abc.resolve(protocol_path)
    if abc.sha(path)!=protocol_sha:raise ValueError('D protocol hash mismatch')
    protocol=abc.read(path)
    if (protocol['protocol']!='authored-force-failed-step-D-half-direction-v0.1'
            or protocol['status']!='FROZEN_AUTHORIZED_ONE_STEP_ONLY'
            or protocol['metric_scale']!=2. or protocol['maximum_iterations']!=250):raise ValueError('Exact D-only scope required')
    for rel,digest in protocol['source_and_input_sha256'].items():
        if abc.sha(abc.resolve(rel))!=digest:raise ValueError('Frozen D dependency changed: '+rel)
    if protocol['source_and_input_sha256'].get(SOURCE.relative_to(abc.ROOT).as_posix())!=abc.sha(SOURCE):raise ValueError('D source not frozen')
    abc_protocol=abc.check_protocol(protocol['ABC_protocol']['path'],protocol['ABC_protocol']['sha256'])
    old_output=abc.resolve(abc_protocol['fresh_output'])
    if not abc.read(old_output/'execution.json').get('completed'):raise ValueError('Completed immutable ABC proof required')
    if Path('/proc',str(abc.read(old_output/'execution.json')['pid'])).exists():raise ValueError('ABC worker must be absent')
    root=abc.resolve(abc.CASE);output=abc.resolve(protocol['fresh_output']);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.monotonic();status={'terminal':False,'completed':False,'pid':os.getpid(),'protocol_sha256':protocol_sha}
    abc.write(output/'execution.json',status);(output/'protocol.json').write_bytes(path.read_bytes())
    try:
        physical=abc.read(root/'private-provenance/physical-inputs.json');source_protocol=abc.read(root/'protocol.json');failure=abc.read(root/'failure.json')
        with np.load(root/'private-provenance/static-reference.npz',allow_pickle=False) as raw:arrays={key:raw[key].copy() for key in raw.files}
        with np.load(root/'last-accepted-state.npz',allow_pickle=False) as raw:saved={key:raw[key].copy() for key in raw.files}
        model=abc.assemble_model(physical,arrays,source_protocol['cell']['translation_m']);h=1/120
        rest=model.initial_state();L0=model.preconditioner(rest,h).clone();state=abc.restore_state(saved);model.validate_state(state)
        original=install_half_direction(model);scaled=model.preconditioner(state,h)
        np.testing.assert_array_equal(original(state,h).numpy(),L0.numpy())
        F=torch.tensor(failure['step_request']['held_handle_forces_world_N'],dtype=torch.float64)
        if bool(F.any()):raise ValueError('Exact failed zero force required')
        result,error,captured=abc.step_capture(lambda:model.step(state,h,handle_forces=F,
            relative_tolerance=1e-7,absolute_tolerance_N=1e-10,maximum_iterations=250),abc.core.NonlinearDynamics.step.__code__)
        history=captured.get('history',[]);abc.write(output/'history.json',history)
        if captured.get('q') is not None:np.savez_compressed(output/'optimizer-q.npz',q=captured['q'].numpy())
        if result is None:raise RuntimeError('D one-step solve failed: '+str(error))
        new,ledger=result;abc.old.save_state(output/'state.npz',new);abc.write(output/'ledger.json',ledger)
        with np.load(old_output/'B/state.npz',allow_pickle=False) as raw:reference={key:raw[key].copy() for key in raw.files}
        fields={'positions':'positions_m','velocities':'velocities_point_chord_m_s','auxiliary_positions':'auxiliary_positions_m',
            'auxiliary_velocities':'auxiliary_velocities_point_chord_m_s','handle_centers':'handle_centers_m','handle_rotations':'handle_rotations',
            'reduced_position_coordinates':'reduced_position_coordinates'}
        differences={field:float(np.max(abs(getattr(new,field).numpy()-reference[key]))) for field,key in fields.items()}
        with np.load(old_output/'B/optimizer-q.npz',allow_pickle=False) as raw:differences['optimizer_q']=float(np.max(abs(captured['q'].numpy()-raw['q'])))
        base_ledger=abc.read(old_output/'B/ledger.json')
        physical={'objective_difference_J':abs(history[-1]['incremental_potential_J']-base_ledger['nonlinear_history'][-1]['incremental_potential_J']),
            'minimum_J_difference':abs(ledger['minimum_J']-base_ledger['minimum_J']),
            'handle_work_difference_J':abs(ledger['held_handle_work_J']-base_ledger['held_handle_work_J']),
            'support_work_difference_J':abs(ledger['support_work_J']-base_ledger['support_work_J']),
            'elastic_energy_difference_J':abs(ledger['new_invariants']['elastic_energy_J']-base_ledger['new_invariants']['elastic_energy_J']),
            'kinetic_energy_difference_J':abs(ledger['new_invariants']['chord_kinetic_energy_J']-base_ledger['new_invariants']['chord_kinetic_energy_J'])}
        limits=protocol['B_agreement_limits']
        agreement=all(value<=limits['state_array_max_abs'] for value in differences.values()) and physical['objective_difference_J']<=limits['objective_J'] and physical['minimum_J_difference']<=limits['minimum_J'] and physical['handle_work_difference_J']<=limits['work_J'] and physical['support_work_difference_J']<=limits['work_J']
        summary={'completed':True,'variant':'D','iterations':len(history)-1,'last':history[-1],
            'elapsed_s':time.monotonic()-started,'optimizer_metric_scale':2.,'original_metric_effective_step_fraction':.5,
            'physical_objective_gradient_linesearch_unchanged':True,'original_P0_factor_sha256':hashlib.sha256(L0.numpy().tobytes()).hexdigest(),
            'scaled_factor_sha256':hashlib.sha256(scaled.numpy().tobytes()).hexdigest(),'B_fullstate_maximum_differences':differences,
            'B_physical_differences':physical,'B_reference_agreement_passed':agreement,'original_failure_requalified':False,
            'full_trajectory_qualified':False,'original_B_cap1000_is_diagnostic_only':True,'CPU_threads':2,'GPU_used':False}
        abc.write(output/'summary.json',summary);status.update(terminal=True,completed=True,elapsed_s=summary['elapsed_s']);abc.write(output/'execution.json',status)
        print(json.dumps(summary),flush=True)
        return summary
    except BaseException as error:
        abc.write(output/'failure.json',{'exception':type(error).__name__,'message':str(error),'original_failure_requalified':False})
        status.update(terminal=True,completed=False);abc.write(output/'execution.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    a=p.parse_args();run(a.protocol,a.protocol_sha256)
