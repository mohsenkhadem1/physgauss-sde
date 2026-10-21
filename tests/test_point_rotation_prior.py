"""NEW cohort inputs/controller tests, with no monkeypatches or physics runs.

Configured private draws are generated but NEVER reopened or analyzed. Tiny
artificial states exercise full-state retention independently of cohort truth.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pytest

from experiments import prepare_point_rotation_prior as inputs
from experiments import run_point_rotation_prior as controller


@pytest.fixture(scope='module')
def generated():
    with tempfile.TemporaryDirectory(prefix='rotation-input-test-',dir=inputs.ROOT/'runs/operator-dev') as temp:
        directory=Path(temp)
        output=(directory/'inputs-point-rotation-prior-v02').relative_to(inputs.ROOT).as_posix()
        finite=(directory/'responses-point-rotation-prior-v02').relative_to(inputs.ROOT).as_posix()
        result=inputs.build(output,finite)
        yield directory,inputs.worker.resolve(output),finite,result


def test_exact_fresh_seed_roles_and_zero_capture_launch(generated):
    _,directory,finite,result=generated
    plan=json.loads((directory/'cohort-plan.json').read_text())
    assert inputs.SEED_BASE==2026120100
    assert [r['seed'] for r in plan['objects']]==list(range(2026120100,2026120148))
    assert [r['role'] for r in plan['objects']]==['CALIBRATION']*16+['RESERVED_EVALUATION']*32
    assert plan['cohort_version']=='point-rotation-prior-v02'
    assert plan['sampling']['coordinates_per_state']==12
    assert plan['sampling']['inactive_nuisances_retained'] is True
    assert plan['sampling']['response_based_selection'] is False
    assert len(plan['objects'])==48 and len(plan['cells'])==96
    assert not inputs.worker.resolve(finite).exists()
    assert result['finite_runs_launched']==0 and result['new_physics_launched'] is False
    assert plan['source_sha256']['experiments/campaign_hypothesis_dynamics.py']==inputs.WORKER_SHA
    assert 'experiments/prepare_point_rotation_prior.py' in plan['source_sha256']
    assert 'experiments/run_point_rotation_prior.py' in plan['source_sha256']


def test_complete12D_and_nuisance_retention_on_artificial_state():
    state=SimpleNamespace(coordinates=np.arange(12,dtype=float)[None],
        boundary_types=np.array([0]),inactive_force_nuisance=np.array([[31.,41.,59.]]),
        inactive_support_nuisance=np.array([[2,3]]))
    payload=inputs.retained_payload(inputs.identities()[0],state,{'draws':4},'prior-hash')
    assert payload['complete_state_dimension']==12
    assert payload['state']==inputs.plain(vars(state))
    assert payload['state']['inactive_force_nuisance']==[[31.,41.,59.]]
    with pytest.raises(ValueError,match='Complete12D'):
        inputs.retained_payload(inputs.identities()[0],SimpleNamespace(coordinates=np.zeros((1,4))),{},'hash')


def test_public_roles_private_permissions_and_no_implicit_reserved_release(generated):
    _,directory,_,_=generated
    public=json.loads((directory/'public/manifest.json').read_text())
    assert public['true_material_or_support_location_present'] is False
    assert not any(k in public for k in ['objects','role','seed','realized_hypothesis','retained_state'])
    assert inputs.worker.sha(directory/'public/acquired-queries.json')==inputs.ACQUIRED_SHA
    assert inputs.worker.sha(directory/'public/held-queries.json')==inputs.HELD_SHA
    plan=json.loads((directory/'cohort-plan.json').read_text())
    assert len(inputs.role_cases(plan,'CALIBRATION'))==32
    with pytest.raises(ValueError,match='Reserved evaluation'):
        inputs.role_cases(plan,'RESERVED_EVALUATION')
    assert all(p.stat().st_mode&0o077==0 for p in (directory/'private-truth').rglob('*.json'))
    assert all((directory/'private-truth'/role).stat().st_mode&0o077==0 for role in inputs.ROLES)
    for role in inputs.ROLES:
        with pytest.raises(ValueError,match='metadata only'):
            controller.read(directory/'private-truth'/role/'point_prior_000/retained-state.json')
    with pytest.raises(ValueError,match='metadata only'):
        controller.read(directory/'fake-trajectory.npz')


def test_controller_full96_preflight_and_metadata_identity_tamper(generated):
    _,directory,_,result=generated
    plan=controller.validate_plan(result['plan']['path'],result['plan']['sha256'])
    assert len(plan['cells'])==96
    assert plan['resources']==dict(maximum_workers=8,threads_per_worker=2,gpu_used=False,
                                  memory_reserve_GiB=8,memory_per_worker_GiB=1)
    plan['objects'][16]['seed']-=1
    p=directory/'tampered-plan.json';inputs.write(p,plan)
    with pytest.raises(ValueError,match='Seed/object/role'):
        controller.validate_plan(p.relative_to(inputs.ROOT).as_posix(),inputs.worker.sha(p))


def test_failure_stops_new_admission_owned_workers_drain_and_queued_slots_persist():
    records=[dict(id=f'cell{i}',state='queued',completed=False) for i in range(96)]
    assert len(controller.eligible_launches(records,stopped=False,live_count=0,capacity=8))==8
    assert len(controller.eligible_launches(records,stopped=False,live_count=7,capacity=8))==1
    assert controller.eligible_launches(records,stopped=True,live_count=7,capacity=8)==[]
    with tempfile.TemporaryDirectory(prefix='rotation-controller-test-',dir=inputs.ROOT/'runs/operator-dev') as temp:
        base=Path(temp)
        class Process:
            def __init__(self,code): self.code=code;self.waited=0
            def poll(self): return self.code
            def wait(self): self.waited+=1;return self.code
        failed,running=Process(1),Process(None)
        first={**records[0], 'output':(base/'first').relative_to(inputs.ROOT).as_posix()}
        second={**records[1], 'output':(base/'second').relative_to(inputs.ROOT).as_posix()}
        first_log=(base/'first.log').open('x');second_log=(base/'second.log').open('x')
        live={900000001:(failed,first,first_log),900000002:(running,second,second_log)}
        stopped=controller.poll_finished(live,False)
        assert stopped and first['state']=='failed' and failed.waited==1 and first_log.closed
        assert list(live)==[900000002] and running.waited==0 and not second_log.closed
        assert controller.eligible_launches(records,stopped=stopped,live_count=len(live),capacity=8)==[]
        # Owned inflight work finishes and is reaped; no worker kill/restart.
        running.code=1
        assert controller.poll_finished(live,stopped) is True
        assert live=={} and running.waited==1 and second_log.closed
        assert all(r['state']=='queued' for r in records) and len(records)==96


def test_fresh_paths_no_rebuild_or_retry(generated):
    _,directory,finite,_=generated
    with pytest.raises(ValueError,match='Fresh inputs/response prefix'):
        inputs.build(directory.relative_to(inputs.ROOT).as_posix(),finite)
