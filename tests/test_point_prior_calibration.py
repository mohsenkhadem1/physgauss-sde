"""Prior/role/controller checks without mechanics or reserved-response access."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
from experiments import prepare_point_prior_calibration as inputs
from experiments import run_point_prior_calibration as controller


@pytest.fixture(scope='module')
def generated():
    with tempfile.TemporaryDirectory(prefix='point-cohort-test-',dir=inputs.ROOT/'runs/operator-dev') as temp:
        directory=Path(temp);output=(directory/'inputs').relative_to(inputs.ROOT).as_posix()
        finite=(directory/'no-responses').relative_to(inputs.ROOT).as_posix()
        rng=np.random.default_rng;seeds=[]
        def observed(seed):seeds.append(seed);return rng(seed)
        def forbidden(*a,**k):raise AssertionError('Inputs must not construct/run mechanics')
        with patch.object(inputs.np.random,'default_rng',side_effect=observed), \
                patch.object(inputs.worker,'run',side_effect=forbidden), \
                patch.object(inputs.worker,'build_model',side_effect=forbidden):
            result=inputs.build(output,finite)
        yield directory,directory/'inputs',finite,result,seeds


def test_exact_independent_seeds_roles_and_complete_calibration_state(generated):
    _,directory,finite,result,seeds=generated
    assert seeds==list(range(2026112000,2026112048))
    plan=json.loads((directory/'cohort-plan.json').read_text())
    assert len(plan['objects'])==48 and len(plan['cells'])==96
    assert [o['role'] for o in plan['objects']]==['CALIBRATION']*16+['RESERVED_EVALUATION']*32
    # Only CALIBRATION truth is reopened by these tests. Reserved remains closed.
    public=json.loads((directory/'public/H0.json').read_text());prior=inputs.public_inputs(public)[0]
    expected,sampling=inputs.conditioned_BC_sample(prior,0,np.random.default_rng(2026112000))
    saved=json.loads((directory/'private-truth/CALIBRATION/point_prior_000/retained-state.json').read_text())
    if saved['state']!=inputs.plain(vars(expected)) or saved['sampling']!=sampling:raise AssertionError('Seeded full prior state mismatch')
    assert np.asarray(saved['state']['coordinates']).shape==(1,12)
    assert np.isfinite(prior.log_prob(expected)).all()
    assert not inputs.worker.resolve(finite).exists() and result['finite_runs_launched']==0


def test_public_separation_and_reserved_read_barrier(generated):
    _,directory,_,_,_=generated
    public=json.loads((directory/'public/manifest.json').read_text())
    assert not any(k in public for k in ['objects','seed','role','truth_spec','realized_hypothesis','retained_state'])
    assert public['true_material_or_support_location_present'] is False
    assert inputs.worker.sha(directory/'public/acquired-queries.json')==inputs.ACQUIRED_SHA
    assert inputs.worker.sha(directory/'public/held-queries.json')==inputs.HELD_SHA
    plan=json.loads((directory/'cohort-plan.json').read_text())
    assert len(inputs.role_cases(plan,'CALIBRATION'))==32
    with pytest.raises(ValueError,match='Reserved evaluation'):inputs.role_cases(plan,'RESERVED_EVALUATION')
    # Stat mode checks do not open any evaluation truth.
    assert all((directory/'private-truth'/r).stat().st_mode&0o077==0 for r in inputs.ROLES)
    assert all(p.stat().st_mode&0o077==0 for p in (directory/'private-truth').rglob('*.json'))


def test_controller_validates_all96_without_parsing_truth_or_response(generated):
    _,directory,_,result,_=generated
    original=controller.read
    def guarded(path):
        if 'private-truth' in str(path) or str(path).endswith('trajectory.npz'):raise AssertionError('Reserved/truth response read forbidden')
        return original(path)
    with patch.object(controller,'read',side_effect=guarded):
        plan=controller.validate_plan(result['plan']['path'],result['plan']['sha256'])
    assert len(plan['cells'])==96
    altered=json.loads((directory/'cohort-plan.json').read_text());altered['cells'][32]['role']='CALIBRATION'
    path=directory/'tampered-plan.json';inputs.write(path,altered)
    with pytest.raises(ValueError,match='Canonical cell'):
        controller.validate_plan(path.relative_to(inputs.ROOT).as_posix(),inputs.worker.sha(path))


def test_stop_on_first_failure_drains_owned_workers_retains_every_queued_record(tmp_path):
    records=[dict(id=f'c{i}',output=f'cells/c{i}',role='CALIBRATION' if i<32 else 'RESERVED_EVALUATION',
                  protocol=dict(path=f'p{i}',sha256='digest')) for i in range(96)]
    plan=dict(output_prefix='capture',cells=records,source_sha256={'source':'hash'},
              resources=dict(maximum_workers=8,threads_per_worker=2,gpu_used=False,memory_reserve_GiB=8,memory_per_worker_GiB=1))
    launched=[];waited=[]
    class Fake:
        def __init__(self,*a,**k):self.pid=900000+len(launched);launched.append(self.pid)
        def poll(self):return 1 if self.pid==900000 else 0
        def wait(self):waited.append(self.pid);return self.poll()
    with patch.object(controller,'validate_plan',return_value=plan), \
            patch.object(controller.inputs,'sources',return_value=plan['source_sha256']), \
            patch.object(controller.worker,'resolve',side_effect=lambda p:tmp_path/p), \
            patch.object(controller.worker,'validate_protocol',return_value={}), \
            patch.object(controller,'resource_sample',return_value={}), \
            patch.object(controller,'resource_capacity',return_value=8), \
            patch.object(controller.subprocess,'Popen',Fake),patch.object(controller,'completed',return_value=True), \
            patch.object(controller.time,'sleep'):
        result=controller.run('plan','sha')
        with pytest.raises(ValueError,match='Fresh response prefix'):controller.run('plan','sha')
    assert len(launched)==8 and len(waited)==8
    assert result['terminal'] and result['stopped_on_failure'] and not result['completed']
    assert len(result['records'])==96
    assert [r['state'] for r in result['records']].count('failed')==1
    assert [r['state'] for r in result['records']].count('completed')==7
    assert [r['state'] for r in result['records']].count('queued')==88
