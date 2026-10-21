"""Preflight/helper tests only; no mechanics model constructed or step called."""
import numpy as np
import pytest
import torch
from experiments import campaign_solver_step_diagnosis as diagnostic


def test_saved_point_chords_and_reduced_coordinates_restored_without_initialization():
    shapes={'positions_m':(4,3),'velocities_point_chord_m_s':(4,3),'auxiliary_positions_m':(12,3),
        'auxiliary_velocities_point_chord_m_s':(12,3),'handle_centers_m':(2,3),'handle_rotations':(2,3,3),'reduced_position_coordinates':(12,)}
    rng=np.random.default_rng(8);arrays={k:rng.normal(size=v) for k,v in shapes.items()};arrays['time_s']=np.array(2.)
    state=diagnostic.restore_state(arrays)
    assert state.time_s==2. and 'point chords' in state.velocity_semantics
    for field,key in [('positions','positions_m'),('velocities','velocities_point_chord_m_s'),('auxiliary_velocities','auxiliary_velocities_point_chord_m_s'),('reduced_position_coordinates','reduced_position_coordinates')]:
        np.testing.assert_array_equal(getattr(state,field).numpy(),arrays[key])
    arrays['positions_m'][:]=0;assert state.positions.abs().sum()>0
    arrays['velocities_point_chord_m_s']=arrays['velocities_point_chord_m_s'].astype(np.float32)
    with pytest.raises(ValueError):diagnostic.restore_state(arrays)


def test_trace_capture_handles_success_and_cap_error_without_global_tracer_leak():
    def step(ok):
        q=torch.tensor([1.,2.],dtype=torch.float64);history=[{'iteration':0}]
        if not ok:raise RuntimeError('intentional cap')
        return 'success'
    for success in [False,True]:
        result,error,captured=diagnostic.step_capture(lambda:step(success),step.__code__)
        assert result==('success' if success else None)
        assert error==(None if success else 'intentional cap')
        np.testing.assert_array_equal(captured['q'],[1,2]);assert captured['history']==[{'iteration':0}]
        assert diagnostic.sys.gettrace() is None


def test_lanczos_recovers_known_quadratic_curvature_without_physics():
    eigen=torch.tensor([.05,.2,1.,2.],dtype=torch.float64)
    result=diagnostic.lanczos_probe(lambda x:eigen*x,4,steps=4)
    assert result['minimum_Ritz']==pytest.approx(.05,abs=2e-15)
    assert result['maximum_Ritz']==pytest.approx(2.,abs=2e-15)
    assert result['not_global_eigenvalue_bounds'] is True


def test_real_terminal_guard_reads_only_and_counts_launched_workers(monkeypatch):
    path='runs/operator-dev/authored-solver-step-diagnosis-v01/protocol-v02.json'
    protocol=diagnostic.read(diagnostic.resolve(path))
    monkeypatch.setattr(diagnostic.core,'NonlinearDynamics',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('No physical calls in preflight')))
    checked=diagnostic.check_protocol(path,diagnostic.sha(diagnostic.resolve(path)))
    assert checked==protocol and len(checked['terminal_guard']['verified_absent_pids'])==28
    assert all(isinstance(pid,int) for pid in checked['terminal_guard']['verified_absent_pids'])
