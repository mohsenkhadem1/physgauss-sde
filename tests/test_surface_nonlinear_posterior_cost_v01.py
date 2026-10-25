import builtins,inspect,symtable,json
import numpy as np
import pytest
from experiments import run_surface_nonlinear_posterior_cost_v01 as m


def fake_final():
    return dict(log_evidence=1.,unit_mean=np.array([.5,.5]),unit_CDF=np.array([np.linspace(0,1,33)]*2),unit_quantiles=np.array([[.05,.5,.95]]*2),predictive_mean=np.zeros(240),predictive_SD=np.ones(240)*1e-3)


def test_absolute_witnesses_reject_relative_SD_loophole_and_missing_or_nonfinite_values():
    row=dict(passed=True,coarse=256,fine=512,maximum_predictive_mean_difference_m=3e-6,maximum_predictive_SD_difference_m=1e-6)
    result=dict(qualified=True,diagnostics=dict(successive={a:[row.copy()]for a in('inner','outer')}))
    assert not m.absolute_engine_witnesses(result)['passed']
    for a in('inner','outer'):result['diagnostics']['successive'][a][-1]['maximum_predictive_mean_difference_m']=1e-6
    assert m.absolute_engine_witnesses(result)['passed']
    result['diagnostics']['successive']['outer'][-1]['maximum_predictive_SD_difference_m']=float('nan');assert not m.absolute_engine_witnesses(result)['passed']
    assert not m.absolute_engine_witnesses(dict(qualified=True))['passed']


def test_cross_atlas_keeps_all_intrinsic_and_strict_absolute_target_checks():
    a=fake_final();b=fake_final();b['predictive_mean'][17]=3e-6
    check=m.cross_atlas_checks(a,b);assert check['original_checks']['passed']and not check['passed']
    b=fake_final();b['unit_mean'][0]+=.002;assert not m.cross_atlas_checks(a,b)['passed']
    b=fake_final();b['unit_CDF'][0,8]+=.006;assert not m.cross_atlas_checks(a,b)['passed']
    assert m.cross_atlas_checks(a,fake_final())['passed']


def test_prefix_only_or_unsealed_atlas_stops_before_truth_arrays_or_rng(monkeypatch,tmp_path):
    from experiments import run_surface_full_atlas_feasibility_v01 as full
    monkeypatch.setattr(full,'LOCAL',tmp_path);f=tmp_path/'summary-LOCAL.json';f.write_text(json.dumps(dict(completed=True,qualified=True,protocol_SHA256='x')))
    monkeypatch.setattr(full,'validate',lambda _:({'labels':['normal_x180']},None));(tmp_path/'execution.json').write_text('{}')
    def forbidden(*a,**k):raise AssertionError('Premature arrays/noise')
    monkeypatch.setattr(m.np,'load',forbidden);monkeypatch.setattr(m.np.random,'default_rng',forbidden)
    with pytest.raises(ValueError,match='full10'):m.full_atlas_gate(f,m.sha(f))


def test_exact_memo_reuses_scalar_logratio_without_snapping_or_parameter_interpolation():
    from experiments.campaign_surface_conditional_amplitude_v01 import ExactRatioMemo
    calls=[]
    def callback(z,labels):calls.append(z.copy());return np.broadcast_to(z[:,None,None],(len(z),len(labels),24)).copy()
    memo=ExactRatioMemo(callback,model_fingerprint='a'*64,maximum_bytes=256*1024**2,minimum_rows=1)
    z=np.array([-7.]);first=memo(z,['normal_x180']);second=memo(z.copy(),['normal_x180']);np.testing.assert_array_equal(first,second);assert len(calls)==1 and memo.hits==1
    memo(np.nextafter(z,np.inf),['normal_x180']);assert len(calls)==2 and memo.allocated_bytes<=256*1024**2


def test_worker_and_controller_names_resolve_and_recipe_remains_source_only():
    assert m.UNIT==(.5,.5)and m.NOISE_SEED==2026102591 and m.RESOURCES['wall_cap_s']==1800 and not m.RESOURCES['GPU']
    for fn in [m.worker,m.run,m.prepare,m.validate,m.full_atlas_gate,m.truth_gate,m.guard,m.release_gate,m.completed_cost_gate,m.phase_completion]:
        table=symtable.symtable(inspect.getsource(fn),'posterior-cost-draft','exec')
        def check(t):
            for s in t.get_symbols():
                if s.is_referenced()and s.is_global():assert s.get_name()in vars(m)or hasattr(builtins,s.get_name()),s.get_name()
            for child in t.get_children():check(child)
        check(table)


def test_controller_failure_cannot_be_relabelled_complete_by_a_successful_worker():
    worker=dict(completed=True,passed=True)
    for e in [dict(failure='monitor error',worker_exit_code=0,cleanup_exit_code=0),dict(worker_exit_code=1,cleanup_exit_code=1),dict(cleanup_exit_code=0)]:
        assert m.phase_completion(e,worker)==dict(completed=False,qualified=False)
    assert m.phase_completion(dict(worker_exit_code=0,cleanup_exit_code=0),worker)==dict(completed=True,qualified=True)


def test_one_hundred_real_short_child_exit_races_use_verified_bounded_RSS_reader():
    import subprocess,sys
    for _ in range(100):
        child=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        try:
            while child.poll()is None:assert m.owned_RSS(child)>=0
            assert child.wait(timeout=2)==0
        finally:
            if child.poll()is None:m.cost.cleanup(child)


def test_future_cohort_cost_gate_rejects_controller_failure_before_arrays(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'validate',lambda _:({},None));monkeypatch.setattr(m,'release_gate',lambda *args:None)
    (tmp_path/'execution.json').write_text(json.dumps(dict(terminal=True,failure='RSS race',worker_exit_code=0,cleanup_exit_code=0)))
    (tmp_path/'worker-result-LOCAL.json').write_text(json.dumps(dict(completed=True,passed=True)))
    def forbidden(*a,**k):raise AssertionError('Premature truth/noise access')
    monkeypatch.setattr(m.np,'load',forbidden);monkeypatch.setattr(m.np.random,'default_rng',forbidden)
    with pytest.raises(ValueError,match='Completed qualified'):m.completed_cost_gate('unavailable')


def test_delegated_CPU_guard_configuration_is_checked_before_any_model_call(monkeypatch):
    monkeypatch.setitem(m.RESOURCES,'worker_RSS_cap_bytes',8*1024**3)
    with pytest.raises(ValueError,match='configuration mismatch'):m.guard('worker')
