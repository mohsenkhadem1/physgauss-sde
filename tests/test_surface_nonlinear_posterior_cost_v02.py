import builtins,inspect,json,symtable,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from experiments import run_surface_nonlinear_posterior_cost_v02 as m


def qualified_result():
    row=dict(passed=True,coarse=256,fine=512,log_evidence_difference=.001,maximum_CDF_difference=.001,maximum_unit_quantile_difference=.001,maximum_unit_mean_difference=.0001,maximum_predictive_mean_difference_m=1e-6,maximum_predictive_SD_difference_m=1e-6)
    adaptive=dict(passed=True,epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6,log_evidence_difference=.001,maximum_CDF_error=.001,adaptive_refinement_log_evidence_difference=.001,adaptive_refinement_maximum_CDF_difference=.001,maximum_QUADPACK_relative_error_estimates={'1e-06':1e-7,'1e-08':1e-9})
    return dict(qualified=True,diagnostics=dict(successive={a:[row.copy()]for a in('inner','outer')},independent_adaptive=adaptive))


def test_every_original_adaptive_gate_and_actual_absolute_witness_is_required():
    r=qualified_result();assert m.engine_gates(r)['passed']
    r['diagnostics']['successive']['inner'][-1]['maximum_predictive_mean_difference_m']=3e-6
    assert not m.engine_gates(r)['passed']
    r=qualified_result();r['diagnostics']['independent_adaptive']['maximum_CDF_error']=.006
    assert not m.engine_gates(r)['passed']
    r=qualified_result();del r['diagnostics']['independent_adaptive']['maximum_QUADPACK_relative_error_estimates']['1e-08']
    assert not m.engine_gates(r)['passed']
    r=qualified_result();r['diagnostics']['successive']['outer'][-1]['maximum_unit_quantile_difference']=float('nan')
    assert not m.engine_gates(r)['passed']


def test_cross_atlas_cannot_use_relative_SD_to_evade_absolute_2um():
    def final():return dict(log_evidence=1.,unit_mean=np.ones(2)*.5,unit_CDF=np.array([np.linspace(0,1,33)]*2),unit_quantiles=np.array([[.05,.5,.95]]*2),predictive_mean=np.zeros(240),predictive_SD=np.ones(240)*1e-3)
    a,b=final(),final();assert m.cross_atlas_checks(a,b)['passed']
    b['predictive_mean'][17]=3e-6
    c=m.cross_atlas_checks(a,b);assert c['original_checks']['passed']and not c['passed']


def test_release_is_required_before_archive_or_noise_access(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'guard',lambda _:None);monkeypatch.setattr(m,'validate',lambda _:({},{}))
    monkeypatch.setattr(m,'LOCAL',tmp_path)
    def forbidden(*a,**k):raise AssertionError('Premature archive/noise')
    monkeypatch.setattr(m.np,'load',forbidden);monkeypatch.setattr(m.np.random,'default_rng',forbidden)
    with pytest.raises(FileNotFoundError):m.worker('unreleased')
    src=inspect.getsource(m.worker)
    assert 'default_rng'not in src and 'direct_truth'not in src and "['observation_m']"in src


def test_origin_pins_exact_old_observation_without_drawing_or_loading(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m.old,'LOCAL',tmp_path/'v1');m.old.LOCAL.mkdir()
    monkeypatch.setattr(m,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=m.sha(p)))
    monkeypatch.setattr(m.old,'validate',lambda _:({'full_atlas':{},'source_SHA256':{}},{}));monkeypatch.setattr(m.old,'release_gate',lambda *a:None);monkeypatch.setattr(m.old,'absent',lambda _:None)
    def put(p,d):p.write_text(json.dumps(d));return m.record(p)
    release=put(m.old.LOCAL/'release.json',{});(m.old.LOCAL/'observation-LOCAL.npz').write_bytes(b'original saved byte stream')
    observation=m.record(m.old.LOCAL/'observation-LOCAL.npz')
    result=dict(completed=True,passed=False,protocol_SHA256=m.ORIGIN_SHA,release=release,observation=observation)
    result_record=put(m.old.LOCAL/'worker-result-LOCAL.json',result);cache=tmp_path/'cache.json';cache.write_bytes((m.old.LOCAL/'worker-result-LOCAL.json').read_bytes())
    put(m.old.LOCAL/'protocol.json',{})
    put(m.old.LOCAL/'execution.json',dict(terminal=True,completed=True,qualified=False,worker_exit_code=0,cleanup_exit_code=0,controller_pid=10,worker_pid=11,protocol_SHA256=m.ORIGIN_SHA,result=result_record,release=release,external_cache=m.record(cache)))
    raw=tmp_path/'diagnostic-raw.json';raw.write_text('{}');diag=tmp_path/'diagnostic.json'
    put(diag,dict(status='PARTIAL_DIAGNOSTIC_WITH_IDENTIFIED_CAUSE',diagnostic_completed=False,qualification=False,closure=dict(root_confirmed_PID_and_group_absent=True),protocol=dict(sha256='47009b52ecf7fd19f9ae55dcf7949f88ad37481940ea2a0011c666fbbf7ab576'),pid=12,raw_result_LOCAL=m.record(raw)))
    monkeypatch.setattr(m,'DIAGNOSTIC',diag);monkeypatch.setattr(m,'DIAGNOSTIC_SHA',m.sha(diag))
    def forbidden(*a,**k):raise AssertionError('Origin gate must be metadata only')
    monkeypatch.setattr(m.np,'load',forbidden);monkeypatch.setattr(m.np.random,'default_rng',forbidden)
    _,_,r=m.origin_gate();assert r['observation']==observation
    (m.old.LOCAL/'observation-LOCAL.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='saved observation'):m.origin_gate()


def test_successful_worker_cannot_hide_controller_or_cleanup_failure():
    for e in(dict(failure='RSS race',worker_exit_code=0,cleanup_exit_code=0),dict(worker_exit_code=0,cleanup_exit_code=1)):
        assert m.phase_completion(e,dict(completed=True,passed=True))==dict(completed=False,qualified=False)


def test_future_admission_stops_on_failed_closure_before_arrays(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'validate',lambda _:({},{}));monkeypatch.setattr(m,'release_gate',lambda *a:None)
    (tmp_path/'execution.json').write_text(json.dumps(dict(terminal=True,failure='controller failure',worker_exit_code=0,cleanup_exit_code=0)))
    (tmp_path/'worker-result-LOCAL.json').write_text(json.dumps(dict(completed=True,passed=True)))
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('Archive opened before closure'))
    with pytest.raises(ValueError,match='qualified v02'):m.completed_cost_gate('x')


def test_100_actual_child_exit_races_use_frozen_bounded_RSS_reader():
    for _ in range(100):
        child=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        try:
            while child.poll()is None:assert m.owned_RSS(child)>=0
            assert child.wait(timeout=2)==0
        finally:
            if child.poll()is None:m.old.cost.cleanup(child)


def test_all_worker_controller_and_guard_global_names_resolve():
    assert m.RESOURCES==m.old.RESOURCES and m.RESOURCES['wall_cap_s']==1800
    for fn in(m.worker,m.run,m.prepare,m.validate,m.origin_gate,m.guard,m.completed_cost_gate,m.engine_gates):
        table=symtable.symtable(inspect.getsource(fn),'costv02','exec')
        def check(t):
            for s in t.get_symbols():
                if s.is_referenced()and s.is_global():assert s.get_name()in vars(m)or hasattr(builtins,s.get_name()),s.get_name()
            for c in t.get_children():check(c)
        check(table)
