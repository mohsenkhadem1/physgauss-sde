import json,os,subprocess,sys
import numpy as np
import pytest
from experiments import run_surface_full_atlas_feasibility_v01 as run


def test_joint_48_channel_controls_all72_cases_and_bound():
    u=np.linspace(0,1,32);P=np.tile((40e-6*u)[:,None],(1,24));D={l:P*(1+i*.1)for i,l in enumerate(run.ACTIONS)}
    exact=run.joint_likelihood_checks(P,P,D,D,truth_indices=[0,1,3,4],seeds=[190,191,192,193])
    assert exact['passed']and exact['all72_joint_cases_retained']and len(exact['cases'])==72
    assert max(r['maximum_allpoint_loglikelihood_error']for r in exact['cases'])==0.
    approximate={l:v+2e-6 for l,v in D.items()}
    bad=run.joint_likelihood_checks(P,P+2e-6,D,approximate,truth_indices=[0,1,3,4],seeds=[190,191,192,193])
    assert not bad['passed']and len(bad['cases'])==72
    assert all(len(r['pointwise_loglikelihood_error'])==32 for r in bad['cases'])


def test_real_short_lived_children_100_exit_races_are_reaped():
    for _ in range(100):
        child=subprocess.Popen(['/bin/true'],start_new_session=True)
        try:
            while child.poll()is None:assert run.owned_RSS(child)>=0
        finally:
            assert child.wait(timeout=2)==0
        run.prefix.absent(child.pid)


def test_bounded_rss_grace_handles_transient_live_exit_status():
    class Child:
        pid=123
        def poll(self):return None
    states=iter(['State:\tR (running)\n','State:\tR (running)\n','State:\tZ (zombie)\n'])
    assert run.owned_RSS(Child(),read_status=lambda:next(states),grace_s=.1,pause_s=.001)==0
    with pytest.raises(RuntimeError,match='unresolved RSS'):run.owned_RSS(Child(),read_status=lambda:'State:\tR\n',grace_s=.005,pause_s=.001)


def test_all16_generated_worker_guards_and_controller():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for slot in range(16):
        r=subprocess.run(run.worker_command(slot,'synthetic',guard_only=True),cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
        assert r.returncode==0,r.stderr
    r=subprocess.run(['taskset','-c','16',sys.executable,'-m','experiments.run_surface_full_atlas_feasibility_v01','controller-guard'],cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr


def test_worker_end_to_end_toy_all10labels_prefix_reused_only(monkeypatch,tmp_path):
    import experiments.campaign_surface_p2_static_v02 as physics
    import experiments.campaign_surface_p2_force_v03 as force
    labels=['normal_x090','paired_axial_x090','normal_x135','paired_axial_x135','normal_x180','paired_axial_x180','normal_x225','paired_axial_x225','normal_x270','paired_axial_x270']
    origins={i:dict(raw=dict(path='synthetic-not-opened',sha256='rawtoy'),protocol_SHA256='originaltoy')for i in range(113)}
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'record',lambda p:dict(path=str(p),sha256=run.sha(p)));monkeypatch.setattr(run,'guard',lambda *a:None);monkeypatch.setattr(run,'validate',lambda _:(dict(labels=labels,points_unit_LOCAL=np.zeros((113,2)).tolist()),origins))
    monkeypatch.setattr(physics,'SparseStaticForward',lambda n:object());monkeypatch.setattr(physics,'point_query_matrix',lambda *a:None);monkeypatch.setattr(force,'load_catalog',lambda n:{l:None for l in labels})
    calls=[]
    def fake_solver(fwd,forces,Q,unit):
        calls.append(1)
        for label in labels:
            if label==run.PREFIX:continue
            yield label,dict(unit=np.asarray(unit),response_m=np.zeros(24),linear_response_m=np.zeros(24),delta_m=np.zeros(24)),dict(passed=True)
    monkeypatch.setattr(run,'solve_unit',fake_solver)
    result=run.worker(0,'synthetic')
    assert result['completed']and result['passed']and len(calls)==8 and len(result['rows'])==80
    assert sum(r['reused']for r in result['rows'])==8
    assert len(list((tmp_path/'worker-0').glob('*-LOCAL.npz')))==72
    assert {r['label']for r in result['rows']}==set(labels)


def test_missing_release_no_actual_process(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'guard',lambda *a:None);monkeypatch.setattr(run,'validate',lambda _:(dict(source_SHA256={}),{}))
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k:pytest.fail('No actualstage'))
    with pytest.raises(FileNotFoundError):run.run('toy')
