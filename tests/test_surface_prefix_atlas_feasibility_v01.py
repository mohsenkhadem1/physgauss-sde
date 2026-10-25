import copy,json,os,subprocess
import numpy as np
import pytest
from experiments import run_surface_prefix_atlas_feasibility_v01 as run


def toy_design():
    # Separate synthetic noise streams; frozen actual control seeds never used.
    d=json.loads(run.DESIGN.read_text());d['observed_likelihood_control']['noise_seeds']=[9120,9121,9122,9123]
    return d


def polynomial(u):
    v=3e-6*(u[:,0]**2+.2*u[:,1]+u[:,0]*u[:,1])
    return np.broadcast_to(v[:,None],(len(u),24)).copy()


def toy_inputs():
    d=toy_design();points=run.point_design(d);table=polynomial(points[:81]).reshape(9,9,1,24)
    L=np.zeros((32,24));D=polynomial(points[81:])
    return d,table,L,D


def test_exact_membership_nested_axes_and_all113_once():
    d=toy_design();p=run.point_design(d)
    assert p.shape==(113,2)and len(np.unique(p,axis=0))==113
    assert sorted(i for slot in range(8)for i in range(slot,113,8))==list(range(113))
    for n,step in((3,4),(5,2),(9,1)):
        np.testing.assert_array_equal(p[:81].reshape(9,9,2)[::step,::step,0],np.broadcast_to(run.unit_axis(n)[:,None],(n,n)))
    bad=copy.deepcopy(d);bad['offgrid_design']['queries'][0]=[0.,0.]
    with pytest.raises(ValueError):run.point_design(bad)


def test_exact_residual_atlas_all_levels_all_queries_all12_likelihood_cases():
    d,t,L,D=toy_inputs();report,atlas=run.atlas_checks(d,t,L,D,fingerprint='synthetic')
    assert report['passed']and report['all32_queries_retained']and report['all12_likelihood_cases_retained']
    assert [r['n']for r in report['levels']]==[3,5,9]
    assert all(len(r['likelihood_cases'])==12 for r in report['levels'])
    assert max(r['maximum_coordinate_error_m']for r in report['levels'])<1e-18
    assert report['maximum_grid5_vs_grid9_coordinate_difference_m']<1e-18


def test_validation_keeps_failures_including_direct_likelihood_disagreement():
    d,t,L,D=toy_inputs();D+=2e-6
    report,_=run.atlas_checks(d,t,L,D,fingerprint='synthetic')
    assert not report['passed']and report['levels'][-1]['maximum_coordinate_error_m']>1e-7
    assert len(report['levels'][-1]['pointwise_maximum_coordinate_error_m'])==32
    assert max(c['maximum_near_tested_peak_loglikelihood_error']for c in report['levels'][-1]['likelihood_cases'])>.01


def test_all_generated_cpu_worker_guards_and_controller_guard():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for slot in range(8):
        result=subprocess.run(run.worker_command(slot,'not-an-actual-protocol',guard_only=True),cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
    result=subprocess.run(['taskset','-c','16',run.sys.executable,'-m','experiments.run_surface_prefix_atlas_feasibility_v01','controller-guard'],cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


def test_missing_release_refuses_before_worker_or_physics(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'guard',lambda *args:None);monkeypatch.setattr(run,'validate',lambda _:dict(source_SHA256={}))
    monkeypatch.setattr(run.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('No actual process permitted'))
    with pytest.raises(FileNotFoundError):run.run('synthetic')


def test_actual_raw_fields_never_read_when_a_point_failed(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'validate',lambda _:dict(points_unit_LOCAL=np.zeros((113,2)).tolist()))
    for slot in range(8):
        directory=tmp_path/f'worker-{slot}';directory.mkdir()
        run.write(directory/'summary.json',dict(completed=True,slot=slot,protocol_SHA256='toy',points=[dict(index=i,passed=i!=44)for i in range(slot,113,8)]))
    monkeypatch.setattr(run.np,'load',lambda *args,**kwargs:pytest.fail('No field reads for incomplete physics qualification'))
    result=run.summarize('toy')
    assert result['all113_retained']and result['physical_passed']==112 and result['failures']==[44]and not result['qualified']
