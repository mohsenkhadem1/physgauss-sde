import json,os,subprocess,sys
import numpy as np
import pytest
from experiments import run_clamped_surface_p2_qualification_v02 as s


def test_normal_reuse_operator_force_and_kernel_byte_proof():
    proof=s.verify_normal_reuse()
    assert set(proof)=={'2','4'}
    assert all(v['geometry_and_operators_byte_identical']for v in proof.values())
    assert s.central_gate() # checks completed immutable predecessor and absence


def test_reuse_refuses_changed_constitutive_kernel(monkeypatch):
    monkeypatch.setattr(s.linear.P2NonlinearReference,'evaluate',lambda *a,**k:None)
    with pytest.raises(ValueError,match='kernel mismatch'):s.verify_normal_reuse()


def test_all10_retained_even_if_one_fails_no_subset_admission():
    rows=[dict(case_index=c['index'],load=name,passed=True)for c in s.CASES for name in s.LOADS]
    rows[0]['passed']=False;d=s.retained_groups(rows)
    assert d['all10_retained']and d['qualified_groups']==9 and not d['all_qualified']
    with pytest.raises(ValueError):s.retained_groups(rows[1:])
    with pytest.raises(ValueError):s.retained_groups(rows[:-1]+[rows[0]])


def test_actual_worker_child_affinity_environment_guards():
    script="import os,json;from experiments import run_clamped_surface_p2_qualification_v02 as s;cpu=int(os.environ['QUALIFICATION_TEST_CPU']);os.sched_setaffinity(0,{cpu});s.guard(cpu);print(json.dumps({'affinity':sorted(os.sched_getaffinity(0))}))"
    env=dict(os.environ,PYTHONPATH=str(s.ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    processes=[]
    for cpu in s.RESOURCES['worker_CPUs']:
        childenv=dict(env,QUALIFICATION_TEST_CPU=str(cpu));processes.append((cpu,subprocess.Popen([sys.executable,'-c',script],env=childenv,cwd=s.ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)))
    for cpu,child in processes:
        out,err=child.communicate(timeout=30)
        assert child.returncode==0,err
        assert json.loads(out)['affinity']==[cpu]
    bad=dict(env,QUALIFICATION_TEST_CPU='4',OMP_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-c',script],env=bad,cwd=s.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'CPU/one-thread/noGPU guard'in result.stderr


def test_common_n8_positive_quadrature_exact_quartic_no_response_solves():
    fields=[]
    for n in (2,4,8):
        X=s.linear.geometry(n)[0];fields.append((n,.001*X*X))
    assert np.max(s.common_rms(fields))<1e-18
    X=s.linear.geometry(2)[0];u=np.zeros_like(X);u[:,0]=3e-5*(X[:,0]/.36)**2
    np.testing.assert_allclose(s.common_rms([(2,np.zeros_like(X)),(2,u)])[1,0],3e-5/np.sqrt(15),rtol=1e-12)


def test_immutable_old_sources_new_resources_and_new_counts():
    assert s.central.RESOURCES['RSS_cap_bytes']==4*1024**3
    assert s.RESOURCES['worker_RSS_cap_bytes']==8*1024**3
    assert s.RESOURCES['global_wall_cap_s']==3600
    assert s.GATES==s.central.GATES
    assert len(s.CASES)==5 and s.LOADS==['normal','paired_axial']
    pins=s.sources()
    for name in('experiments/campaign_surface_p2_static_v02.py','experiments/campaign_surface_p2_static_v01.py','experiments/run_clamped_surface_p2_qualification_v02.py','tests/test_surface_p2_static_v02.py'):
        assert name in pins


def test_independent_parent_deadline_reaps_owned_hanging_child(monkeypatch,tmp_path):
    import time
    monkeypatch.setattr(s,'LOCAL',tmp_path)
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(3600)'],start_new_session=True)
    row=dict(pid=process.pid,reaped=False,started_monotonic=time.monotonic()-s.RESOURCES['worker_wall_cap_s']-1)
    active={process:row};execution=dict(status='FAILED',workers=[row])
    try:
        with pytest.raises(s.central.ResourceStop,match='parent worker wall cap'):
            s.check_worker_deadlines(active,time.monotonic())
    finally:s.cleanup_workers(active,execution)
    assert row['reaped'] and process.returncode is not None and not active
    receipt=json.loads((tmp_path/'execution.json').read_text())
    assert receipt['cleanup_completed']and receipt['all_owned_workers_reaped']and receipt['live_groups']==0
    with pytest.raises(ProcessLookupError):os.kill(process.pid,0)
    with pytest.raises(ProcessLookupError):os.killpg(process.pid,0)
