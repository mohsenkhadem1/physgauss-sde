import json,os,subprocess,sys,time
import numpy as np
import pytest
from experiments import run_clamped_surface_p2_qualification_v03 as s


def test_force_reuse_proof_and_immutable_gates_counts():
    p=s.verify_force_reuse();assert set(p)=={'2','4','8'}
    assert all(v['axial_byte_identical']for v in p.values())
    assert s.GATES==s.central.GATES
    assert len(s.CASES)==5 and s.LOADS==['normal','paired_axial']
    assert s.RESOURCES==s.central.RESOURCES
    assert s.forces.NORMAL_WARMSTART_SCALE==.8


def test_manufactured_linear_scaling_and_new_NH_equilibrium_not_fake_warmstart():
    f=s.linear.SparseStaticForward(2);rng=np.random.default_rng(84);u=rng.normal(size=f.X.shape)*1e-6;u[f.fixed]=0
    A=4700*f.Kunit+17*f.W;force=(A@u.ravel()).reshape(u.shape)
    np.testing.assert_allclose((A@(.8*u).ravel()).reshape(u.shape),.8*force,rtol=1e-12,atol=1e-15)
    for order in(24,80):
        model=s.linear.P2NonlinearReference.from_forward(f,4700.,17.,quadrature_order=order)
        target=model.evaluate(u,np.zeros_like(u))['residual']
        warm,wd=model.solve(target,initial=.8*u);cold,cd=model.solve(target)
        np.testing.assert_allclose(warm,u,rtol=2e-6,atol=1e-11)
        np.testing.assert_allclose(cold,warm,rtol=2e-6,atol=1e-11)
        assert np.linalg.norm(warm-.8*u)>1e-7 and wd['iterations']>=1
        assert np.linalg.norm(model.evaluate(warm,target)['residual'].ravel()[model.free])<=1e-10


def test_all_ten_retained_even_if_one_fails():
    rows=[dict(case_index=c['index'],load=name,passed=True)for c in s.CASES for name in s.LOADS];rows[0]['passed']=False
    r=s.retained_groups(rows);assert r['all10_retained']and r['qualified_groups']==9 and not r['all_qualified']
    with pytest.raises(ValueError):s.retained_groups(rows[1:])


def test_actual_four_worker_guard_and_parent_cleanup(monkeypatch,tmp_path):
    script="import os,json;from experiments import run_clamped_surface_p2_qualification_v03 as s;cpu=int(os.environ['QUALIFICATION_TEST_CPU']);os.sched_setaffinity(0,{cpu});s.guard(cpu);print(json.dumps({'cpu':list(os.sched_getaffinity(0))}))"
    env=dict(os.environ,PYTHONPATH=str(s.ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');children=[]
    for cpu in s.RESOURCES['worker_CPUs']:
        children.append((cpu,subprocess.Popen([sys.executable,'-c',script],cwd=s.ROOT,env=dict(env,QUALIFICATION_TEST_CPU=str(cpu)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)))
    for cpu,p in children:
        out,err=p.communicate(timeout=30);assert p.returncode==0,err;assert json.loads(out)['cpu']==[cpu]
    bad=subprocess.run([sys.executable,'-c',script],cwd=s.ROOT,env=dict(env,QUALIFICATION_TEST_CPU='4',OMP_NUM_THREADS='2'),capture_output=True,text=True,timeout=30)
    assert bad.returncode!=0 and 'CPU/one-thread/noGPU guard'in bad.stderr
    monkeypatch.setattr(s,'LOCAL',tmp_path);p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(3600)'],start_new_session=True)
    row=dict(pid=p.pid,reaped=False,started_monotonic=time.monotonic()-s.RESOURCES['worker_wall_cap_s']-1);active={p:row};execution=dict(status='FAILED',workers=[row])
    try:
        with pytest.raises(s.central.central.ResourceStop):s.check_worker_deadlines(active,time.monotonic())
    finally:s.cleanup_workers(active,execution)
    assert row['reaped']and not active
    with pytest.raises(ProcessLookupError):os.killpg(p.pid,0)


def test_source_pins_include_external_force_adapter_and_frozen_physics():
    pins=s.sources()
    for name in('experiments/campaign_surface_p2_force_v03.py','experiments/campaign_surface_p2_static_v02.py','experiments/run_clamped_surface_p2_qualification_v03.py','tests/test_surface_p2_force_v03.py'):
        assert name in pins


def test_quadrature_force_resultants_use_new_external_program():
    for n in (2,4,8):
        row=s.quadrature_checks(s.linear.SparseStaticForward(n))
        assert row['passed']
        assert all(x['net_force_error_N']<1e-12 and x['passed']for x in row['loads'].values())
