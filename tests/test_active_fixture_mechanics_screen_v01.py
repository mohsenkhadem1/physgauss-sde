import json
import numpy as np
import pytest
from experiments import run_active_fixture_mechanics_screen_v01 as m


def test_fixed_cartesian_membership_and_equal_common_control_menu():
    assert len(m.E_VALUES)*len(m.K_VALUES)*len(m.bank.LAYOUTS)*2==72
    assert len(m.bank.load_specs())==11
    assert len(m.bank.measurement_specs())==60
    assert m.RESOURCES['wall_cap_s']==120 and m.RESOURCES['worker_CPU']==17
    assert 'layout'not in m.bank.load_catalog.__code__.co_varnames


def test_missing_or_tampered_root_release_refuses_before_solve(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'sources',lambda:{'s':'h'})
    p=dict(source_SHA256={'s':'h'},resources=m.RESOURCES,design=m.bank.design(),mesh_n=[1,2],E_Pa=list(m.E_VALUES),K_N_m=list(m.K_VALUES))
    (tmp_path/'protocol.json').write_text(json.dumps(p));digest=m.sha(tmp_path/'protocol.json')
    (tmp_path/'release.json').write_text(json.dumps(dict(approved=False)))
    with pytest.raises(ValueError,match='root release'):m.validate(digest)
    (tmp_path/'release.json').write_text(json.dumps(dict(approved=True,protocol_SHA256=digest,source_SHA256={'s':'h'})))
    assert m.validate(digest)==p
    p['E_Pa'][0]=1999.;(tmp_path/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):m.validate(m.sha(tmp_path/'protocol.json'))


def test_actual_deformed_wrench_uses_known_world_dead_force():
    x=np.array([[0.,0.,0.],[1.,0.,0.]]);u=np.array([[0.,.1,0.],[0.,-.1,0.]])
    f=np.array([[0.,0.,1.],[0.,0.,-1.]])
    out=m.force_wrench(x,f,u)
    np.testing.assert_allclose(out['resultant_N'],0)
    np.testing.assert_allclose(out['moment_at_deformed_nodes_Nm'],np.cross(x+u-m.bank.ORIGIN_M,f).sum(0))


def test_controller_interrupt_drains_only_owned_child_and_restores_signals(tmp_path,monkeypatch):
    import signal
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda role:None);monkeypatch.setattr(m,'validate',lambda digest:{})
    monkeypatch.setattr(m,'record',lambda path:dict(path=str(path),sha256='toy'))
    old_handlers={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGINT)};cleaned=[]
    class Child:
        pid=999999
        def poll(self):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    child=Child();monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(m.shared.old.cost,'cleanup',lambda c:cleaned.append(c)or-15)
    result=m.run('toy')
    assert result['terminal']and not result['completed']and 'InterruptedError'in result['failure']
    assert cleaned==[child]and all(signal.getsignal(s)==h for s,h in old_handlers.items())


def test_generated_worker_guard_real_cpu17_and_controller18_no_solve():
    import os,subprocess,sys
    m.guard('controller')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    out=subprocess.run(['taskset','-c','17',sys.executable,'-c',"from experiments.run_active_fixture_mechanics_screen_v01 import guard;guard('worker');print('guard-PASS')"],cwd=m.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert out.returncode==0 and out.stdout.strip()=='guard-PASS',out.stderr
