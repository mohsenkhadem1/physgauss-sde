import json,os,signal,subprocess,sys
import numpy as np
import pytest
from scipy.sparse.linalg import norm
from experiments import run_active_fixture_gaussian_nh80_cost_v02 as m


def test_full_catalog_retains_four_categories_five_parameters_all_controls_and_limited_pilot():
    c=m.catalog();assert len(c)==20 and len({x['id'] for x in c})==20
    assert {x['layout'] for x in c}=={x['id'] for x in m.bank.LAYOUTS}
    assert all(sum(x['layout']==s['id'] for x in c)==5 for s in m.bank.LAYOUTS)
    p=m.plan();assert p['future_NH80_fields']==20*11*3==660
    assert p['pilot']['new_equilibria']==6 and p['future_execution_authorized'] is False
    assert p['pilot']['loads']==['prefix-center-down','probe-x270-y060-plusY']
    assert p['gates']==m.metrics.GATES and p['resources']['wall_cap_s']==600
    assert 'tests/test_active_fixture_gaussian_load_v02.py' in m.sources()


def test_weak_prefix_exempts_only_absolute_excitation_not_fraction_or_physics():
    d=dict(minimum_J=.9,maximum_absolute_symmetric_strain_eigenvalue=.04,nonrigid_RMS_m=10e-6,nonrigid_fraction=.8)
    check=lambda x,label:m.admissibility(x,1e-12,True,dict(passed=True),dict(passed=True),label)
    assert check(d,'prefix-center-down')['load_eligible']
    assert not check(d,'probe-x270-y060-plusY')['load_eligible']
    for key,value in [('nonrigid_fraction',.2),('minimum_J',.79),('maximum_absolute_symmetric_strain_eigenvalue',.051)]:
        assert not check(dict(d,**{key:value}),'prefix-center-down')['load_eligible']
    assert not m.admissibility(d,2e-10,True,dict(passed=True),dict(passed=True),'prefix-center-down')['load_eligible']
    assert not m.admissibility(d,1e-12,True,dict(passed=True),dict(passed=False),'prefix-center-down')['load_eligible']


def test_new_fixture_gram_is_exact_same_nonlinear_spring_not_original_fixture():
    f=m.bank.make_forward(1,m.PILOT_LAYOUT);E,K=m.PILOT_PARAMETERS
    # Manufactured energy/gradient/tangent check, never an equilibrium solve.
    model=m.bank.core.P2NonlinearReference.from_forward(f,E,K,quadrature_order=80)
    u=np.zeros_like(f.X);u.ravel()[f.free]=np.sin(np.arange(len(f.free)))*1e-6
    force=np.zeros_like(u);a=model.evaluate(u,force,tangent=True)
    direction=np.zeros_like(u);direction.ravel()[f.free]=np.cos(np.arange(len(f.free)))
    h=1e-7
    ep=model.evaluate(u+h*direction,force)['potential_J'];em=model.evaluate(u-h*direction,force)['potential_J']
    np.testing.assert_allclose((ep-em)/(2*h),np.sum(a['residual']*direction),atol=2e-10,rtol=2e-6)
    assert model.W is f.W
    zero=model.evaluate(np.zeros_like(u),force,tangent=True)['tangent']
    assert norm(zero-(E*f.Kunit+K*f.W))/norm(zero)<2e-14
    # Independent Gram quadratic energy/gradient extracted from nonlinear potential.
    U,g,_,_=model.elastic(f.X+u,False);Wu=(f.W@u.ravel()).reshape(u.shape)
    np.testing.assert_allclose(a['potential_J']-U,.5*K*np.sum(u*Wu),atol=1e-18)
    np.testing.assert_allclose(a['residual']-g,K*Wu,atol=1e-16)


def test_spatial_gate_cannot_hide_mesh_or_order_failure(monkeypatch):
    fields={(n,label):np.zeros((1,3)) for n in m.LEVELS for label in m.PILOT_LOADS}
    monkeypatch.setattr(m.metrics,'common_rms',lambda fields:np.array([[0,0,0],[100e-6,0,0],[0,40e-6,0]]))
    monkeypatch.setattr(m.bank.core,'field_queries',lambda n,u,p:np.full((len(p),3),n*10e-6))
    r=m.spatial_results(fields)
    assert all(not x['spatial_development_passed'] and not x['bank_admitted'] for x in r.values())
    assert all(x['medium_indicator_m']==80e-6 for x in r.values())


def test_unreleased_or_tampered_plan_refuses_before_physics(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'sources',lambda:{'s':'h'});monkeypatch.setattr(m,'negative_origin',lambda:{'old_negative_preserved':True})
    monkeypatch.setattr(m,'sha',lambda path:'toy')
    p=dict(plan=m.plan(),source_SHA256={'s':'h'},origin={'old_negative_preserved':True})
    (tmp_path/'protocol.json').write_text(json.dumps(p));(tmp_path/'release.json').write_text(json.dumps(dict(approved=False)))
    with pytest.raises(ValueError,match='Root release'):m.validate('toy')
    (tmp_path/'release.json').write_text(json.dumps(dict(approved=True,protocol_SHA256='toy',source_SHA256={'s':'h'})))
    assert m.validate('toy')==p
    p['plan']['pilot']['loads'][0]='held-offaxis-down';(tmp_path/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Frozen'):m.validate('toy')


def test_controller_stop_drains_owned_child_restores_handlers(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda role:None);monkeypatch.setattr(m,'validate',lambda digest:{})
    (tmp_path/'release.json').write_text('{}');monkeypatch.setattr(m,'record',lambda p:{'path':str(p),'sha256':'toy'})
    before={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};cleaned=[]
    class Child:
        pid=999999
        def poll(self):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    child=Child();monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:child)
    monkeypatch.setattr(m.prior.shared.old.cost,'cleanup',lambda c:cleaned.append(c) or -15)
    r=m.run('toy');assert r['terminal'] and not r['completed'] and 'InterruptedError' in r['failure']
    assert cleaned==[child] and all(signal.getsignal(s)==h for s,h in before.items())


def test_real_controller_and_generated_worker_guards_no_equilibria():
    m.guard('controller')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    r=subprocess.run(['taskset','-c','17',sys.executable,'-c',"from experiments.run_active_fixture_gaussian_nh80_cost_v02 import guard;guard('worker');print('guard-PASS')"],cwd=m.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert r.returncode==0 and r.stdout.strip()=='guard-PASS',r.stderr


def test_new_load_law_is_explicit_and_old_negative_required_before_new_physics(tmp_path,monkeypatch):
    assert m.plan()['bank_design']['schema']=='active-fixture-Gaussian-load-development-v02'
    assert m.plan()['bank_design']['load_std_m']==[.035,.025]
    assert m.bank.load_specs()[0]['force_N']==.04
    assert all(x['force_N']==.06 for x in m.bank.load_specs()[1:])
    monkeypatch.setattr(m.predecessor,'LOCAL',tmp_path)
    with pytest.raises(FileNotFoundError):m.negative_origin()
    assert m.predecessor.LOCAL!=m.LOCAL
