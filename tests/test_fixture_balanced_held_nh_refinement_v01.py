import copy,json,os,signal,subprocess,sys
import numpy as np
import pytest
from scipy.sparse.linalg import norm
from experiments import run_fixture_balanced_held_nh_refinement_v01 as m


def test_exact_six_new_fields_original_gates_no_full_queue():
    p=m.plan()
    assert p['pilot']['loads']==['held-oblique-positiveY-down','held-oblique-negativeY-down']
    assert p['pilot']['levels']==[3,6,12]and p['pilot']['new_equilibria']==6
    assert (p['pilot']['E_Pa'],p['pilot']['K_N_m'])==(2000.,5.)and p['gates']==m.metrics.GATES
    assert p['quadrature_points']==80 and p['weak_prefix_excitation_exemption']is False
    assert not any(k.startswith('future_')for k in p)
    assert p['resources']['wall_cap_s']==1800 and p['resources']['aggregate_RSS_cap_bytes']==28*1024**3
    assert m.bank.measurement_points().shape==(16,3)
    assert 'tests/test_active_fixture_resolution_v03.py'in m.sources()


def test_both_held_require_original_excitation_and_all_physics():
    d=dict(minimum_J=.9,maximum_absolute_symmetric_strain_eigenvalue=.04,nonrigid_RMS_m=150e-6,nonrigid_fraction=.8)
    for label in m.PILOT_LOADS:
        check=lambda x:m.admissibility(x,1e-12,True,dict(passed=True),dict(passed=True),label)
        assert check(d)['load_eligible']and not check(d)['weak_prefix_control']
        for key,value in [('nonrigid_RMS_m',99e-6),('nonrigid_fraction',.2),('minimum_J',.79),('maximum_absolute_symmetric_strain_eigenvalue',.051)]:assert not check(dict(d,**{key:value}))['load_eligible']
        assert not m.admissibility(d,2e-10,True,dict(passed=True),dict(passed=True),label)['physical_passed']
    with pytest.raises(ValueError,match='no weak'):m.admissibility(d,1e-12,True,dict(passed=True),dict(passed=True),'prefix-center-down')


def test_physical_vs_excitation_spatial_and_membership_are_separate():
    rows=[dict(n=n,label=l,completed=True,physical_passed=True,load_eligible=True,excitation_goal_met=True,weak_prefix_control=False)for n in m.LEVELS for l in m.PILOT_LOADS]
    spatial={l:dict(spatial_development_passed=True)for l in m.PILOT_LOADS}
    assert m.classification(rows,spatial)['pilot_qualified']and not m.classification(rows,spatial)['bank_admitted']
    bad=copy.deepcopy(rows);bad[0]['load_eligible']=False;v=m.classification(bad,spatial);assert v['passed']and not v['pilot_qualified']
    bad=copy.deepcopy(spatial);bad[m.PILOT_LOADS[0]]['spatial_development_passed']=False;assert not m.classification(rows,bad)['pilot_qualified']
    bad=copy.deepcopy(rows);bad[-1]=bad[0];assert not m.classification(bad,spatial)['all6_retained']


def test_manufactured_new_force_NH_energy_gradient_spring_and_rest_tangent():
    # Tiny manufactured fields only, not a declared equilibrium/cost case.
    f=m.bank.make_forward(1,m.PILOT_LAYOUT);E,K=m.PILOT_PARAMETERS
    model=m.bank.bank.core.P2NonlinearReference.from_forward(f,E,K,quadrature_order=80)
    loads,_=m.bank.load_catalog(1);u=np.zeros_like(f.X);u.ravel()[f.free]=np.sin(np.arange(len(f.free)))*1e-6
    direction=np.zeros_like(u);direction.ravel()[f.free]=np.cos(np.arange(len(f.free)));h=1e-7
    for label in m.PILOT_LOADS:
        force=loads[label];a=model.evaluate(u,force,tangent=True)
        ep=model.evaluate(u+h*direction,force)['potential_J'];em=model.evaluate(u-h*direction,force)['potential_J']
        np.testing.assert_allclose((ep-em)/(2*h),np.sum(a['residual']*direction),atol=2e-10,rtol=2e-6)
        U,g,_,_=model.elastic(f.X+u,False);Wu=(f.W@u.ravel()).reshape(u.shape)
        np.testing.assert_allclose(a['potential_J']-U+np.sum(force*u),.5*K*np.sum(u*Wu),atol=1e-18)
        np.testing.assert_allclose(a['residual']-g+force,K*Wu,atol=1e-16)
    assert model.W is f.W
    H=model.evaluate(np.zeros_like(u),np.zeros_like(u),tangent=True)['tangent'];assert norm(H-(E*f.Kunit+K*f.W))/norm(H)<2e-14


def test_original_public_query_spatial_gate_no_hidden_rescue(monkeypatch):
    fields={(n,label):np.zeros((1,3))for n in m.LEVELS for label in m.PILOT_LOADS}
    monkeypatch.setattr(m.resolution,'common_rms',lambda fields,finest:np.array([[0,0,0],[100e-6,0,0],[0,5e-6,0]]))
    monkeypatch.setattr(m.resolution,'field_queries',lambda n,u,p:np.full((len(p),3),n*10e-6))
    out=m.spatial_results(fields)
    assert set(out)==set(m.PILOT_LOADS)
    assert all(r['spatial_passed']and r['medium_adequate']and not r['public_point20um_passed']and not r['spatial_development_passed']for r in out.values())


def fake_origin(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m.screen,'LOCAL',tmp_path)
    monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256=m.sha(p)))
    for path in('experiments/audit_fixture_balanced_decision_screen_v01.py','tests/test_audit_fixture_balanced_decision_screen_v01.py'):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic procedure')
    for name in('protocol.json','release.json'):(tmp_path/name).write_text('{}')
    adeq=dict(qualified_cases=36,adequacy_qualified=True);child=lambda pid:dict(pid=pid,reaped=True,exit_code=0,cleanup_exit_code=0)
    e=dict(terminal=True,completed=True,controller_pid=100,builder=child(101),workers=[child(i)for i in(102,103,104)],adequacy=adeq)
    m.write(tmp_path/'execution.json',e);p=dict(source_SHA256={'synthetic':'procedure'})
    monkeypatch.setattr(m.screen,'validate',lambda d:p);seen=[];monkeypatch.setattr(m.screen,'absent',lambda pid:seen.append(pid));monkeypatch.setattr(m.screen,'verify_records',lambda v:None)
    closure=dict(root_exit_code=0,root_reaped=True,all_owned_PID_and_groups_absent=True,protocol_SHA256=m.sha(tmp_path/'protocol.json'),source_SHA256=p['source_SHA256'],execution_SHA256=m.sha(tmp_path/'execution.json'),owned_pids=[100,101,102,103,104]);m.write(tmp_path/'root-closure.json',closure)
    summary=dict(integrity_passed=True,execution_clean=True,completed_case_records=36,qualified_category_references=144,completed_category_references=144,completed_decision_banks=180,independently_recomputed_rankings=36,completed_utility_jobs=1728,adequacy=adeq)
    review=dict(summary=summary,source_SHA256={k:m.sha(tmp_path/k)for k in('experiments/audit_fixture_balanced_decision_screen_v01.py','tests/test_audit_fixture_balanced_decision_screen_v01.py')},scientific_scalar_bindings=[])
    for key,name in(('protocol','protocol.json'),('release','release.json'),('execution','execution.json'),('root_closure','root-closure.json')):review[key]=m.record(tmp_path/name)
    m.write(tmp_path/'independent-terminal-review.json',review)
    return e,review,seen


def test_adequacy_closure_and_hash_gate_before_any_new_field(tmp_path,monkeypatch):
    e,review,seen=fake_origin(tmp_path,monkeypatch)
    assert m.balanced_origin()['all36_numerical_and_adequacy_passed']and seen==[100,101,102,103,104]
    review['summary']['adequacy']['adequacy_qualified']=False;m.write(tmp_path/'independent-terminal-review.json',review)
    with pytest.raises(ValueError,match='adequacy'):m.balanced_origin()
    review['summary']['adequacy']['adequacy_qualified']=True;review['release']['sha256']='tamper';m.write(tmp_path/'independent-terminal-review.json',review)
    with pytest.raises(ValueError,match='bytes'):m.balanced_origin()


def test_live_gate_does_not_open_policy_outcomes(tmp_path,monkeypatch):
    monkeypatch.setattr(m.screen,'LOCAL',tmp_path);(tmp_path/'execution.json').write_text('{"terminal":false}')
    monkeypatch.setattr(m.screen,'validate',lambda _:pytest.fail('Must not validate/read current policy'))
    with pytest.raises(ValueError,match='terminal'):m.balanced_origin()


def test_unreleased_or_tampered_new_plan_refuses(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'sources',lambda:{'s':'h'});monkeypatch.setattr(m,'balanced_origin',lambda:{'qualified':True});monkeypatch.setattr(m,'sha',lambda p:'toy')
    p=dict(plan=m.plan(),source_SHA256={'s':'h'},origin={'qualified':True});m.write(tmp_path/'protocol.json',p);m.write(tmp_path/'release.json',dict(approved=False))
    with pytest.raises(ValueError,match='release'):m.validate('toy')
    m.write(tmp_path/'release.json',dict(approved=True,protocol_SHA256='toy',source_SHA256=p['source_SHA256'],origin=p['origin']));assert m.validate('toy')==p
    p['plan']['pilot']['loads'][0]='prefix-center-down';m.write(tmp_path/'protocol.json',p)
    with pytest.raises(ValueError,match='Frozen'):m.validate('toy')


def test_owned_controller_stop_drains_restores_and_preserves_negative(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda role:None);monkeypatch.setattr(m,'validate',lambda digest:{})
    (tmp_path/'release.json').write_text('{}');monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256='toy'))
    before={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGINT)};cleaned=[]
    class Child:
        pid=999999
        def poll(self):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    c=Child();monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:c);monkeypatch.setattr(m.prior.shared.old.cost,'cleanup',lambda child:cleaned.append(child)or-15)
    v=m.run('toy');assert v['terminal']and not v['completed']and 'InterruptedError'in v['failure']and cleaned==[c]
    assert all(signal.getsignal(s)==h for s,h in before.items())


def test_real_controller_and_worker_startup_guards_only():
    m.guard('controller')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    v=subprocess.run(['taskset','-c','17',sys.executable,'-c',"from experiments.run_fixture_balanced_held_nh_refinement_v01 import guard;guard('worker');print('guard-PASS')"],cwd=m.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert v.returncode==0 and v.stdout.strip()=='guard-PASS',v.stderr


def test_tiny_n3_surface_load_fixture_alias_and_independent_quadrature():
    # Geometry and quadrature controls only: no equilibrium or NH response.
    f=m.resolution.make_forward(3,m.PILOT_LAYOUT);loads,_=m.bank.load_catalog(3)
    checks=m.surface_checks(f,loads)
    assert checks['positive_quadrature_Gram_PSD']and checks['relative_support16_vs20']<=1e-10 and checks['maximum_load16_vs20_N']<=1e-12
    assert set(m.PILOT_LOADS)<=set(checks['refined_known_load_checks'])
