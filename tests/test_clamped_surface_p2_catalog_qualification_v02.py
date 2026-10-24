import os,subprocess,sys,json
import numpy as np
import pytest
from experiments import run_clamped_surface_p2_catalog_qualification_v02 as s


def test_validity_and_prefix_excitation_are_separate_prospective_requirements():
    d=dict(minimum_J=.99,maximum_absolute_symmetric_strain_eigenvalue=.01,nonrigid_RMS_m=5e-6,nonrigid_fraction=.1,physical_passed=False)
    weak=s.catalog_deformation(d,False);prefix=s.catalog_deformation(d,True)
    assert weak['validity_passed']and not weak['excitation_passed']and weak['catalog_physical_passed']
    assert not prefix['catalog_physical_passed']
    assert weak['nonrigid_RMS_m']==5e-6 and weak['nonrigid_fraction']==.1
    assert not s.catalog_deformation(dict(d,maximum_absolute_symmetric_strain_eigenvalue=.06),False)['catalog_physical_passed']
    assert not s.catalog_deformation(dict(d,minimum_J=.79),False)['catalog_physical_passed']


def test_exact_central_load_reuse_and_nonzero_offcenter_moment_controls():
    assert s.verify_central_loads()
    rows=s.catalog_force_checks(2)
    assert len(rows)==10 and all(d['passed']for d in rows.values())
    X=s.central.linear.geometry(2)[0];force=s.force_program.load_catalog(2)['normal_x090']
    np.testing.assert_allclose(np.cross(X-[.18,.04,.04],force).sum(0),[0,-.0072,0],atol=1e-13)
    assert len(s.force_pins()['4'])==10


def test_all_laws_keep_validity_and_no_action_excitation_filter():
    deformation=s.catalog_deformation(dict(minimum_J=.99,maximum_absolute_symmetric_strain_eigenvalue=.01,nonrigid_RMS_m=1e-6,nonrigid_fraction=.01),False)
    row={law:dict(deformation=deformation.copy(),equilibrium=dict(passed=True),stability=dict(passed=True))for law in('linear','nonlinear','NH80_reference')}
    row.update(linear_vs_nonlinear_passed=True,NH24_vs_NH80_passed=True)
    record=dict(completed=True,levels={'2':row},spatial={law:dict(spatial_passed=True,medium_adequate=True)for law in('linear','nonlinear')})
    assert s.group_passed(record)
    record['levels']['2']['NH80_reference']['equilibrium']['passed']=False
    assert not s.group_passed(record)


def test_actual_eight_worker_guards_and_wrong_thread_rejection():
    script="import os,json;from experiments import run_clamped_surface_p2_catalog_qualification_v02 as s;cpu=int(os.environ['CATALOG_TEST_CPU']);os.sched_setaffinity(0,{cpu});s.guard(cpu);print(json.dumps({'cpu':list(os.sched_getaffinity(0))}))"
    env=dict(os.environ,PYTHONPATH=str(s.ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    children=[]
    for cpu in s.RESOURCES['worker_CPUs']:
        children.append((cpu,subprocess.Popen([sys.executable,'-c',script],cwd=s.ROOT,env=dict(env,CATALOG_TEST_CPU=str(cpu)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)))
    for cpu,child in children:
        out,err=child.communicate(timeout=45);assert child.returncode==0,err
        assert json.loads(out)['cpu']==[cpu]
    result=subprocess.run([sys.executable,'-c',script],cwd=s.ROOT,env=dict(env,CATALOG_TEST_CPU='4',OMP_NUM_THREADS='2'),capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'CPU/one-thread/noGPU guard'in result.stderr


def test_new_counts_resources_and_source_pins_no_inference():
    assert len(s.CASES)==5 and len(s.force_program.load_catalog(2))==10
    assert s.RESOURCES['workers']==8 and s.RESOURCES['worker_CPUs']==list(range(4,12))
    assert s.RESOURCES['RAM_preflight_bytes']==94*1024**3
    assert s.GATES==s.central.GATES
    pins=s.sources()
    for name in('experiments/run_clamped_surface_p2_catalog_qualification_v02.py','experiments/campaign_surface_p2_static_v02.py','experiments/campaign_surface_p2_force_v03.py'):
        assert name in pins
    assert 'experiments/campaign_surface_p2_study_v02.py' not in pins


def test_new_solver_linear_and_manufactured_NH_share_exact_rest_without_equilibrium_waiver():
    f=s.central.linear.SparseStaticForward(2);E,K=4300.,14.
    rest=s.ExactRestPCG(E*f.Kunit_free+K*f.W_free)
    rng=np.random.default_rng(338);u=rng.normal(size=f.X.shape)*1e-6;u[f.fixed]=0
    linear_force=((E*f.Kunit+K*f.W)@u.ravel()).reshape(u.shape)
    np.testing.assert_allclose(rest.rest_solve(linear_force.ravel()[f.free]),u.ravel()[f.free],atol=1e-14)
    for order in (24,80):
        model=s.PreconditionedP2Reference.from_forward(f,E,K,quadrature_order=order,direction_solver=rest)
        force=model.evaluate(u,np.zeros_like(u))['residual']
        solved,diag=model.solve(force)
        np.testing.assert_allclose(solved,u,atol=1e-11)
        state=model.evaluate(solved,force,tangent=True)
        assert np.linalg.norm(state['residual'].ravel()[f.free])<=1e-10
        assert s.generalized_stability(state['tangent'][f.free][:,f.free],rest.rest,rest.rest_LU.solve)['passed']
    assert rest.rest_factorizations==1
    assert s.SOLVER_DESIGN['stability']['eigenvalue_units'].startswith('dimensionless')


def test_terminal_bindings_reject_raw_or_public_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path);base=tmp_path/'local';public=tmp_path/'public'
    base.mkdir();public.mkdir();(base/'summary.json').write_text('{}');(base/'execution.json').write_text('{}')
    raw={}
    for case in s.CASES:
        for name in s.CENTRAL_ALIASES.values():
            folder=base/f'case-{case["index"]:02d}-{name}';folder.mkdir()
            for n in (2,4,8):
                path=folder/f'n{n}-LOCAL.npz';path.write_bytes(b'fixed-archive');raw[str(path.relative_to(base))]=s.sha(path)
    own=public/'terminal-ownership.json'
    own.write_text(json.dumps(dict(summary_sha256=s.sha(base/'summary.json'),execution_sha256=s.sha(base/'execution.json'),raw_fields_sha256=raw)))
    inv=base/'publication-inventory-completed.json'
    inv.write_text(json.dumps(dict(files=[dict(path=str(own.relative_to(tmp_path)),sha256=s.sha(own))])))
    assert s.verify_terminal_bindings(base,public)
    archive=base/next(iter(raw));archive.write_bytes(b'changed-archive')
    with pytest.raises(ValueError,match='archive expected hash'):s.verify_terminal_bindings(base,public)
    archive.write_bytes(b'fixed-archive');(base/'summary.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='ownership binding'):s.verify_terminal_bindings(base,public)
    (base/'summary.json').write_text('{}');own.write_text('{}')
    with pytest.raises(ValueError,match='published artifact'):s.verify_terminal_bindings(base,public)


def test_empirical_cost_projection_uses_both_observed_recipes_and_refuses_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s.central,'LOCAL',tmp_path/'central');s.central.LOCAL.mkdir()
    p=tmp_path/'runs/operator-dev/pcg-n8-parity-result.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(wall_s=287.8)))
    summary=s.central.LOCAL/'summary.json';summary.write_text(json.dumps(dict(per_case=[dict(worker_wall_s=200.)])))
    r=s.resource_projection();assert r['empirical_base_s']==287.8 and r['empirical_planning_estimate_wall_s']==3078.
    assert not r['proven_worst_case_bound']
    summary.write_text(json.dumps(dict(per_case=[dict(worker_wall_s=500.)])))
    assert s.resource_projection()['empirical_base_s']==500.
    summary.write_text(json.dumps(dict(per_case=[dict(worker_wall_s=800.)])))
    with pytest.raises(ValueError,match='declared catalog budget'):s.resource_projection()
