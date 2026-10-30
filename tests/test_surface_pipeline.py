import copy
import hashlib
import json
import numpy as np
import pytest
from scripts.coupled_contact_v13 import surface_pipeline as surface
from scripts.coupled_contact_v13 import pipeline as original
from scripts.coupled_contact_v13.physics import make_block,Collider,FEMSystem
from scripts.coupled_contact_v13.rigid import RigidSystem
from scripts.coupled_contact_v13.clipped_surface_contact import ClippedSurfaceContactMixin
from scripts.coupled_contact_v13.consistent_mass import ConsistentMassMixin,ConsistentRigidMassMixin


def test_private_namespace_preserves_original_paths_classes_and_source():
    source=surface.ROOT/'scripts/coupled_contact_v13/pipeline.py'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    old=(original.BASE,original.REPORT,original.model_class,original.execute)
    impl=surface.implementation()
    assert impl is not original
    assert impl.BASE==surface.BASE and impl.REPORT==surface.REPORT
    assert impl.BASE!=original.BASE and impl.REPORT!=original.REPORT
    assert impl.model_class is surface.model_class and impl.execute is surface.execute
    assert (original.BASE,original.REPORT,original.model_class,original.execute)==old
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before


def test_surface_families_same_contact_mro_and_rigid_integration(monkeypatch):
    monkeypatch.setattr(surface.implementation(),'protocol',lambda: dict(surface_spacing_m=.0015,friction_quadrature_level=1))
    for family in ['fem','native','rigid','oracle']:
        cls=surface.model_class(family)
        assert cls.contact_forces is ClippedSurfaceContactMixin.contact_forces
        assert cls.step is (ConsistentRigidMassMixin.step if family=='rigid' else ConsistentMassMixin.step)
    assert RigidSystem in surface.model_class('rigid').__mro__
    rest,tets=make_block(cells=(1,1,1),center=(0,0,.019))
    cols=[Collider(kind='plane',name='floor')]
    a=surface.model_class('fem')(rest,tets,colliders=copy.deepcopy(cols))
    b=surface.model_class('rigid')(rest,tets,colliders=copy.deepcopy(cols))
    np.testing.assert_allclose(a.contact_forces()[0],b.contact_forces()[0])
    np.testing.assert_allclose(a.mass_matrix.toarray(),b.mass_matrix.toarray())
    b.step(1e-5)
    assert b.contact_samples>0
    np.testing.assert_allclose(np.linalg.det(b.deformation()),1,atol=1e-12)


def test_new_cache_only_and_path_escape_rejected(tmp_path,monkeypatch):
    impl=surface.implementation();monkeypatch.setattr(impl,'BASE',tmp_path);monkeypatch.setattr(impl,'REPORT',tmp_path)
    p=dict(reference_cells=[8]*3,coarse_cells=[4]*3,dt_reference=.000025,dt_coarse=.0000125,duration_seconds=.6,surface_spacing_m=.0015,friction_quadrature_level=1,
           reference_settings_by_scenario={'edge_drop':dict(cells=[12]*3,dt=.00001,companion_cells=[16]*3)})
    (tmp_path/'protocol.json').write_text(json.dumps(p))
    job=dict(path='reference/fresh/edge_drop',scenario='edge_drop',reference=True,family='fem',scale=6000.,friction=.6)
    settings=surface.simulation_settings(job,p)
    assert settings['cells']==[12]*3 and settings['dt']==.00001
    assert surface.simulation_settings(dict(job,spatial=True),p)['cells']==[16]*3
    assert surface.simulation_settings(dict(job,spatial=True),p)['dt']==.00001
    assert surface.simulation_settings(dict(job,refinement=True),p)['dt']==.000005
    assert surface.simulation_settings(dict(job,reference=False),p)['dt']==.0000125
    assert surface.simulation_settings(dict(job,scenario='curved'),p)['cells']==[8]*3
    destination=tmp_path/job['path'];destination.mkdir(parents=True)
    marker=dict(success=True,job=job,simulation_settings=settings,protocol_sha256=hashlib.sha256((tmp_path/'protocol.json').read_bytes()).hexdigest())
    (destination/'complete.json').write_text(json.dumps(marker))
    assert surface.simulate_job(job)==marker
    with pytest.raises(RuntimeError):surface.simulate_job(dict(job,scale=9000.))
    with pytest.raises(ValueError):surface.simulate_job({'path':'../coupled-contact-v13/anything'})
    with pytest.raises(ValueError):surface.simulate_job({'path':'/tmp/anything'})


def test_new_reference_job_protocol_is_used(tmp_path,monkeypatch):
    impl=surface.implementation();monkeypatch.setattr(impl,'REPORT',tmp_path)
    protocol={'objects':[{'id':'fresh0','split':'reserved','young':7777.,'friction':.33}],
              'forecast_scenarios':['edge_drop']}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    jobs=impl.reference_jobs('reserved')
    assert len(jobs)==6
    assert all('fresh0' in job['path'] and job['scale']==7777. for job in jobs)
    assert all(job['reference'] for job in jobs)


def test_surface_scene_bindings_geometry_and_contact_stiffness(monkeypatch):
    monkeypatch.setattr(surface.implementation(),'protocol',lambda: dict(surface_spacing_m=.0015,friction_quadrature_level=1))
    from scripts.coupled_contact_v13 import engineered_scenarios,scenarios
    impl=surface.implementation()
    for name in ('build','run','command','shared_node_indices'):
        assert getattr(impl,name) is getattr(engineered_scenarios,name)
        assert getattr(original,name) is getattr(scenarios,name)
    block=impl.build('edge_drop')
    edge=next(c for c in block.colliders if c.name=='edge')
    assert np.isclose(block.rest.mean(0)[0]-edge.center[0],.02)
    assert all(c.stiffness==76800. for c in block.colliders)
    historical=scenarios.build('edge_drop')
    oldedge=next(c for c in historical.colliders if c.name=='edge')
    assert np.isclose(historical.rest.mean(0)[0]-oldedge.center[0],-.002)
    assert all(c.stiffness==4800. for c in historical.colliders)
    # Changing only the contact family retains identical new public geometry.
    newvertex=impl.build('squeeze',system_class=FEMSystem)
    newsurface=impl.build('squeeze',system_class=surface.model_class('fem'))
    np.testing.assert_array_equal(newvertex.rest,newsurface.rest)
    for a,b in zip(newvertex.colliders,newsurface.colliders):
        np.testing.assert_array_equal(a.center,b.center)
        assert a.stiffness==b.stiffness==76800.


def test_actual_job_records_override_and_refuses_legacy_cache(tmp_path,monkeypatch):
    m=surface.implementation();monkeypatch.setattr(m,'BASE',tmp_path/'runs');monkeypatch.setattr(m,'REPORT',tmp_path)
    p=dict(reference_cells=[8]*3,coarse_cells=[4]*3,dt_reference=25e-6,dt_coarse=12.5e-6,duration_seconds=.6,
           reference_settings_by_scenario={'edge_drop':dict(cells=[12]*3,dt=10e-6)},surface_spacing_m=.00075,friction_quadrature_level=2)
    (tmp_path/'protocol.json').write_text(json.dumps(p))
    seen=[]
    def run(*args):
        seen.append(args)
        return dict(frames=np.zeros((2,1,3)),min_j=1.,max_penetration_m=0.)
    monkeypatch.setattr(m,'run',run)
    j=dict(path='reference/fresh/edge_drop',scenario='edge_drop',reference=True,refinement=True,family='fem',scale=6000.,friction=.6)
    result=surface.simulate_job(j)
    assert result['success']
    assert seen[0][3:6]==((12,12,12),5e-6,.6)
    assert seen[0][6].surface_spacing==.00075 and seen[0][6].friction_quadrature_level==2
    assert result['simulation_settings']['dt']==5e-6
    surface.simulate_job(j)
    assert len(seen)==1
    result.pop('simulation_settings')
    (m.BASE/j['path']/'complete.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError):surface.simulate_job(j)


def test_identical_spatial_resolution_is_rejected():
    p=dict(reference_cells=[8]*3,coarse_cells=[4]*3,dt_reference=25e-6,dt_coarse=12.5e-6,duration_seconds=.6,surface_spacing_m=.0015,friction_quadrature_level=1,
           reference_settings_by_scenario={'curved':dict(cells=[8]*3,companion_cells=[8]*3,dt=25e-6)})
    with pytest.raises(ValueError,match='differ'):
        surface.simulation_settings(dict(reference=True,scenario='curved',spatial=True),p)


def test_largest_first_scheduling_preserves_inputs_ties_and_result_order(monkeypatch):
    p=dict(reference_cells=[8]*3,coarse_cells=[4]*3,dt_reference=25e-6,dt_coarse=12.5e-6,duration_seconds=.6,surface_spacing_m=.0015,friction_quadrature_level=1,
           reference_settings_by_scenario={'edge_drop':dict(cells=[12]*3,companion_cells=[16]*3,dt=25e-6)})
    jobs=[dict(path='a',scenario='curved',reference=True),dict(path='b',scenario='edge_drop',reference=True,spatial=True),dict(path='c',scenario='edge_drop',reference=True,refinement=True),dict(path='d',scenario='curved',reference=True)]
    original_jobs=copy.deepcopy(jobs)
    monkeypatch.setattr(surface.implementation(),'protocol',lambda:p)
    seen=[]
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,function,ordered):
            seen.extend(ordered)
            return [dict(success=True,job=j,seconds=0.) for j in ordered]
    monkeypatch.setattr(surface.concurrent.futures,'ProcessPoolExecutor',Pool)
    rows=surface.execute(jobs,2)
    assert [j['path'] for j in seen]==['b','c','a','d']
    assert [r['job'] for r in rows]==jobs==original_jobs
    assert surface.execute([],2)==[]
