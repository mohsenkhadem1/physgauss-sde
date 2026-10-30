import copy
import hashlib
import json
import numpy as np
import pytest
from scripts.coupled_contact_v13 import surface_pipeline as surface
from scripts.coupled_contact_v13 import pipeline as original
from scripts.coupled_contact_v13.physics import make_block,Collider,FEMSystem
from scripts.coupled_contact_v13.rigid import RigidSystem
from scripts.coupled_contact_v13.surface_contact import SurfaceContactMixin


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


def test_surface_families_same_contact_mro_and_rigid_integration():
    for family in ['fem','native','rigid','oracle']:
        cls=surface.model_class(family)
        assert cls.contact_forces is SurfaceContactMixin.contact_forces
        assert cls.step is SurfaceContactMixin.step
    assert RigidSystem in surface.model_class('rigid').__mro__
    rest,tets=make_block(cells=(1,1,1),center=(0,0,.019))
    cols=[Collider(kind='plane',name='floor')]
    a=surface.model_class('fem')(rest,tets,colliders=copy.deepcopy(cols))
    b=surface.model_class('rigid')(rest,tets,colliders=copy.deepcopy(cols))
    np.testing.assert_allclose(a.contact_forces()[0],b.contact_forces()[0])
    b.step(1e-5)
    assert b.contact_samples>0
    np.testing.assert_allclose(np.linalg.det(b.deformation()),1,atol=1e-12)


def test_new_cache_only_and_path_escape_rejected(tmp_path,monkeypatch):
    impl=surface.implementation();monkeypatch.setattr(impl,'BASE',tmp_path);monkeypatch.setattr(impl,'protocol',lambda: {})
    job={'path':'prediction/fresh/example/fem'}
    destination=tmp_path/job['path'];destination.mkdir(parents=True)
    marker={'success':True,'surface_cache':True}
    (destination/'complete.json').write_text(json.dumps(marker))
    assert surface.simulate_job(job)==marker
    with pytest.raises(ValueError):surface.simulate_job({'path':'../coupled-contact-v13/anything'})
    with pytest.raises(ValueError):surface.simulate_job({'path':'/tmp/anything'})


def test_new_reference_job_protocol_is_used(tmp_path,monkeypatch):
    impl=surface.implementation();monkeypatch.setattr(impl,'REPORT',tmp_path)
    protocol={'objects':[{'id':'fresh0','split':'reserved','young':7777.,'friction':.33}],
              'forecast_scenarios':['edge_drop']}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    jobs=impl.reference_jobs('reserved')
    assert len(jobs)==4
    assert all('fresh0' in job['path'] and job['scale']==7777. for job in jobs)
    assert all(job['reference'] for job in jobs)


def test_surface_scene_bindings_geometry_and_contact_stiffness():
    from scripts.coupled_contact_v13 import surface_scenarios,scenarios
    impl=surface.implementation()
    for name in ('build','run','command','shared_node_indices'):
        assert getattr(impl,name) is getattr(surface_scenarios,name)
        assert getattr(original,name) is getattr(scenarios,name)
    block=impl.build('edge_drop')
    edge=next(c for c in block.colliders if c.name=='edge')
    assert np.isclose(block.rest.mean(0)[0]-edge.center[0],.008)
    assert all(c.stiffness==19200. for c in block.colliders)
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
        assert a.stiffness==b.stiffness==19200.
