import numpy as np
from scripts.phystwin.v08_study import indices,Experiment
from scripts.phystwin.v08_reference import StrongFixture
from scripts.phystwin.sequential_fixture import SequentialFixture,ACTIONS,points
from scripts.phystwin.v08_physbridge import forces

def test_stronger_force_preserves_gravity_and_resultants():
 m=StrongFixture(2,4000,500,1000,'crossed',device='cpu');old=SequentialFixture(2,4000,500,1000,'crossed',device='cpu')
 for a in ACTIONS:
  m.set_action(a);old.set_action(a);g=np.zeros_like(old.load);g[:,2]=-.02*old.volume_weights
  np.testing.assert_allclose(m.load,g+(16 if a=='initial' else 4)*(old.load-g),atol=1e-14)
  np.testing.assert_allclose(m.load.sum(0),forces(a).sum(0),atol=1e-12)
  np.testing.assert_allclose(np.cross(m.rest,m.load).sum(0),np.cross(points(),forces(a)).sum(0),atol=1e-12)

def test_new_view_has_additional_end_face_points():
 ix=indices();assert len(ix)==297
 assert set(ix[:9]).isdisjoint(ix[27:54])
 np.testing.assert_array_equal(ix[27:54],np.arange(216,243))
 ex=Experiment.__new__(Experiment)
 np.testing.assert_array_equal(ex.action_indices(6),np.arange(27,54))

def test_expected_value_ignores_unacquired_measurements():
 from scipy.interpolate import RegularGridInterpolator
 from scipy.sparse.linalg import spsolve
 from scripts.phystwin.v08_reference import BOUNDS
 rng=np.random.default_rng(5);axes=[np.linspace(*b,4) for b in BOUNDS];interps=[RegularGridInterpolator(axes,rng.normal(0,.001,(4,4,4,297)),method='cubic',solver=spsolve) for _ in range(3)]
 ex=Experiment(interps,5,np.zeros(297),np.eye(297)*1e-7,'cpu');a=np.zeros(54);b=a.copy();b[9:]=1e3
 ra,ca=ex.choose([],a,samples=4);rb,cb=ex.choose([],b,samples=4)
 np.testing.assert_array_equal(ra,rb);assert ca==cb
 p=ex.predict([6],a);assert p['mean'].shape==(1,243) and np.isfinite(p['sd']).all()

def test_incremental_loading_matches_direct_equilibrium():
 from scripts.phystwin.consistent_fixture import ConsistentFixture
 m=StrongFixture(2,4000,500,1000,'crossed',device='cpu');m.linear_solver='superlu';m.set_action('held_oblique');target=m.load.copy()
 direct=ConsistentFixture.solve(m);state=None
 for f in [.25,.5,.75,1.]:
  m.load=target*f;state=ConsistentFixture.solve(m,initial=None if state is None else state['displacement_local_m'])
 np.testing.assert_allclose(state['displacement_local_m'],direct['displacement_local_m'],atol=1e-9,rtol=1e-6)
 assert np.linalg.norm(state['residual'])<1e-10
 np.testing.assert_array_equal(m.load,target)

def test_selection_ensemble_cannot_see_dense_or_future_truth():
 from scripts.phystwin.v08_physbridge import restrict_selection_prefix
 x=points();data=dict(object_points=np.full((31,81,3),999.),object_visibilities=np.ones((31,81),bool),object_motions_valid=np.ones((30,81),bool))
 y=np.arange(9)*.0001;restrict_selection_prefix(data,y);visible=data['object_visibilities'][0];assert visible.sum()==3
 np.testing.assert_allclose(data['object_points'][:,~visible],np.broadcast_to(x[~visible],(31,78,3)))
 np.testing.assert_allclose(data['object_points'][10,visible]-x[visible],y.reshape(3,3))
 assert np.all(data['object_motions_valid'].sum(1)==3)
 np.testing.assert_allclose(data['object_points'][0],x)
