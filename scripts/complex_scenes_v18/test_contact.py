import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import normal_contact_response
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.complex_scenes_v18.geometry import SceneBody,RigidBowl,RigidCylinder
from scripts.complex_scenes_v18.pair_contact import DeformablePairContact,material_surface,inside_closed_surface,closest_squared


def body():
    return SceneBody(GeometrySpec(dimensions=(.12,.06,.06)),n=1,order=3,support_mode='free',bulk_energy=StableNeoHookean(),surface_order=3)


def test_closed_mesh_and_exact_point_triangle_distances():
    q,tri=material_surface(5)
    edges={}
    for face in tri:
        for i,j in zip(face,np.roll(face,-1)):
            key=tuple(sorted([i,j]));edges.setdefault(key,[]).append((i,j))
    assert all(len(e)==2 and e[0]==e[1][::-1]for e in edges.values())
    p=torch.tensor([[.5,.5,.5],[1.1,.5,.5],[-.1,0,0]],dtype=torch.float64)
    x=torch.tensor(q,dtype=torch.float64);f=torch.tensor(tri)
    assert inside_closed_surface(p,x,f).tolist()==[True,False,False]
    np.testing.assert_allclose(closest_squared(p,x,f).numpy(),[.25,.01,.01],atol=1e-14)


def test_pair_force_is_energy_gradient_conserves_force_torque_and_rotates():
    a,b=body(),body();pair=DeformablePairContact([a,b],samples=5,penalty_N_m3=1e6)
    x=[a.rest.copy(),b.rest+np.array([.107,.007,.003])]
    U,F,d=pair.evaluate(x);assert U>0 and d['active_vertices']>0
    assert d['net_contact_force_N']<1e-11 and d['net_contact_torque_Nm']<1e-11
    rng=np.random.default_rng(7);v=[rng.normal(size=z.shape)for z in x];v=[z/np.linalg.norm(z)for z in v]
    eps=1e-7
    deriv=(pair.evaluate([z+eps*w for z,w in zip(x,v)])[0]-pair.evaluate([z-eps*w for z,w in zip(x,v)])[0])/(2*eps)
    np.testing.assert_allclose(deriv,-sum(np.sum(f*w)for f,w in zip(F,v)),rtol=1e-5,atol=1e-8)
    Q=Rotation.from_rotvec([.3,-.2,.5]).as_matrix();offset=np.array([.2,-.7,.4])
    Ur,Fr,_=pair.evaluate([z@Q.T+offset for z in x]);np.testing.assert_allclose(Ur,U,rtol=1e-10)
    for f,fr in zip(F,Fr):np.testing.assert_allclose(fr,f@Q.T,rtol=1e-9,atol=1e-10)
    U,F,d=pair.evaluate([a.rest,b.rest+np.array([.3,0,0])]);assert U==0 and not any(np.any(f)for f in F)


def test_curved_contact_gradient_and_no_tension():
    for c,p in [(RigidBowl(),[[.06,0,-.081],[.06,0,0]]),(RigidCylinder(),[[.06,0,-.026],[.06,0,.1]])]:
        x=torch.tensor(p,dtype=torch.float64)
        W,F,H,g=normal_contact_response(c,x,tangent=True)
        assert g[0]<0 and g[1]>0 and W[0]>0 and W[1]==0
        np.testing.assert_array_equal(F[1],np.zeros(3))
        eps=1e-7
        for k in range(3):
            delta=torch.zeros_like(x);delta[0,k]=eps
            fd=(normal_contact_response(c,x+delta)[0].sum()-normal_contact_response(c,x-delta)[0].sum())/(2*eps)
            np.testing.assert_allclose(fd,F[0,k],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(H,H.transpose(-1,-2),atol=1e-7)


def test_scene_matches_frozen_single_body_integrator():
    from scripts.compositional_dynamics_v17.dynamics import VelocityVerlet
    from scripts.complex_scenes_v18.scene import CoupledScene
    a,b=body(),body();a.gravity[:]=[0,0,-9.81];b.gravity[:]=[0,0,-9.81]
    a.set_controls(np.zeros((2,6)));b.set_controls(np.zeros((2,6)))
    old=VelocityVerlet(a);new=CoupledScene([b]);s=old.initial_state();t=new.initial()
    for _ in range(20):s=old.step(s,1e-4);t=new.step(t,1e-4)
    np.testing.assert_allclose(s.u,t.u[0],atol=1e-14);np.testing.assert_allclose(s.v,t.v[0],atol=1e-14)
    np.testing.assert_allclose(old.energy(s),new.energy(t),atol=1e-13)


def test_two_body_scene_momentum_conservation_with_contact():
    from scripts.complex_scenes_v18.scene import CoupledScene
    a,b=body(),body();pair=DeformablePairContact([a,b],samples=5,penalty_N_m3=1e6)
    scene=CoupledScene([a,b],pair_contact=pair);s=scene.initial([np.zeros_like(a.rest),np.tile([.115,.007,.003],(len(b.rest),1))]);P=scene.momentum(s);E=scene.energy(s)
    for _ in range(12):s=scene.step(s,1e-6)
    np.testing.assert_allclose(scene.momentum(s),P,atol=1e-12)
    assert abs(scene.energy(s)-E)<1e-6
