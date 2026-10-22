import numpy as np
import torch
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_torch import TorchP2
from experiments.campaign_block_surface_controls import surface_control,SIZE
from experiments.campaign_static_p2_block import EDGES,shape_gradients


def test_uniform_factory_reproduces_original_geometry_force_and_full_mass():
    old=StableP2Block((4,2,2),young=4500,gravity=0,traction=1,order=3)
    new=P2Geometry((4,2,2),young=4500,pattern='transverse_end')
    old.load,_=surface_control(old.rest,old.tets,old.faces,'transverse_end',old.elements)
    for name in ['rest','tets','elements','faces','linear_gradients','volumes','mask','load']:np.testing.assert_array_equal(getattr(new,name),getattr(old,name))
    a,b=TorchP2(old),TorchP2(new);v=torch.tensor(np.random.default_rng(98).normal(size=new.rest.shape));torch.testing.assert_close(a.mass(v),b.mass(v),atol=0,rtol=0)
    assert not hasattr(new,'rows') and not hasattr(new,'cols')


def test_graded_reference_is_straight_same_solid_conservative_and_affine_exact():
    m=P2Geometry((4,2,2),gravity=9.81,mapping='clamp_graded');corners=m.rest[m.tets]
    for k,(i,j) in enumerate(EDGES):np.testing.assert_allclose(m.rest[m.elements[:,4+k]],(corners[:,i]+corners[:,j])/2,atol=0,rtol=0)
    assert np.isclose(m.volumes.sum(),np.prod(SIZE))
    assert np.isclose(m.load[:,2].sum(),-.12-625*np.prod(SIZE)*9.81)
    for b in np.eye(4):
        g=shape_gradients(b[None],m.linear_gradients);F=np.einsum('eai,eaj->eij',m.rest[m.elements],g);np.testing.assert_allclose(F,np.broadcast_to(np.eye(3),F.shape),atol=1e-14,rtol=0)
    # A physical affine displacement remains exact after nonuniform meshing.
    points=np.array([[.03,-.01,.02],[.13,.01,-.02],[.24,.04,.04]])
    q=m.rest@np.diag([1.1,.97,1.02])+[.01,-.02,.03]
    np.testing.assert_allclose(m.query(q,points),points@np.diag([1.1,.97,1.02])+[.01,-.02,.03],atol=1e-14,rtol=0)
