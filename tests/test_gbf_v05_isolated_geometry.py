import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.isolated_geometry_repair import repair,canonical


def data():
    nodes=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.],[.2,.2,.2]])
    return {'nodes':nodes,'support_geometry':nodes.copy(),'observed_initial':nodes.copy(),'observed':np.broadcast_to(nodes,(3,5,3)).copy(),'observed_mask':np.ones((3,5),bool),'times':np.arange(5),'train_end':3,'controller_targets':np.broadcast_to(nodes,(5,1,5,3)).copy(),'attachment_weights':np.ones((1,5)),'observation_matrix':np.eye(5)}


def test_only_isolated_vertex_removed_observations_preserved_and_affine_complete():
    original=data();out,tet,diag=repair(original,np.array([[0,1,2,3]]),np.array([.25]*4+[0]))
    assert out['nodes'].shape==(4,3)and out['observed'].shape==(3,5,3)
    assert np.array_equal(out['observed'],original['observed'])and np.array_equal(out['support_geometry'],original['support_geometry'])
    assert np.allclose(out['observation_matrix']@out['nodes'],out['observed_initial'],atol=1e-10)
    assert np.allclose(out['observation_matrix'].sum(1),1)
    assert out['controller_targets'].shape==(5,1,4,3)and canonical(tet)=={(0,1,2,3)}


def test_connected_zero_mass_vertex_is_not_removed():
    with pytest.raises(ValueError,match='exactly isolated'):repair(data(),[[0,1,2,3]],np.array([.25,.25,.5,0,0]))


def test_positive_volume_component_cannot_be_deleted():
    d=data();d['nodes']=np.vstack([d['nodes'],[[3,0,0],[3,1,0],[3,0,1],[9,9,9]]])
    with pytest.raises(ValueError,match='components'):repair(d,[[0,1,2,3],[4,5,6,7]],np.r_[np.ones(8),0])
