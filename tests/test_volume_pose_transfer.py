import numpy as np
from scipy.spatial.transform import Rotation
from scripts.coupled_contact_v13.physics import make_block
from scripts.coupled_contact_v13.benchmark_transfer.pose import unit_volume_mass,center_inertia,integrate_pose
from scripts.contact_benchmark_v12.scene import StaticScene


def test_volume_moments_exact_block_translation_and_rotation():
    size=np.array([.08,.06,.05]);expected=np.diag([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])/12
    R=Rotation.from_rotvec([.2,.3,-.1]).as_matrix();translation=np.array([.3,.2,.4])
    for cells in [(1,1,1),(2,2,2),(4,4,4)]:
        nodes,tets=make_block(size=size,cells=cells,center=(0,0,0));M=unit_volume_mass(nodes,tets)
        center,inertia=center_inertia(nodes,M);np.testing.assert_allclose(center,0,atol=1e-16);np.testing.assert_allclose(inertia,expected,rtol=1e-13,atol=1e-16)
        center,inertia=center_inertia(nodes@R.T+translation,M)
        np.testing.assert_allclose(center,translation,atol=1e-15);np.testing.assert_allclose(inertia,R@expected@R.T,atol=1e-15)


def test_orientation_order_does_not_change_mass_and_free_fall_shape():
    nodes,tets=make_block(cells=(1,1,1),center=(0,0,1));reverse=tets[:,[0,2,1,3]]
    M=unit_volume_mass(nodes,tets);np.testing.assert_allclose(M.toarray(),unit_volume_mass(nodes,reverse).toarray(),rtol=1e-14,atol=1e-16)
    x,v,w,meta=integrate_pose(nodes,[0,0,0],[0,0,0],.01,StaticScene(floor=0),M,substeps=10)
    np.testing.assert_allclose(v,[0,0,-.0981],atol=1e-14);np.testing.assert_allclose(w,0,atol=1e-14)
    np.testing.assert_allclose(x-x.mean(0),nodes-nodes.mean(0),atol=1e-14)
    assert meta['contact_substeps']==0


def test_held_forecasts_unchanged_and_future_targets_forbidden():
    import pytest
    from scripts.coupled_contact_v13.benchmark_transfer.pose import rollout
    nodes,tets=make_block(cells=(1,1,1),center=(0,0,1));T=6;end=4
    trajectory=np.broadcast_to(nodes,(T,*nodes.shape)).copy()
    payload=dict(dataset='PhysTwin',nodes=nodes,train_end=end,observed=trajectory[:end].copy(),
                 observed_initial=nodes.copy(),observed_mask=np.ones((end,len(nodes)),bool),
                 observation_matrix=np.eye(len(nodes)),times=np.arange(T)/20,
                 attachment_active=np.ones((T,1)),controller_targets=trajectory[:,None].copy())
    result,meta=rollout(payload,trajectory,StaticScene(),tets)
    np.testing.assert_array_equal(result,trajectory);assert not meta['diagnostics']
    with pytest.raises(ValueError,match='Future'):rollout(dict(payload,evaluation={'points':'forbidden'}),trajectory,StaticScene(),tets)
