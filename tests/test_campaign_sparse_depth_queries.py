"""Independent rest-ray parity and acquired-time isolation for sparse views."""
import numpy as np
import pytest

from experiments.boundary_block import block_mesh
from experiments.campaign_sparse_depth_queries import make_queries,predict_ranges


def test_public_rest_queries_match_independent_depth_reference():
    rest,_,faces = block_mesh(cells=(2,2,2))
    queries = make_queries(rest,faces,frame_indices=(20,),offsets_deg=((0.,0.),(60.,10.)),grid_size=4)
    positions = np.broadcast_to(rest,(2,61,*rest.shape)).copy()
    observed = predict_ranges(positions,faces,queries)
    assert observed.visible.shape==(2,32)
    np.testing.assert_array_equal(observed.visible[0],np.concatenate([q['rest_visible'] for q in queries]))
    np.testing.assert_allclose(observed.residual_m,0.,atol=2e-13,rtol=0)
    assert 0<observed.visible.sum()<observed.visible.size
    for query in queries:
        R = query['pose'][:3,:3]
        np.testing.assert_allclose(R@R.T,np.eye(3),atol=3e-16,rtol=0)
        assert np.isfinite(query['rest_depth_m']).all()
    assert not np.array_equal(queries[0]['pose'],queries[1]['pose'])


def test_only_acquired_frames_affect_likelihood_inputs():
    rest,_,faces = block_mesh(cells=(2,2,2))
    queries = make_queries(rest,faces,frame_indices=(10,20),grid_size=4)
    positions = np.broadcast_to(rest,(1,61,*rest.shape)).copy()
    baseline = predict_ranges(positions,faces,queries)
    positions[:,60] += .03
    unchanged = predict_ranges(positions,faces,queries)
    np.testing.assert_array_equal(baseline.visible,unchanged.visible)
    np.testing.assert_array_equal(baseline.residual_m,unchanged.residual_m)
    positions[:,20] += .002*queries[-1]['pose'][2,:3]
    changed = predict_ranges(positions,faces,queries)
    assert np.max(np.abs(changed.residual_m-baseline.residual_m))>.0005


@pytest.mark.parametrize('kwargs',[{'frame_indices':(20,20)},{'frame_indices':(61,)},{'grid_size':100}])
def test_invalid_query_plans_fail_before_observing_motion(kwargs):
    rest,_,faces = block_mesh(cells=(2,2,2))
    with pytest.raises(ValueError):make_queries(rest,faces,**kwargs)
