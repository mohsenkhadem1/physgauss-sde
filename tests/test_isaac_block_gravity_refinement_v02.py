import numpy as np
import pytest
from experiments.isaac_block_gravity_refinement_v02 import temporal_metrics,prepare_output_parent


def capture(delta=0):
 return dict(rest_m=np.zeros((2,3)),tets=np.zeros((1,4),int),faces=np.zeros((1,3),int),fixed_node_ids=np.array([0]),times_s=np.arange(19)/30,positions_m=np.zeros((19,2,3))+np.array([delta,0,0]))

def test_same_material_queries_exact20um_boundary_and_allpairs():
 r=temporal_metrics([capture(),capture(10e-6),capture(20e-6)]);assert r['passed'] and len(r['pairs'])==3
 assert not temporal_metrics([capture(),capture(10e-6),capture(20.01e-6)])['passed']


def test_mismatched_clocks_or_geometry_fail_no_interpolation():
 b=capture();b['times_s'][3]+=.001
 with pytest.raises(AssertionError):temporal_metrics([capture(),b,capture()])
 b=capture();b['rest_m'][0,0]=1e-6
 with pytest.raises(AssertionError):temporal_metrics([capture(),b,capture()])


def test_per_hz_parent_exists_before_worker_creates_fresh_case(tmp_path):
 parent=tmp_path/'hz240';prepare_output_parent(parent)
 case=parent/'gravity_left';case.mkdir();assert case.is_dir()
 with pytest.raises(FileExistsError):prepare_output_parent(parent)
