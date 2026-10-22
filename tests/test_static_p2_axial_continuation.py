import numpy as np
from experiments.run_static_p2_axial_continuation import comparison

def test_conservative_first_order_error_cannot_hide_small_adjacent_difference():
    a=np.zeros((2,3));b=a.copy();b[1,0]=8e-6
    r=comparison(a,b,3)
    assert r['maximum_Euclidean_error_m']<20e-6
    assert not r['passed'] and r['minimum_order_one_fine_error_estimate_m']==24e-6
    b[1,0]=6e-6
    assert comparison(a,b,3)['passed']

def test_query_gate_uses_euclidean_norm_not_coordinate_threshold():
    a=np.zeros((1,3));b=np.full((1,3),15e-6)
    assert not comparison(a,b,1)['passed']
