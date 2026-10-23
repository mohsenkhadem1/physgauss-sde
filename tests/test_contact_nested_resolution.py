from fractions import Fraction as Q
import numpy as np
import pytest
from experiments import contact_nested_resolution as m


def test_exact_reference_triangle_area_does_not_hide_area_rounding():
 X=np.array([[0.,0.,0.],[.24,0.,0.],[0.,.08,0.]])
 assert m.exact_axis_aligned_area(X)==Q(.24)*Q(.08)/2
 with pytest.raises(ValueError):m.exact_axis_aligned_area(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,1.]]))


def test_kappa_amplification_and_resolution_do_not_waive_fixed_gate():
 assert m.height_gradient_bound(1e-17)>1e-10
 assert m.interpret(2e-10,1e-8).startswith('FAIL_ORIGINAL_RAW_GATE_CAUSE_UNRESOLVED')
 assert m.interpret(1e-12,1e-8)=='PASS_ORIGINAL_RAW_GATE'
 assert m.interpret(2e-8,1e-8).startswith('FAIL_ORIGINAL_RAW_GATE_EXCEEDS')


def test_sparse_dot_error_budget_is_independent_of_a_future_saved_field():
 assert m.gamma(20)>20*np.finfo(float).eps
 with pytest.raises(ValueError):m.gamma(10**20)
