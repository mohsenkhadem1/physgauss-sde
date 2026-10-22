from experiments.run_newton_stable_p2_spatial import admission
from experiments.run_static_p2_axial_continuation import comparison
import numpy as np

def validrows():return [dict(completed=True,physics_passed=True,excitation_goal_met=True,global_affine_F_positive_J_sufficient_lower_bound=.7) for _ in range(2)]
def test_independent_gates_require_both_rest_solves_certificate_and_excitation():
 r=validrows();assert admission(r,{'passed':True},{'passed':True})['integrated_static_case_passed']
 r[1]['global_affine_F_positive_J_sufficient_lower_bound']=None
 assert not admission(r,{'passed':True},{'passed':True})['static_fixed_probe_qualified']
 r=validrows();r[1]['excitation_goal_met']=False
 a=admission(r,{'passed':True},{'passed':True});assert a['static_fixed_probe_qualified'] and not a['integrated_static_case_passed']
 assert not admission(r[:1],{'passed':True},{'passed':True})['static_fixed_probe_qualified']
def test_minimum_order_one_indicator_not_raw_delta_alone():
 a=np.zeros((1,3));b=np.array([[6e-6,0,0]])
 c=comparison(a,b,4);assert c['maximum_Euclidean_error_m']<20e-6 and not c['passed']
 assert not admission(validrows(),c,{'passed':True})['integrated_static_case_passed']
