from experiments.run_p2_two_level_static import original_physics

def test_original_fine_gate_does_not_admit_only_good_residual():
 a=dict(original_free_residual_N=1e-12,force_balance_defect_N=1e-12,actual_moment_balance_defect_Nm=1e-12,maximum_constraint_error_m=0.,minimum_J=.9,global_affine_F_positive_J_sufficient_lower_bound=.2)
 assert original_physics(a)
 for key,value in [('original_free_residual_N',2e-9),('force_balance_defect_N',2e-8),('actual_moment_balance_defect_Nm',2e-8),('maximum_constraint_error_m',1e-20),('minimum_J',.1),('global_affine_F_positive_J_sufficient_lower_bound',None)]:
  b=a.copy();b[key]=value;assert not original_physics(b)
