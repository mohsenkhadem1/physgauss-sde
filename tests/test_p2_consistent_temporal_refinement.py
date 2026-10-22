from experiments.run_p2_consistent_temporal_refinement import eligible_larger_case

def test_frozen_larger_solve_requires_time_signal_and_remaining_budget():
 assert eligible_larger_case([30e-6,10e-6],40.,230.)
 assert not eligible_larger_case([30e-6,21e-6],40.,230.)
 assert not eligible_larger_case([5e-6,10e-6],40.,230.)
 assert not eligible_larger_case([10e-6],40.,230.)
 assert not eligible_larger_case([30e-6,10e-6],70.,230.)
 assert not eligible_larger_case([30e-6,10e-6],65.,230.)
