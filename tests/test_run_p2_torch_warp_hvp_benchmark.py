from experiments.run_p2_torch_warp_hvp_benchmark import gates,GPU

def test_original_operator_and_candidate_gates_unchanged():
 r=dict(old_HVP_true_R_N=1e-11,original_candidate_free_R_N=1e-11,maximum_constraint_m=0.,minimum_J=.9,positive_Bernstein=True,maximum_candidate_vs_exact80_m=1e-9)
 assert gates(r)
 for key,value in [('old_HVP_true_R_N',1.1e-10),('original_candidate_free_R_N',1.1e-10),('maximum_constraint_m',1e-20),('minimum_J',.1),('positive_Bernstein',False),('maximum_candidate_vs_exact80_m',1.1e-8)]:assert not gates({**r,key:value})
 assert GPU=='GPU-571378aa-c44e-ffb8-b346-1e409f9821f9'
