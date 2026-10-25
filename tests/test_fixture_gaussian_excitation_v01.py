import numpy as np
from experiments import diagnose_fixture_gaussian_excitation_v01 as m


def test_coordinate_not_Euclidean_threshold_keeps_physics_and_missingfraction_separate():
    rows=[]
    for x in m.m.mechanics.worklist():
        nr=np.full(11,100e-6)
        if x['E_Pa']==8000:nr[1:9]=80e-6
        rows.append(dict(x,nonrigid_coordinate_RMS_m=nr.tolist(),nonrigid_Euclidean_RMS_m=(nr*np.sqrt(3)).tolist(),minimum_sampled_linear_J=[.99]*11,maximum_linear_symmetric_strain=[.01]*11,equilibrium=[dict(passed=True) for _ in range(11)],exact_clamp_zeros=True,free_residual_l2_N=1e-14))
    # 80um coordinate >100um Euclidean: wrongunits wouldmiss ALL96 failures.
    result=m.summary(rows)
    for n in ('3','6'):
        assert result[n]['candidate_below100um']==96
        assert result[n]['states_with_low_candidate']==12
        assert result[n]['candidate_low_witnesses']['all_original_linear_equilibrium_passed']
        assert result[n]['fraction_status']=='NOT_RETAINED' and not result[n]['fraction_goal_evaluated']
        assert result[n]['mapped_five_parameter_catalog']['below100um_by_role']=={'candidate':64}
