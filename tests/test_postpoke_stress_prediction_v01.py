import numpy as np
import torch
from experiments.score_postpoke_stress_prediction_v01 import target,cohort,DESIGN


def test_independent_truth_oracle_matches_original135_feature_order():
    torch.set_num_threads(1)
    theta=cohort.physical(np.array([[.4,.6,.3,.7]]))
    expected,_=cohort.joint_features(cohort.old.ArbitraryForceForward(),theta)
    np.testing.assert_allclose(target(theta[0]),expected[0],atol=1e-13,rtol=1e-11)
    assert DESIGN['no_policy_comparison']and DESIGN['all_numerical_failures_retained']
