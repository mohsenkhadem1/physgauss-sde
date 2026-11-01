import numpy as np
from experiments.gaussian_active_design import score_actions
from scripts.physical_refinement_v15.risk import score_query


def test_direct_moments_match_original_gram_acquisition_with_common_random_draws():
    rng=np.random.default_rng(12)
    for particles,dimensions in [(7,13),(100,5)]:
        target=rng.normal(size=(particles,dimensions));means=rng.normal(size=(particles,4))
        factor=rng.normal(size=(4,4));cov=factor@factor.T+np.eye(4)
        weights=rng.uniform(size=particles);weights[0]=0;weights/=weights.sum()
        old=score_actions(target,means[None],cov,weights,fantasies=256,seed=99)
        new=score_query(target,means,cov,weights,fantasies=256,seed=99)
        for key in ('expected_risk_reduction','score_standard_error','expected_effective_sample_size'):
            np.testing.assert_allclose(new[key],old[key][0],rtol=1e-10,atol=1e-12)
