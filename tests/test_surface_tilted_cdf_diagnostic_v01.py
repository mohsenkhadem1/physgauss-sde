import numpy as np
from experiments.run_surface_tilted_cdf_diagnostic_v01 import raw_probability,mass_diagnostics
from experiments.campaign_nonlinear_tilted_pwl_v01 import NonlinearTiltedPWL


def test_raw_arithmetic_matches_original_both_sides_without_changing_guard():
    p=NonlinearTiltedPWL(np.array([.4,.6]),np.array([.2,np.inf]),np.array([.1,.1]),np.array([.9,.9]),log_ratio=np.zeros(2),linear_peak=np.zeros(2),full_loglik=lambda u:-np.sum(u*u,axis=1),panels=128)
    for q in(np.array([.1,.9]),np.array([.25,.55])):
        for tail in(False,True):
            raw,stats=raw_probability(p,q,tail)
            assert np.array_equal(raw,p._probability(q,tail=tail))
            assert stats['primitive_bit_arithmetic_maximum_difference']==0
    d=mass_diagnostics(p)
    assert d['density_pairwise_relative']['maximum']==0
    assert d['reflected_width_relative']['maximum']>0
    # A changed normalizer is detected, never repaired; old guard still rejects.
    p.mass0*=.5
    import pytest
    with pytest.raises(FloatingPointError):p._probability(np.array([.899,.899]),tail=False)
    raw,stats=raw_probability(p,np.array([.899,.899]),False)
    assert stats['above_original_guard']>0 and raw.max()>1
