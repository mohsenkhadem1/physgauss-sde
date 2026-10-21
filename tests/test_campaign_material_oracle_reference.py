"""Check evidence normalization and weighted numerical summaries analytically."""
import numpy as np
from experiments.campaign_material_oracle_reference import importance_summary


def test_constant_likelihood_keeps_prior_and_evidence_offset():
    physical = np.array([[1500.,400.,.28,.35],[5000.,625.,.35,.6],[20000.,850.,.42,.85]])
    summary,weights = importance_summary(physical,np.full(3,-1234.5))
    np.testing.assert_allclose(weights,np.ones(3)/3,rtol=0,atol=1e-15)
    np.testing.assert_allclose(summary['log_evidence'],-1234.5,rtol=0,atol=1e-12)
    np.testing.assert_allclose(summary['ESS'],3.,rtol=0,atol=1e-14)
    np.testing.assert_allclose([r['mean'] for r in summary['material']],physical.mean(0))


def test_concentrated_likelihood_reports_degeneracy_without_dropping_particles():
    physical = np.array([[1500.,400.,.28,.35],[5000.,625.,.35,.6],[20000.,850.,.42,.85]])
    summary,weights = importance_summary(physical,np.array([-1000.,0.,-1000.]))
    assert summary['particles']==3 and len(weights)==3
    assert summary['ESS']==1 and summary['maximum_weight']==1
    np.testing.assert_allclose([r['mean'] for r in summary['material']],physical[1],atol=0,rtol=0)
