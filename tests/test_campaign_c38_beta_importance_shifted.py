import json
from pathlib import Path
import numpy as np
from scipy.integrate import cubature
from experiments import campaign_c38_beta_importance_shifted as s


def test_shift_changes_only_proposal_and_exact_density_normalizes():
    original=dict(logits=np.zeros((1,16)),alpha=np.broadcast_to([.001,10000.],(1,16,2)).copy(),beta=np.broadcast_to([10000.,.001],(1,16,2)).copy());snapshot={k:v.copy() for k,v in original.items()};p=s.shifted_parameters(original)
    for k in original:np.testing.assert_array_equal(original[k],snapshot[k])
    np.testing.assert_array_equal(p['alpha'],original['alpha']+2);np.testing.assert_array_equal(p['beta'],original['beta']+2);np.testing.assert_array_equal(p['logits'],original['logits'])
    # Moderate shapes for independent adaptive2D normalized-density check.
    moderate=s.shifted_parameters(dict(logits=np.zeros((1,16)),alpha=np.broadcast_to([2.,4.],(1,16,2)),beta=np.broadcast_to([5.,3.],(1,16,2))))
    result=cubature(lambda u:np.exp(s.original.proposal_log_density(u,moderate)),np.zeros(2),np.ones(2),rule='gk21',rtol=1e-9)
    assert result.status=='converged' and abs(result.estimate-1)<1e-9
    u=s.original.sample_proposal(s.original.sobol_points(927,power=14),moderate);expected=.25+.5*np.array([4/11,6/11]);np.testing.assert_allclose(u.mean(0),expected,atol=5e-5)


def test_inverse_cdf_extremes_and_exact_original_failure_parameters():
    u=s.original.sobol_points(2026123351);extreme=s.shifted_parameters(dict(logits=np.zeros((1,16)),alpha=np.broadcast_to([.001,np.exp(12)],(1,16,2)),beta=np.broadcast_to([np.exp(12),.001],(1,16,2))))
    for x in (u,np.array([[.9,.5,1e-15,1-1e-15],[.9,.5,1-1e-15,1e-15]])):
        q=s.original.sample_proposal(x,extreme);assert np.isfinite(q).all() and ((q>0)&(q<1)).all()
    fixture=Path(__file__).resolve().parents[1]/'runs/operator-dev/c38-beta-importance-v01/failure-proposal-fixture.json';actual=json.loads(fixture.read_text())['parameters'];draw=s.original.sample_proposal(u,s.shifted_parameters(actual));assert np.isfinite(draw).all() and ((draw>0)&(draw<1)).all()


def test_uninformative_likelihood_recovers_original_prior_not_shifted_density():
    original=dict(logits=np.zeros((1,16)),alpha=np.broadcast_to([3.,6.],(1,16,2)),beta=np.broadcast_to([5.,4.],(1,16,2)));p=s.shifted_parameters(original);u=s.original.sample_proposal(s.original.sobol_points(193,power=14),p);lw=-s.original.proposal_log_density(u,p);q=np.tile(u[:,0,None],(1,24));result=s.original.summarize_cloud(u,lw,q,[.25,.5,.75])
    assert abs(result['log_evidence'])<3e-4
    np.testing.assert_allclose(result['CDF_grid'],[[.25,.5,.75]]*2,atol=8e-4)
