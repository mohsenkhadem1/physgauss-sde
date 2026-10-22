import numpy as np
from scipy.integrate import quad,cubature
from experiments.campaign_c38_beta_metrics import *
from experiments.campaign_c38_bar_posterior_z import BarPosteriorBatch
from experiments.campaign_c38_beta_posterior import joint_log_prob
import torch


def test_joint_density_and_reference_jacobian_independent_integral():
    params=dict(logits=np.zeros((1,16)),alpha=np.ones((1,16,2)),beta=np.ones((1,16,2)))
    u=np.array([[.2,.3],[.9,.1]])
    np.testing.assert_allclose(log_density_numpy(u,params),0,atol=1e-14)
    posterior=BarPosteriorBatch([[.005]],[1.],[0],rtol=1e-9,conditional_order=64)
    answer=cubature(lambda x:np.exp(reference_log_density(x,posterior,0)),np.zeros(2),np.ones(2),rule='gk21',rtol=1e-8,atol=1e-12)
    assert answer.status=='converged' and abs(answer.estimate-1)<1e-8
    kl,_=reference_joint_kl(posterior,0,params,0,outer_order=24,conditional_order=64)
    expected=quad(lambda x:float(np.exp(reference_log_density(np.array([[.5,x]]),posterior,0))[0]*reference_log_density(np.array([[.5,x]]),posterior,0)[0]),0,1,epsabs=1e-9)[0]
    assert abs(kl-expected)<1e-7 and kl>0


def test_numpy_torch_log_density_skewed_components_match():
    params=dict(logits=np.array([[0.,-1.]]),alpha=np.array([[[.01,2000.],[20.,.02]]]),beta=np.array([[[2.,1500.],[.03,3.]]]))
    x=np.array([[.001,.5],[.9,.01]])
    expanded={k:torch.tensor(np.repeat(v,2,axis=0)) for k,v in params.items()}
    np.testing.assert_allclose(log_density_numpy(x,params),joint_log_prob(torch.tensor(x),expanded),atol=1e-10)


def test_calibration_counts_objects_not_contexts_and_rejects_collapse():
    B=128;params=dict(logits=np.zeros((B*8,16)),alpha=np.ones((B*8,16,2)),beta=np.ones((B*8,16,2)))
    u=np.column_stack([(np.arange(B)+.5)/B,1-(np.arange(B)+.5)/B])
    report,_,_=parameter_diagnostics(params,u)
    assert report['independent_objects_per_context']==B and report['calibration_gate_passed']
    expected=np.exp(LOWER+(UPPER-LOWER)*.95)-np.exp(LOWER+(UPPER-LOWER)*.05)
    for j,key in enumerate(('E','K')):assert abs(report['context_results'][0][key]['mean_interval_width_physical']-expected[j])<1e-6
    bad={**params,'alpha':np.ones((B*8,16,2))*2000,'beta':np.ones((B*8,16,2))*2000}
    report,_,_=parameter_diagnostics(bad,u);assert not report['calibration_gate_passed']


def test_two_sensor_correlated_reference_is_normalized_in_intrinsic_measure():
    from experiments.campaign_c38_bar_reference import prior_physical,endpoint_displacements
    ends=endpoint_displacements(prior_physical(np.array([[.42,.63]])),[1.])[0,0]
    posterior=BarPosteriorBatch([ends+[.00007,-.00005]],[1.,1.],[0,1],rtol=1e-9,atol=1e-13,conditional_order=128)
    result=cubature(lambda x:np.exp(reference_log_density(x,posterior,0)),np.zeros(2),np.ones(2),rule='gk21',rtol=1e-8,atol=1e-12)
    assert result.status=='converged' and abs(result.estimate-1)<2e-8
