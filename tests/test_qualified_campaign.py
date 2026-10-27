import numpy as np
import pytest
from scripts.phystwin.qualify_fixture import points
from scripts.phystwin.qualified_reference import checked_compare,sample81
from scripts.phystwin.consistent_fixture import P2Fixture


def test_support_sampling_tracks_side_and_extent():
    for side,x in [('left',0.),('right',.24)]:
        for extent in ['full','lower']:
            _,_,p=points(side+'_'+extent)
            assert np.all(p[:,0]==x)
            assert p[:,2].max() < (0. if extent=='lower' else .04)
            if extent=='full':assert p[:,2].max()>0
    assert sample81(points()[0]).shape==(81,3)


def test_analytic_zero_strain_gate_does_not_hide_deformation_error():
    import json
    from scripts.phystwin.qualified_reference import ROOT
    spec={'qualification':json.loads((ROOT/'reports/phystwin-fixture-convergence-v04/protocol.json').read_text())}
    r=dict(layout='left_full',K=800.)
    b=dict(sampled_u=np.tile([0,.03/800,0],(425,1)),strain=np.ones((512,3,3))*1e-17,
        traction=np.tile([0,-.03/.0064,0],(64,1)),elastic_J=1e-30,spring_J=1e-6,
        force_balance_N=1e-14,moment_balance_Nm=1e-14,residual_N=1e-12,minimum_J=1.)
    a={**b,'strain':b['strain']*2,'elastic_J':2e-30}
    assert checked_compare(a,b,spec,r,'push_left')['pass']
    assert not checked_compare(a,b,spec,r,'push_right')['pass']
    assert not checked_compare({**a,'strain':a['strain']+1e-8},b,spec,r,'push_left')['pass']
    assert not checked_compare({**a,'sampled_u':a['sampled_u']+1e-6},b,spec,r,'push_left')['pass']


def test_gpu_multigrid_preserves_nonlinear_solutions_across_loads():
    pytest.importorskip('pyamg')
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=P2Fixture(2,2000,600,'right_lower',grading=True,device='cuda:0');m.linear_solver='gpu_multigrid'
    try:
        for action in ['initial','push_left','held_oblique']:
            m.set_action(action);actual=m.solve();fresh=P2Fixture(2,2000,600,'right_lower',grading=True);fresh.set_action(action);expected=fresh.solve()
            assert np.linalg.norm(actual['residual'])<=1e-10
            assert np.allclose(actual['displacement_local_m'],expected['displacement_local_m'],atol=1e-9,rtol=1e-8)
    finally:m.clear_linear_cache()


def test_physical_audit_detects_wrong_confidence_after_probe():
    from scripts.phystwin.qualified_policy import physical_score,convergence
    spec=dict(test=[dict(E=1.,K=1.,layout='left_full')],noise_repeats=1,quadrature_gate_mean_sd_m=1e-5,quadrature_gate_log_quantile=.02)
    before=dict(log_parameter_quantiles=np.array([[[-1.,0.,1.],[-1.,0.,1.]]]),layout_probability=np.array([[1.,0.,0.,0.]]),mean=np.zeros((1,3)),sd=np.ones((1,3)))
    after={**before,'log_parameter_quantiles':np.array([[[.2,.3,.4],[.2,.3,.4]]])}
    result=physical_score([before,after,after],spec,'test',dict(initial=np.array([0]),adaptive=np.array([1]),random=None))
    assert result['initial']['E_90_coverage']==1 and result['adaptive']['E_90_coverage']==0
    assert result['adaptive']['E_mean_log_interval_width']<result['initial']['E_mean_log_interval_width']
    assert result['adaptive']['E_log_median_MAE']>result['initial']['E_log_median_MAE']
    assert not convergence([before],[after],spec)['passed']
