import numpy as np
from scripts.phystwin.active_support_reference import ActiveBlock
from scripts.phystwin.active_support_policy import Mixture


def test_half_face_area_and_load_conservation():
    full=ActiveBlock(2,3000,800,'left_full');half=ActiveBlock(2,3000,800,'right_lower')
    assert np.isclose(half.attachment_area,full.attachment_area/2)
    assert np.isclose(half.support_weights.sum(),1)
    assert np.all(half.rest[half.support_weights>0,2]<=1e-13)
    half.set_action('push_left')
    assert np.allclose(half.load.sum(0),[0,.03,0])


def test_partial_fixture_energy_tangent_and_force_balance():
    m=ActiveBlock(2,3000,800,'left_lower');m.set_action('held_oblique')
    rng=np.random.default_rng(2);u=rng.normal(size=m.rest.shape)*1e-5;d=rng.normal(size=u.shape);d/=np.linalg.norm(d)
    h=1e-6;s=m.evaluate(u,True);a=m.evaluate(u+h*d);b=m.evaluate(u-h*d)
    assert np.isclose((a['potential_J']-b['potential_J'])/(2*h),np.sum(s['residual']*d),rtol=1e-6,atol=1e-9)
    assert np.allclose((a['residual']-b['residual']).ravel()/(2*h),s['tangent']@d.ravel(),rtol=1e-5,atol=1e-6)
    solution=m.solve();u=solution['displacement_local_m']
    assert np.linalg.norm((m.K*m.support_weights[:,None]*u-m.load).sum(0))<1e-9


def test_expected_risk_for_single_gaussian_is_analytic():
    fields=np.zeros((4,2));prior=np.ones(4)/4;C=np.array([[4.,1.],[1.,2.]])
    m=Mixture(fields,prior,np.zeros(2),C,[0],[1],1.,'cpu')
    # Var(target | observed) = 2 - 1^2/(4+1), independently of observed value.
    assert np.allclose(m.risk(np.array([[-2.],[0.],[3.]])),1.8)
    result=m.infer(np.array([[3.]]))
    assert np.allclose(result['mean'],.6)
    assert np.allclose(result['sd']**2,1.8)


def test_probe_informativeness_and_utility_scoring_consistency():
    fields=np.array([[-.01,-.02],[-.01,-.02],[.01,.02],[.01,.02]])
    m=Mixture(fields,np.ones(4)/4,np.zeros(2),np.zeros((2,2)),[0],[1],.001,'cpu')
    y=np.array([[-.01],[.01],[0.]])
    result=m.infer(y)
    assert np.allclose(m.risk(y),np.mean(result['sd']**2,axis=1),atol=1e-14)
    assert m.risk(y)[0]<1e-12 and np.isclose(m.risk(y)[2],.0004)
    m.weight_batch_size=2
    m.moment_chunk_size=2
    batched=m.infer(y)
    for key in result:assert np.allclose(batched[key],result[key],atol=1e-14)
    assert np.allclose(m.risk(y),np.mean(result['sd']**2,axis=1),atol=1e-14)


def test_multidevice_posterior_and_expected_risk_match_cpu():
    import pytest,torch
    from scripts.phystwin.active_support_policy import Experiment
    if torch.cuda.device_count()<3:pytest.skip('Three CUDA devices required')
    axis=np.linspace(0,2*np.pi,972)
    def field(nodes):return 1e-4*(nodes[:,0,None]*np.sin(axis)+nodes[:,1,None]*np.cos(axis))
    interp=[field]*4;bias=np.zeros(972);C=np.eye(972)*1e-10
    cpu=Experiment(interp,3,bias,C,.001,'cpu');gpu=Experiment(interp,3,bias,C,.001,'cuda:0,cuda:1,cuda:2')
    assert [str(m.M.device) for m in gpu.models]==['cuda:0','cuda:1','cuda:2']
    rng=np.random.default_rng(15);ys=[rng.normal(0,.001,(3,len(m.chol))) for m in cpu.models]
    for a,b in zip(cpu.predict(ys),gpu.predict(ys)):
        for key in a:assert np.allclose(a[key],b[key],atol=1e-12,rtol=1e-10)
    parallel=gpu.choose(ys[0][0],256);gpu.parallel_actions=False
    assert np.allclose(gpu.choose(ys[0][0],256),parallel,atol=1e-12,rtol=1e-10)
    assert np.allclose(cpu.choose(ys[0][0],256),parallel,atol=1e-12,rtol=1e-10)


def test_memory_bounded_weights_match_gaussian_density_reference():
    from scipy.special import softmax
    from scipy.stats import multivariate_normal
    rng=np.random.default_rng(81);fields=rng.normal(0,.001,(12,6));bias=rng.normal(0,.0001,6)
    A=rng.normal(size=(6,6));C=1e-8*(A@A.T);sigma=.0003;prior=rng.uniform(.1,1,12);prior/=prior.sum()
    y=rng.normal(0,.001,(7,3));model=Mixture(fields,prior,bias,C,[0,1,2],[3,4,5],sigma,'cpu')
    S=C[:3,:3]+sigma**2*np.eye(3)
    log_density=np.column_stack([multivariate_normal(mean=f[:3]+bias[:3],cov=S).logpdf(y) for f in fields])
    expected=softmax(log_density+np.log(prior)[None,:],axis=1)
    assert np.allclose(model.weights(y).numpy(),expected,atol=1e-12,rtol=1e-10)
    from scripts.phystwin.audit_qualified_oracle import posterior_mean
    model.weight_batch_size=2
    assert np.allclose(posterior_mean(model,y),model.infer(y)['mean'],atol=1e-14)


def test_exact_spline_mean_preserves_conditioned_prediction_and_risk():
    import torch
    from scipy.interpolate import RegularGridInterpolator
    from scipy.sparse.linalg import spsolve
    from scripts.phystwin.active_support_policy import Experiment,BOUNDS
    rng=np.random.default_rng(811);axes=[np.linspace(*b,9) for b in BOUNDS]
    interp=[RegularGridInterpolator(axes,rng.normal(0,.001,(9,9,972)),method='cubic',solver=spsolve) for _ in range(4)]
    bias=rng.normal(0,.00001,972);factor=rng.normal(0,.00001,(972,3));C=factor@factor.T+np.eye(972)*1e-10
    dense=Experiment(interp,17,bias,C,.0003,'cpu',tensor_mean=False)
    fast=Experiment(interp,17,bias,C,.0003,'cpu')
    weights=rng.uniform(size=(5,4*17**2));weights/=weights.sum(1,keepdims=True)
    for m,reference in zip(fast.models,dense.models):
        assert m.tensor_basis is not None
        assert np.allclose(m.target_mean(m.tensor(weights)).numpy(),weights@reference.M.numpy(),atol=1e-14,rtol=1e-10)
    ys=[rng.normal(0,.001,(3,len(m.chol))) for m in dense.models]
    expected=dense.predict(ys);expected_risk=dense.choose(ys[0][0],256)
    devices=['cpu']+(['cuda:0,cuda:1,cuda:2'] if torch.cuda.device_count()>=3 else [])
    for device in devices:
        model=fast if device=='cpu' else Experiment(interp,17,bias,C,.0003,device)
        for a,b in zip(expected,model.predict(ys)):
            for key in a:assert np.allclose(a[key],b[key],atol=1e-12,rtol=1e-9)
        assert np.allclose(expected_risk,model.choose(ys[0][0],256),atol=1e-12,rtol=1e-9)
