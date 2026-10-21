import json
import numpy as np
import pytest
import torch
from experiments.campaign_projected_point_targets import ProjectedPointTargets


def toy(seed=73):
    rng=np.random.default_rng(seed);P,T,N,R=5,2,4,3;D=T*N*3
    pins=np.array([0,1,1,3,0],dtype=np.int64)
    X=rng.normal(size=(P,T,N,3))*.01
    for p,pin in enumerate(pins):X[p,:,pin]=0
    m=rng.normal(size=D)*.003;B=rng.normal(size=(D,R))*.005
    w=rng.uniform(size=(3,P));w/=w.sum(1)[:,None];a=rng.normal(size=(3,P,R))*.4
    Cs=[];groups=[]
    for G in [2,3,4]:
        L=rng.normal(size=(G,R,R))*.2;Cs.append(L@L.transpose(0,2,1));groups.append(rng.integers(G,size=P,dtype=np.int64))
    return X.reshape(P,D),m,B,pins,(T,N,3),w,a,Cs,groups


def dense(X,m,B,pins,layout,w,a,C,groups):
    """Independent explicit perparticle fullcoordinate projection/moment oracle."""
    mu=[];diagonal=[];traces=[]
    T,N,_=layout
    for p,pin in enumerate(pins):
        indices=np.array([(t*N+int(pin))*3+c for t in range(T) for c in range(3)])
        mp=m.copy();Bp=B.copy();mp[indices]=0;Bp[indices]=0
        mu.append(X[p]+mp+Bp@a[p])
        cov=Bp@C[groups[p]]@Bp.T
        diagonal.append(np.diag(cov));traces.append(np.trace(cov)/len(m))
    mu=np.stack(mu);diagonal=np.stack(diagonal);mean=w@mu
    between=w@((mu-mean)**2);within=w@diagonal
    return mean,between,within,np.array(traces)


def make(values,device='cpu'):
    X,m,B,pins,layout,*_=values
    return ProjectedPointTargets(torch.tensor(X,device=device),torch.tensor(m,device=device),torch.tensor(B,device=device),torch.tensor(pins,device=device),layout)


def test_chunked_moments_match_dense_projected_gaussian_mixture_total_variance():
    v=toy();X,m,B,pins,layout,w,a,Cs,groups=v;model=make(v)
    expected=dense(X,m,B,pins,layout,w[0],a[0],Cs[0],groups[0])
    result=model.moments(w[0],a[0],Cs[0],groups[0],feature_chunk=5)
    np.testing.assert_allclose(result['mean_displacement'],expected[0],atol=1e-16,rtol=1e-12)
    np.testing.assert_allclose(result['between_particle_variance'],expected[1],atol=1e-18,rtol=1e-12)
    np.testing.assert_allclose(result['within_gp_variance'],expected[2],atol=1e-18,rtol=1e-12)
    np.testing.assert_allclose(result['std'].square(),expected[1]+expected[2],atol=1e-18,rtol=1e-12)


def test_fantasy_risk_ragged_covariance_and_normalized_trace_equal_dense_without_FPD():
    v=toy();X,m,B,pins,layout,w,a,Cs,groups=v;model=make(v)
    expected=[dense(X,m,B,pins,layout,w[f],a[f],Cs[f],groups[f]) for f in range(3)]
    result=model.risk(w,a,Cs,groups,feature_chunk=7)
    risk=np.array([(r[1]+r[2]).mean() for r in expected])
    np.testing.assert_allclose(result['risk'],risk,atol=1e-18,rtol=1e-12)
    trace=np.stack([r[3] for r in expected]);other=model.risk(w,a,conditional_trace=trace,feature_chunk=9)
    torch.testing.assert_close(other['risk'],result['risk'],atol=1e-18,rtol=1e-12)
    assert result['risk'].shape==(3,)


def test_shared_and_stacked_groups_agree_with_moments_average_variance():
    v=toy();*_,w,a,Cs,groups=v;model=make(v)
    shared=model.risk(w,a,Cs[0],groups[0],feature_chunk=4)
    stacked=model.risk(w,a,np.stack([Cs[0]]*3),np.stack([groups[0]]*3))
    torch.testing.assert_close(stacked['risk'],shared['risk'],atol=1e-18,rtol=1e-12)
    for f in range(3):
        moments=model.moments(w[f],a[f],Cs[0],groups[0],feature_chunk=3)
        torch.testing.assert_close(shared['risk'][f],moments['total_variance'].mean(),atol=1e-18,rtol=1e-12)


def test_common_hard_pin_has_exact_zero_mean_and_variance_with_nonzero_global_GP():
    X,m,B,pins,layout,w,a,Cs,groups=toy();pins[:]=0;X[:]=0
    model=ProjectedPointTargets(X,m,B,pins,layout)
    result=model.moments(w[0],a[0],Cs[0],groups[0],feature_chunk=5)
    for key in ['mean_displacement','std','between_particle_variance','within_gp_variance']:
        field=result[key].reshape(layout)
        assert torch.count_nonzero(field[:,0])==0
    X[0,0]=1e-15
    with pytest.raises(ValueError,match='already be exactlyzero'):ProjectedPointTargets(X,m,B,pins,layout)


def test_no_substantive_covariance_or_weight_repair():
    v=toy();*_,w,a,Cs,groups=v;model=make(v)
    bad=Cs[0].copy();bad[0]=np.diag([-1.,1.,1.])
    with pytest.raises(ValueError,match='PSD'):model.moments(w[0],a[0],bad,groups[0])
    with pytest.raises(ValueError,match='never renormalized'):model.moments(w[0]*.99,a[0],Cs[0],groups[0])
    with pytest.raises(ValueError,match='nonnegative normalized'):model.risk(w,a,conditional_trace=np.full_like(w,-1e-6))


def test_factor_confined_to_projected_pin_rows_has_only_roundoff_risk_and_exact_pointwise_zero():
    X,m,B,pins,layout,w,a,Cs,groups=toy();X[:]=0;m[:]=0;pins[:]=0
    shaped=B.reshape(*layout,3);shaped[:,1:]=0
    model=ProjectedPointTargets(X,m,B,pins,layout)
    moments=model.moments(w[0],a[0],Cs[0],groups[0])
    assert torch.count_nonzero(moments['mean_displacement'])==0
    assert torch.count_nonzero(moments['std'])==0
    risk=model.risk(w,a,Cs,groups)['risk']
    assert torch.isfinite(risk).all() and float(risk.max())<1e-18


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count()<3,reason='GPU2 unavailable')
def test_synthetic_CPU_GPU2_parity():
    v=toy();*_,w,a,Cs,groups=v;cpu=make(v);gpu=make(v,'cuda:2')
    c=cpu.moments(w[0],a[0],Cs[0],groups[0],feature_chunk=5);g=gpu.moments(w[0],a[0],Cs[0],groups[0],feature_chunk=5)
    cr=cpu.risk(w,a,Cs,groups,feature_chunk=7);gr=gpu.risk(w,a,Cs,groups,feature_chunk=7)
    error=max(float((g[k].cpu()-c[k]).abs().max()) for k in ['mean_displacement','std','total_variance'])
    risk_error=float((gr['risk'].cpu()-cr['risk']).abs().max())
    assert error<1e-14 and risk_error<1e-16
    print(json.dumps(dict(GPU2_mean_std_variance_max_absolute_error=error,GPU2_risk_max_absolute_error=risk_error,synthetic_only=True)))
