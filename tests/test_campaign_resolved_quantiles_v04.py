import numpy as np
import sympy as sp
import pytest
from experiments import campaign_resolved_quantiles_v04 as m
from experiments import campaign_c38_postprobe_memo as legacy


def test_exact_tail_bypasses_invalid_GL_full_mass_enclosure(monkeypatch):
    monkeypatch.setattr(legacy,'_global_target_bounds',lambda *a:(sp.Rational(1,3),sp.Rational(2,3)))
    def bad(*a,**k):raise AssertionError('No quadrature outside exact support')
    monkeypatch.setattr(legacy,'coordinate_cdf_bounds',bad)
    for left in (False,True):
        assert m.polynomial_support_cdf_bounds(None,{},np.nextafter(float(sp.Rational(1,3)),-np.inf),left_limit=left)['cdf']==0
        assert m.polynomial_support_cdf_bounds(None,{},np.nextafter(float(sp.Rational(2,3)),np.inf),left_limit=left)['cdf']==1


def test_exact_rational_comparison_and_interior_identity(monkeypatch):
    # float(1/3) lies strictly below 1/3; float comparison alone loses this fact.
    lo,hi=sp.Rational(1,3),sp.Rational(2,3)
    monkeypatch.setattr(legacy,'_global_target_bounds',lambda *a:(lo,hi))
    row=dict(cdf=.5,lower=.5,upper=.5);calls=[]
    monkeypatch.setattr(legacy,'coordinate_cdf_bounds',lambda *a,**k:(calls.append((a,k))or row))
    assert m.polynomial_support_cdf_bounds(None,{},float(lo))['cdf']==0
    assert m.polynomial_support_cdf_bounds(None,{},.5,left_limit=False)is row
    assert calls[0][1]==dict(left_limit=False)
    # At exactly representable support endpoints delegation preserves atoms.
    assert m.polynomial_support_cdf_bounds(None,{},0.,support=(sp.Rational(0),sp.Rational(1)))is row


def test_no_interior_guard_waiver(monkeypatch):
    monkeypatch.setattr(legacy,'_global_target_bounds',lambda *a:(sp.Rational(0),sp.Rational(1)))
    invalid=dict(cdf=.6,lower=.7,upper=.8)
    monkeypatch.setattr(legacy,'coordinate_cdf_bounds',lambda *a,**k:invalid)
    assert m.polynomial_support_cdf_bounds(None,{},.5)is invalid
    with pytest.raises(FloatingPointError):m.bounded_continuous_quantile(lambda x:invalid,0.,1.,.5)


def test_actual_all_methods_tiny_quadratic_and_mixture_atoms():
    from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood,MemoPolynomialPosterior
    from experiments.campaign_resolved_quantiles_v02 import polynomial_coordinate_quantile as old
    def model():return BoundsMemoLikelihood(np.zeros((3,1)),np.zeros(1),np.eye(1),young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
    for method,order in [('adaptive',128),('legendre',64),('legendre',128)]:
        post=MemoPolynomialPosterior({'a':model(),'b':model()},method=method,order=order)
        for amp in (1e-3,1e-12):
            target={c:np.array([0.,0.,amp])for c in post.categories}
            for p in (.025,.5,.975):
                row=m.polynomial_coordinate_quantile(post,target,p);prior=old(post,target,p)
                expected=amp*2**(2*(p-1))
                assert row['lower_m']<=expected<=row['upper_m']
                for k in ('value_m','lower_m','upper_m'):assert row[k]==prior[k]
                for k in ('CDF_calls','iterations'):assert row['CDF_bracket'][k]==prior['CDF_bracket'][k]
        mixed={'a':np.array([0.,0.,0.]),'b':np.array([1e-12,0.,0.])}
        assert m.polynomial_coordinate_quantile(post,mixed,.025)['kind']=='category_mixture_atom'


def test_actual_near_quantile_arithmetic_fallback_preserves_law_and_probability_gate(monkeypatch):
    from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood,MemoPolynomialPosterior
    def model():return BoundsMemoLikelihood(np.zeros((3,1)),np.zeros(1),np.eye(1),young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
    post=MemoPolynomialPosterior({'a':model(),'b':model()},method='legendre',order=128)
    targets={c:np.array([0.,1e-12,0.])for c in post.categories}
    # Force the exact observed faulty scalar-reduction shape at every interior
    # query. Only the CDF reduction is replaced; underlying actual quadrature
    # integrals stay available and untouched for the arithmetic fallback.
    monkeypatch.setattr(legacy,'coordinate_cdf_bounds',lambda *a,**k:dict(cdf=1.0000000000000002,lower=1.0000000000000002,upper=1.))
    for p in (.025,.5,.975):
        row=m.polynomial_coordinate_quantile(post,targets,p)
        assert row['lower_m']<=1e-12*2**(p-1)<=row['upper_m']
        assert row['CDF_bracket']['maximum_CDF_error_bound']<=1e-6


def test_mixture_atom_left_limit_uses_strict_topology_not_probability_subtraction():
    from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood,MemoPolynomialPosterior
    def model():return BoundsMemoLikelihood(np.zeros((3,1)),np.zeros(1),np.eye(1),young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
    for method,order in [('adaptive',128),('legendre',64),('legendre',128)]:
        post=MemoPolynomialPosterior({'a':model(),'b':model()},method=method,order=order)
        targets={'a':np.array([0.,0.,0.]),'b':np.array([1e-12,0.,0.])}
        left=m.polynomial_support_cdf_bounds(post,targets,0.,left_limit=True)
        right=m.polynomial_support_cdf_bounds(post,targets,0.)
        assert left['cdf']==0 and right['cdf']==.5
        assert m.polynomial_coordinate_quantile(post,targets,.025)['kind']=='category_mixture_atom'
        common={c:np.array([1e-12,0.,0.])for c in post.categories}
        assert m.polynomial_coordinate_quantile(post,common,.5)['kind']=='common_constant_atom'
