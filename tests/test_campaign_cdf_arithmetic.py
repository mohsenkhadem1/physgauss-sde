from fractions import Fraction
import math
import pytest
from experiments.campaign_cdf_arithmetic import exact_positive_mass_ratio,directed_float


def test_exact_observed_public_scalar_reduction_without_discarding_partial_tail():
    full=[3.1722888058930474e-10,.3429933744958182]
    active=[[3.1722888058929414e-10],[.3429933744958182]]
    z=sum(.5*v for v in full)
    assert sum(.5*p[0]/z for p in active)>1
    r=exact_positive_mass_ratio([.5,.5],full,active)
    assert r['cdf']==1. and r['lower']==math.nextafter(1.,0.)and r['upper']==1.
    exact=Fraction(6634456748292010027274812,6634456748292010027275017)
    assert Fraction.from_float(r['lower'])<=exact<=Fraction.from_float(r['upper'])
    assert exact<1


def test_directed_binary_rounding_both_directions_and_exact_values():
    for x in (Fraction(1,3),Fraction(2,3),Fraction(1,2),Fraction(1,10**320),Fraction(1,10**400),Fraction(0),Fraction(1)):
        lo,hi=directed_float(x,-1),directed_float(x,1)
        assert Fraction.from_float(lo)<=x<=Fraction.from_float(hi)
        assert lo==hi or math.nextafter(lo,math.inf)==hi


def test_unresolved_split_quadrature_disagreement_is_failure_not_projection():
    with pytest.raises(FloatingPointError,match='inconsistent'):
        exact_positive_mass_ratio([.5,.5],[.25,.75],[[.25],[math.nextafter(.75,math.inf)]])


def test_full_event_is_exact_one_even_if_individual_divisions_sum_above_one():
    full=[3.1722888058930474e-10,.3429933744958182]
    r=exact_positive_mass_ratio([.5,.5],full,[[x]for x in full])
    assert r['cdf']==r['lower']==r['upper']==1.


def test_positive_mass_invariants_and_tiny_nonzero_category_are_retained():
    full=[1.,1e-200];r=exact_positive_mass_ratio([.5,.5],full,[[1.],[]])
    assert r['lower']<1 and r['upper']==1
    for args in [([],[],[]),([0.],[1.],[[1.]]),([1.],[1.],[[-1.]])]:
        with pytest.raises(ValueError):exact_positive_mass_ratio(*args)


def test_existing_quadrature_normalizer_and_root_uncertainties_are_conservative():
    r=exact_positive_mass_ratio([1.],[2.],[[1.]],full_errors=[.1],active_errors=[[.02]],root_mass=.03,old_normalizer_lower=2.,minimum_error_components=[.02,.06,.03])
    assert r['lower']<=.5-.11 and r['upper']>=.5+.11
    assert r['lower']<=(1.-.02)/(2.+.1)-.03
    assert r['upper']>=(1.+.02)/(2.-.1)+.03
