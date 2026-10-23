"""Exact arithmetic reduction of already-computed positive quadrature scalars.

This accounts only for scalar weighting, normalization and reduction arithmetic.
It does not certify quadrature/integrand accuracy or alter quadrature controls.
"""
from fractions import Fraction
import math


def directed_float(value, direction):
    value=Fraction(value)
    result=float(value)
    if not math.isfinite(result):raise FloatingPointError('Nonfinite exact ratio conversion')
    represented=Fraction.from_float(result)
    if (direction<0 and represented>value)or(direction>0 and represented<value):
        result=math.nextafter(result,-math.inf if direction<0 else math.inf)
    return result


def exact_positive_mass_ratio(priors, full_integrals, active_integrals, *, full_errors=None, active_errors=None, root_mass=0., old_normalizer_lower=None, minimum_error_components=()):
    """Return outward float enclosure of an exact ratio of supplied float data.

    Integrals are treated as the already-computed numerical quadrature values.
    Their errors must be handled separately; zero estimated quadrature error is
    never promoted to a mathematical integration certificate. A numerator above
    its denominator fails, exposing split-quadrature inconsistency explicitly.
    """
    if not(len(priors)==len(full_integrals)==len(active_integrals))or not priors:
        raise ValueError('Matching nonempty category mass lists required')
    full_errors=[0.]*len(priors) if full_errors is None else full_errors
    active_errors=[[0.]*len(p)for p in active_integrals] if active_errors is None else active_errors
    if len(full_errors)!=len(priors)or len(active_errors)!=len(priors):raise ValueError('Matching error lists required')
    numerator=Fraction(0);denominator=Fraction(0);numerator_error=Fraction(0);denominator_error=Fraction(0)
    for pi,full,pieces,full_error,piece_errors in zip(priors,full_integrals,active_integrals,full_errors,active_errors):
        if not math.isfinite(pi)or pi<=0 or not math.isfinite(full)or full<0:
            raise ValueError('Finite positive prior and nonnegative full integral required')
        pi=Fraction.from_float(float(pi));denominator+=pi*Fraction.from_float(float(full))
        if len(pieces)!=len(piece_errors)or not math.isfinite(full_error)or full_error<0:raise ValueError('Finite nonnegative error estimates required')
        denominator_error+=pi*Fraction.from_float(float(full_error))
        for err in piece_errors:
            if not math.isfinite(err)or err<0:raise ValueError('Finite nonnegative error estimates required')
            numerator_error+=pi*Fraction.from_float(float(err))
        for v in pieces:
            if not math.isfinite(v)or v<0:raise ValueError('Finite nonnegative active integral required')
            numerator+=pi*Fraction.from_float(float(v))
    if denominator<=0 or not 0<=numerator<=denominator:
        raise FloatingPointError('Computed integral ratio inconsistent with positive probability law')
    ratio=numerator/denominator
    if denominator<=denominator_error or not math.isfinite(root_mass)or root_mass<0:raise FloatingPointError('Unresolved normalizer/root mass')
    root=Fraction.from_float(float(root_mass))
    if old_normalizer_lower is not None:
        if not math.isfinite(old_normalizer_lower)or old_normalizer_lower<=0:raise ValueError('Positive original root-mass normalizer lower estimate required')
        root*=max(Fraction(1),Fraction.from_float(float(old_normalizer_lower))/(denominator-denominator_error))
    lower=max(Fraction(0),(numerator-numerator_error)/(denominator+denominator_error)-root)
    upper=min(Fraction(1),(numerator+numerator_error)/(denominator-denominator_error)+root)
    old_radius=Fraction(0)
    for error in minimum_error_components:
        if not math.isfinite(error)or error<0:raise ValueError('Finite nonnegative original errors required')
        old_radius+=Fraction.from_float(float(error))
    lower=max(Fraction(0),min(lower,ratio-old_radius))
    upper=min(Fraction(1),max(upper,ratio+old_radius))
    return dict(cdf=float(ratio),lower=directed_float(lower,-1),upper=directed_float(upper,1),
                exact_arithmetic_reduction=True,arithmetic_scope='Already-computed quadrature scalars only; not an integration certificate')
