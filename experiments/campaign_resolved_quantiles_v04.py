"""Exact support tails and directed normalized-mass arithmetic; v01/v02/v03 preserved.

A positional tolerance alone can collapse distinct quantiles when an entire
target distribution is narrower than that tolerance. Require BOTH a relative
position bracket and a bounded CDF error at the returned point. This refines the
calculated quantile; it never widens an interval or changes the probability law.
Atoms must be handled explicitly by a caller before this continuous routine.
"""
import math


def bounded_continuous_quantile(cdf, lower, upper, probability, *,
                                absolute_tolerance=1e-8, relative_tolerance=1e-8,
                                probability_tolerance=1e-6, max_iterations=80):
    if (not all(math.isfinite(v) for v in (lower, upper, probability, absolute_tolerance,
                                         relative_tolerance, probability_tolerance)) or
            not lower < upper or not 0 < probability < 1 or
            absolute_tolerance <= 0 or not 0 < relative_tolerance < 1 or
            not 0 < probability_tolerance < min(probability, 1-probability) or
            type(max_iterations) is not int or max_iterations <= 0):
        raise ValueError('Explicit finite bracket and positive position/probability controls required')
    span = upper - lower
    tolerance = min(absolute_tolerance, relative_tolerance * span)
    if not math.isfinite(span) or tolerance <= 0:
        raise FloatingPointError('Quantile bracket or requested relative scale is not representable')
    calls = 0
    def query(x):
        nonlocal calls
        calls += 1
        row = cdf(x)
        if (not all(math.isfinite(row[k]) for k in ('cdf', 'lower', 'upper')) or
                not 0 <= row['lower'] <= row['upper'] <= 1):
            raise FloatingPointError('Finite ordered numerical CDF enclosure required')
        # Existing CDF error bounds have already been intersected with [0,1].
        # Their central quadrature estimate can lie just outside that range.
        # Project only that estimate; NEVER change the supplied bounds, which
        # alone decide every root sign and its returned probability guarantee.
        estimate = row['cdf']
        if not row['lower'] <= estimate <= row['upper']:
            if estimate < 0 and row['lower'] == 0:
                estimate = 0.
            elif estimate > 1 and row['upper'] == 1:
                estimate = 1.
            else:
                raise FloatingPointError('Interior CDF estimate contradicts its numerical enclosure')
        return dict(row, cdf=estimate, unprojected_CDF_estimate=row['cdf'],
                    CDF_domain_estimate_projection=estimate-row['cdf'])
    L, H = query(lower), query(upper)
    if L['upper'] >= probability or H['lower'] < probability:
        raise FloatingPointError('Initial numerical CDF bounds do not bracket quantile')
    for iteration in range(max_iterations):
        middle = lower + (upper - lower) / 2
        if not lower < middle < upper:
            raise FloatingPointError('Binary64 bracket exhausted before both quantile controls passed')
        M = query(middle)
        cdf_qualified = (M['lower'] >= probability - probability_tolerance and
                         M['upper'] <= probability + probability_tolerance)
        if upper - lower <= tolerance and cdf_qualified:
            return dict(value=middle, lower=lower, upper=upper,
                        cdf_at_value=M, cdf_at_lower=L, cdf_at_upper=H,
                        iterations=iteration, CDF_calls=calls,
                        position_tolerance=tolerance, initial_bracket_span=span,
                        absolute_position_tolerance=absolute_tolerance,
                        relative_position_tolerance=relative_tolerance,
                        probability_tolerance=probability_tolerance,
                        maximum_CDF_error_bound=max(abs(M['lower']-probability), abs(M['upper']-probability)),
                        kind='continuous_relative_position_and_probability_qualified_quantile')
        if M['upper'] < probability:
            lower, L = middle, M
        elif M['lower'] >= probability:
            upper, H = middle, M
        else:
            # Fixed quarter queries preserve only sign-qualified endpoints.
            # No midpoint sign substitution or quadrature/tolerance escalation.
            q1 = lower + (middle-lower)/2
            q3 = middle + (upper-middle)/2
            A, B = query(q1), query(q3)
            changed = False
            if A['upper'] < probability:
                lower, L, changed = q1, A, True
            if B['lower'] >= probability:
                upper, H, changed = q3, B, True
            if not changed:
                raise FloatingPointError('Quantile unresolved under fixed numerical CDF uncertainty')
    raise FloatingPointError('Frozen quantile iteration budget exhausted')


def polynomial_support_cdf_bounds(post, targets, value, *, left_limit=False, support=None):
    """Exact CDF outside the rational global polynomial image; interior unchanged.

    A value strictly below/above every possible target has probability zero/one,
    independently of likelihood quadrature and category normalization arithmetic.
    Equality stays with existing atom and endpoint semantics. This performs no
    projection, interval widening, extra integration or tolerance change.
    """
    from experiments.campaign_c38_postprobe_memo import _global_target_bounds, coordinate_cdf_bounds
    from experiments.campaign_c38_polynomial_likelihood_v02 import rational
    if not math.isfinite(value):
        raise ValueError('Finite polynomial CDF query required')
    lo, hi = _global_target_bounds(targets, post) if support is None else support
    exact = rational(value)
    if exact < lo:
        return dict(cdf=0., lower=0., upper=0., exact_polynomial_support_tail=True)
    if exact > hi:
        return dict(cdf=1., lower=1., upper=1., exact_polynomial_support_tail=True)
    row = coordinate_cdf_bounds(post, targets, value, left_limit=left_limit)
    if (left_limit or any(c[1] == 0 and c[2] == 0 for c in targets.values()) or
            (row['cdf'] > 1 and row['lower'] > row['upper'])):
        return arithmetic_coordinate_cdf_bounds(post, targets, value, left_limit=left_limit)
    return row


def arithmetic_coordinate_cdf_bounds(post, targets, value, *, left_limit=False):
    """Recompute faulty reduction with exact rational scalar arithmetic only.

    The very same likelihood integrator, nodes, partitions, target root boxes,
    prior and posterior law are retained. Invalid ratios remain failures. Each
    atom is implemented by strict/non-strict event topology, never by subtracting
    rounded category probabilities from a rounded aggregate CDF.
    """
    from experiments.campaign_c38_exact_posterior_memo import _topology
    from experiments.campaign_c38_polynomial_likelihood_v02 import rational
    from experiments.campaign_cdf_arithmetic import exact_positive_mass_ratio
    full=[];full_errors=[];active=[];active_errors=[]
    for category, model in post.models.items():
        value_full, error_full = model.integrate_x(model.lo, model.hi, reference_D=post.reference_D, **post.numerics)
        full.append(float(value_full));full_errors.append(float(error_full))
        coefficients=tuple(rational(v)for v in targets[category])
        events,pieces=_topology(coefficients,rational(value),model.lo,model.hi,model.root_digits)
        # Nonconstant polynomial equality has zero mass under the continuous
        # material prior. A constant category at the threshold is an atom:
        # exclude its entire domain for the strict left-limit event.
        if left_limit and coefficients[1] == 0 and coefficients[2] == 0 and coefficients[0] == rational(value):
            pieces=()
        integrals=[];errors=[]
        for a,b in pieces:
            v,e=model.integrate_x(a,b,reference_D=post.reference_D,**post.numerics)
            integrals.append(float(v));errors.append(float(e))
        active.append(integrals);active_errors.append(errors)
    # Original root-box mass bound is retained and scaled conservatively if
    # the exact scalar normalizer lower estimate is below the stored one.
    raw=post.coordinate_cdf(targets,float(value),maximum_root_mass=1e-8)
    return exact_positive_mass_ratio(list(post.prior),full,active,
        full_errors=full_errors,active_errors=active_errors,
        root_mass=raw['uncertain_target_root_mass_upper'],
        old_normalizer_lower=float(post.Z-post.Z_error_estimate),
        minimum_error_components=[raw[k]for k in ('numerical_error_estimate','normalizer_error_estimate','uncertain_target_root_mass_upper')])


def polynomial_coordinate_quantile(post, targets, probability):
    """Opt-in refinement for the existing exact polynomial pushforward.

    Existing constant/mixture-atom semantics and numerical CDF error bounds are
    reused. C38/C39 production modules do not import this prospective function.
    """
    import numpy as np
    from experiments.campaign_c38_postprobe_memo import _global_target_bounds, coordinate_cdf_bounds
    lo, hi = _global_target_bounds(targets, post)
    cdf = lambda value, **kw: polynomial_support_cdf_bounds(post, targets, value, support=(lo, hi), **kw)
    if lo == hi:
        x = float(lo)
        return dict(value_m=x, lower_m=x, upper_m=x, kind='common_constant_atom',
                    CDF_left=dict(cdf=0., lower=0., upper=0.), CDF_at=dict(cdf=1., lower=1., upper=1.))
    atoms = sorted({float(c[0]) for c in targets.values() if c[1] == 0 and c[2] == 0})
    for value in atoms:
        left = cdf(value, left_limit=True)
        right = cdf(value)
        if left['upper'] < probability <= right['lower']:
            return dict(value_m=value, lower_m=value, upper_m=value, kind='category_mixture_atom',
                        CDF_left=left, CDF_at=right)
    lower, upper = np.nextafter(float(lo), -np.inf), np.nextafter(float(hi), np.inf)
    row = bounded_continuous_quantile(cdf,
                                     lower, upper, probability)
    value = row['value']
    return dict(value_m=value, lower_m=row['lower'], upper_m=row['upper'], kind=row['kind'],
                CDF_bracket=row, CDF_left=cdf(value, left_limit=True),
                CDF_at=cdf(value))
