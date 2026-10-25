"""Prospective object-level calibration diagnostics; no data selection/tuning.

Rows are independent objects, never correlated probe resets. Missing entries
remain in coverage denominators and block distribution/comparison summaries.
Raw PIT arrays and object-level scores are not publication artifacts.
"""
import numpy as np
from scipy.stats import beta, kstest, norm


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite p-values in [0,1]')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def binomial_intervals(successes, total, *, alpha=.05, family_size=6):
    if (type(successes) is not int or type(total) is not int or total < 1 or
            not 0 <= successes <= total or not 0 < alpha < 1 or
            type(family_size) is not int or family_size < 1):
        raise ValueError('Valid integer binomial counts and family size')
    p = successes / total
    z = float(norm.ppf(1-alpha/2))
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    tail = alpha/(2*family_size)
    exact = [0. if successes == 0 else float(beta.ppf(tail, successes, total-successes+1)),
             1. if successes == total else float(beta.ppf(1-tail, successes+1, total-successes))]
    return dict(pointwise_Wilson=[float(max(0., center-half)), float(min(1., center+half))],
                simultaneous_Bonferroni_Clopper_Pearson=exact,
                pointwise_alpha=alpha, simultaneous_family_size=family_size)


def primary_calibration(covered, pit, *, labels, alpha=.05):
    """Six predeclared parameter/phase diagnostics, using reset zero only.

    NaN means unresolved. Do not use this binomial calculation for averages
    over multiple probe resets; those require object-cluster resampling.
    """
    c = np.asarray(covered, float)
    p = np.asarray(pit, float)
    names = tuple(labels)
    if (c.ndim != 2 or c.shape[1] != 6 or c.shape != p.shape or not len(c) or
            len(names) != 6 or len(set(names)) != 6 or not 0 < alpha < 1 or
            np.isinf(c).any() or np.isinf(p).any() or
            np.any(np.isfinite(c) & (c != 0) & (c != 1)) or
            np.any(np.isfinite(p) & ((p < 0) | (p > 1)))):
        raise ValueError('Object-by-six Bernoulli coverage/PIT with NaN unresolved')
    rows = []
    pvalues = []
    for j, name in enumerate(names):
        valid = np.isfinite(c[:, j])
        successes = int(np.nansum(c[:, j]))
        missing = int((~valid).sum())
        missing_pit = int(np.isnan(p[:, j]).sum())
        row = dict(label=name, object_count=len(c), covered=successes,
                   unresolved_coverage=missing, missing_PIT=missing_pit,
                   coverage_bounds=[successes/len(c), (successes+missing)/len(c)],
                   nominal_coverage=.9, intervals=None, PIT_diagnostic=None)
        if not missing:
            row['coverage'] = successes/len(c)
            row['intervals'] = binomial_intervals(successes, len(c), alpha=alpha,
                                                  family_size=6)
        if not missing_pit:
            value = kstest(p[:, j], 'uniform', method='exact')
            pvalues.append(float(value.pvalue))
            row['PIT_diagnostic'] = dict(KS_statistic=float(value.statistic),
                KS_unadjusted_pvalue=float(value.pvalue),
                histogram_10_equal_bins=np.histogram(p[:, j], bins=np.linspace(0, 1, 11))[0].tolist(),
                simultaneous_DKW_half_width=float(np.sqrt(np.log(12/alpha)/(2*len(c)))))
        rows.append(row)
    complete = bool(np.isfinite(c).all() and np.isfinite(p).all())
    if complete:
        for row, adjusted in zip(rows, holm_adjust(pvalues)):
            row['PIT_diagnostic']['KS_Holm_adjusted_pvalue'] = float(adjusted)
    return dict(complete=complete, entries=rows,
                object_clusters=len(c), primary_family_size=6,
                comparison_conclusions_blocked=not complete,
                interpretation='Exploratory diagnostics; no coverage pass rule, tuning or calibration proof from non-rejection')


def paired_cluster_intervals(metrics, *, labels, seed=2026102999, resamples=10000):
    """Column means and percentile intervals using shared object resamples.

    Form signed paired differences BEFORE calling. For secondary post-reset
    analyses, first average the three resets within each object. All columns
    share precisely the same sampled row indices.
    """
    values = np.asarray(metrics, float)
    names = tuple(labels)
    if (values.ndim != 2 or not len(values) or values.shape[1] != len(names) or
            not names or len(set(names)) != len(names) or not np.isfinite(values).all() or
            type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 2):
        raise ValueError('Complete aligned object metrics; never drop missing objects')
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, len(names)))
    for start in range(0, resamples, 256):
        indices = rng.integers(0, len(values), size=(min(256, resamples-start), len(values)))
        means[start:start+len(indices)] = values[indices].mean(axis=1)
    intervals = np.quantile(means, [.025, .975], axis=0, method='linear')
    return dict(object_clusters=len(values), resamples=resamples, seed=seed,
                same_object_resamples_across_all_metrics=True,
                entries=[dict(label=name, mean=float(values[:, j].mean()),
                              percentile_95_interval=intervals[:, j].tolist())
                         for j, name in enumerate(names)],
                interpretation='Exploratory paired object-cluster percentile intervals; no multiplicity-adjusted superiority claim')
