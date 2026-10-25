"""Secondary clean-target Bayes-risk diagnostics; bootstrap independent objects.

Declared after study launch, before any scientific result inspection. This
does not change the original calibration protocol or define a new pass rule.
"""
import numpy as np

MODELS = ('linear', 'nonlinear')
PHASES = ('prefix', 'normal-r0', 'axial-r0')


def summarize(mse, risk, *, seed=2026102999, resamples=10000):
    m, v = np.asarray(mse, float), np.asarray(risk, float)
    if (m.ndim != 3 or m.shape[1:] != (2, 3) or m.shape != v.shape or len(m) < 2
            or np.isinf(m).any() or np.isinf(v).any()
            or np.any(m[np.isfinite(m)] < 0) or np.any(v[np.isfinite(v)] < 0)
            or type(seed) is not int or seed < 0
            or type(resamples) is not int or resamples < 2):
        raise ValueError('Aligned independent-object by two-model by three-phase nonnegative risks')
    common = dict(objects=len(m), resamples=resamples, seed=seed,
                  units='m^2', shared_object_indices=True, Gaussian_assumption=False,
                  observation_noise_added=False, no_pass_rule=True,
                  scope='Secondary after-launch/blinded diagnostic; exploratory percentile intervals, no multiplicity-adjusted superiority claim')
    if not np.isfinite(m).all() or not np.isfinite(v).all():
        return dict(common, complete=False, comparison_blocked=True,
                    unresolved_entries=int(np.isnan(m).sum()+np.isnan(v).sum()),
                    all_object_denominators_retained=True)
    means_m, means_v = m.mean(axis=0), v.mean(axis=0)
    bm = np.empty((resamples, 2, 3)); bv = np.empty_like(bm)
    gains_m = m[:, :, :1] - m[:, :, 1:]
    gains_v = v[:, :, :1] - v[:, :, 1:]
    bgm = np.empty((resamples, 2, 2)); bgv = np.empty_like(bgm)
    generator = np.random.default_rng(seed)
    for start in range(0, resamples, 256):
        indices = generator.integers(0, len(m), size=(min(256, resamples-start), len(m)))
        stop = start+len(indices)
        bm[start:stop] = m[indices].mean(axis=1)
        bv[start:stop] = v[indices].mean(axis=1)
        bgm[start:stop] = gains_m[indices].mean(axis=1)
        bgv[start:stop] = gains_v[indices].mean(axis=1)

    def interval(values):
        return np.quantile(values, [.025, .975], method='linear').tolist()

    cells, gains = [], []
    for model_index, model in enumerate(MODELS):
        for phase_index, phase in enumerate(PHASES):
            mm, vv = means_m[model_index, phase_index], means_v[model_index, phase_index]
            bmm, bvv = bm[:, model_index, phase_index], bv[:, model_index, phase_index]
            undefined = int(np.count_nonzero(bvv == 0))
            # No floor, filtering, or redraw when a bootstrap denominator is zero.
            ratio_CI = None if undefined else interval(bmm/bvv)
            cells.append(dict(model=model, phase=phase, objects=len(m),
                MSE=float(mm), predicted_variance=float(vv),
                MSE_percentile_95_interval=interval(bmm),
                predicted_variance_percentile_95_interval=interval(bvv),
                MSE_minus_variance=float(mm-vv),
                gap_percentile_95_interval=interval(bmm-bvv),
                ratio=None if vv == 0 else float(mm/vv),
                ratio_percentile_95_interval=ratio_CI,
                undefined_bootstrap_ratios=undefined,
                RMS_error_m=float(np.sqrt(mm)), RMS_posterior_SD_m=float(np.sqrt(vv))))
        for action_index, action in enumerate(PHASES[1:]):
            gm, gv = gains_m[:, model_index, action_index], gains_v[:, model_index, action_index]
            gains.append(dict(model=model, action=action, sign='prefix minus post; positive means reduction',
                mean_MSE_reduction=float(gm.mean()), mean_variance_reduction=float(gv.mean()),
                MSE_reduction_percentile_95_interval=interval(bgm[:, model_index, action_index]),
                variance_reduction_percentile_95_interval=interval(bgv[:, model_index, action_index])))
    return dict(common, complete=True, comparison_blocked=False, cells=cells, paired_reductions=gains,
                ratio_intervals_all_defined=all(c['ratio_percentile_95_interval'] is not None for c in cells),
                interpretation='Under a correct prior-predictive Bayes model, expected clean-target MSE equals expected trace posterior variance. Ratio one is a population reference, not per-object equality or a calibration proof. Numerical refinement indicators must accompany these diagnostics.')
