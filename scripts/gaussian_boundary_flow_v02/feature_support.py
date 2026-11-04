"""Empirical simulator-bank support screening in retained AND discarded PCA space.

This is not a calibrated OOD probability or proof of posterior validity. Reference
features are bank worlds plus a declared absolute-error/shared-frame-zero noise
model. A finite empirical quantile is only a transparent screening threshold.
"""
import torch
from scripts.gaussian_boundary_flow_v02.checkpoint_contract import relative_observation_noise


@torch.no_grad()
def feature_support_diagnostic(clean_features, observed_features, mean, basis, scale, *,
                               frames, points, noise_sigma, coordinate_mask=None,
                               reference_draws=4096, quantile=.99, seed=72461):
    """Compare one observed prefix with a seeded noisy empirical bank distribution.

    All features must use the SAME relative-coordinate mask and persisted PCA.
    Reference draws sample bank rows with replacement and add relative noise
    before projection. The discarded-space threshold includes bank PCA truncation
    error as well as observation noise. No future observations are used.
    """
    if reference_draws < 32 or not 0 < quantile < 1:
        raise ValueError('At least 32 reference draws and 0 < quantile < 1 required')
    # CPU float64 makes this cheap screening reproducible across GPU workflows.
    clean, real, center, vectors, std = [torch.as_tensor(x).detach().cpu().double()
                                       for x in (clean_features, observed_features, mean, basis, scale)]
    if clean.ndim != 2 or clean.shape[0] < 2 or real.shape != (clean.shape[1],):
        raise ValueError('Expected bank [N,D] and a single observation [D]')
    if center.shape != real.shape or vectors.ndim != 2 or vectors.shape[0] != real.numel() or std.shape != (vectors.shape[1],):
        raise ValueError('PCA shape mismatch')
    if not all(torch.isfinite(x).all() for x in (clean, real, center, vectors, std)) or not (std > 0).all():
        raise ValueError('Finite features/PCA and positive scales required')
    if not torch.allclose(vectors.T @ vectors, torch.eye(vectors.shape[1], dtype=torch.float64), atol=1e-5, rtol=1e-5):
        raise ValueError('PCA basis must be orthonormal')
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(clean), (reference_draws,), generator=generator)
    noise = relative_observation_noise(reference_draws, frames, points, noise_sigma,
                                       mask=coordinate_mask, generator=generator, dtype=torch.float64)
    if noise.shape[1] != clean.shape[1]:
        raise ValueError('Noise layout/mask does not match features')

    def statistics(features):
        centered = features - center
        projected = centered @ vectors
        return ((projected / std).norm(dim=-1),
                (centered - projected @ vectors.T).square().mean(dim=-1).sqrt())

    reference = statistics(clean[indices] + noise)
    real_stats = statistics(real)
    names = ('projected_standardized_norm', 'unrepresented_rms_m')
    scores = {}
    for name, observed, samples in zip(names, real_stats, reference):
        threshold = float(torch.quantile(samples, quantile))
        value = float(observed)
        scores[name] = {'observed': value, 'reference_quantile': threshold,
                        'reference_median': float(samples.median()),
                        'reference_max': float(samples.max()), 'flag': value > threshold}
    return {'scope': 'Empirical bank-plus-relative-noise screening; not a calibrated OOD probability',
            'out_of_bank_warning': any(s['flag'] for s in scores.values()),
            'scores': scores, 'threshold_quantile': quantile, 'reference_draws': reference_draws,
            'seed': seed, 'bank_worlds': len(clean), 'feature_dimension': clean.shape[1],
            'noise_model': {'absolute_sigma_m': noise_sigma, 'frames': frames, 'points': points,
                            'description': 'iid absolute coordinate errors minus shared frame-zero error; mask after subtraction'},
            'limitations': ['Empirical bank coverage and declared noise model only',
                           'Two marginal screening thresholds, not a joint confidence region',
                           'Passing does not establish physical accuracy or posterior calibration']}
