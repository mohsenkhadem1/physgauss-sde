"""Paired layout-level uncertainty for scientific comparisons, not pixel CIs."""
import numpy as np


def paired_layout_bootstrap(candidate,baseline,groups=None,draws=10000,seed=20261108):
    candidate=np.asarray(candidate,dtype=float);baseline=np.asarray(baseline,dtype=float)
    if candidate.shape!=baseline.shape or candidate.ndim!=1 or not len(candidate):raise ValueError('Matched case-level vectors required')
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():raise ValueError('Finite case metrics required')
    groups=np.arange(len(candidate)) if groups is None else np.asarray(groups)
    if groups.shape!=candidate.shape:raise ValueError('One layout group per case required')
    unique=np.unique(groups)
    # Average matched material replicates inside each independent layout first.
    c=np.array([candidate[groups==group].mean() for group in unique])
    b=np.array([baseline[groups==group].mean() for group in unique])
    rng=np.random.default_rng(seed);ids=rng.integers(len(unique),size=(draws,len(unique)))
    means_c=c[ids].mean(-1);means_b=b[ids].mean(-1)
    differences=means_c-means_b
    result={'independent_layouts':len(unique),'candidate_mean':float(c.mean()),'baseline_mean':float(b.mean()),
            'paired_difference':float((c-b).mean()),
            'paired_difference95_interval':np.quantile(differences,[.025,.975]).tolist(),
            'bootstrap_draws':draws,'seed':seed,
            'qualification':'Paired percentile bootstrap over independent layout groups; exploratory if used during model selection'}
    if (means_b>0).all():
        result.update({'relative_improvement':float(1-c.mean()/b.mean()),
                       'relative_improvement95_interval':np.quantile(1-means_c/means_b,[.025,.975]).tolist()})
    return result
