# Bird recovery and mixing audit

The Bird posterior remains unvalidated. The larger comparison stays gated on
strict convergence, independent raw replay and the existing uncertainty checks.

## Recovery

Run39214740152 saved3350 draws/chain before its330-minute limit. Its recovery
worker rejected recomputed settings. The mismatch is reproducible with the
unchanged a14766e worker: both image correlations differ by at most3.00e-15
between CPUs. Restoring the original saved correlations makes the complete
original settings guard pass. Saved-target replay differs by at most3.64e-12.

`pacnerf_recovery.py` restores those original constants exactly. It permits only
a64-epsilon float64 recomputation discrepancy (1.42e-14); every other model,
source, input and kernel field still requires exact equality. Tests reject
meaningful correlation changes and changed sources, inputs and kernels.

`prove_pacnerf_bird_recovery.py` runs a separate unmodified checkout of a14766e.
The free CPU recovery-proof workflow appends32 diagnostic draws, preserves the
3350-draw prefix, and compares one32-draw continuation with two16-draw
continuations, including RNG and acceptance state. This is a recovery check;
it does not score predictions or declare convergence. Free CPU proof
run39391075353 passed. Its recomputation discrepancy was1.15e-14; after exact
restoration, density replay error was7.28e-12 and direct/split paths were
bit-identical through3382 draws/chain. The original3350-draw prefix is exact.

## Mixing

The observed-only audit is in `examples/pacnerf-bird-mixing-audit`. No future
particle positions or withheld images were used. The latest full-worker
diagnostics fail (worstRhat1.856; minimum bulkESS5.72). The separate256-draw
full joint-HMC trial also fails (Rhat1.656; minimum bulkESS6.45).

RGB scale mixes well at3350 draws (Rhat1.0005, bulkESS4898), as does smooth-load
scale (Rhat1.0038, bulkESS962). Mechanics and added diffusion drift slowly.
Stiffness chain means range1.66–2.27; diffusion-hyperparameter means range0.42–0.77.

The empirical covariance in the existing sampling coordinates has eigenvalues
0.0110–91.99, a condition ratio of about8338. The fixed reference learned from
an early150-draw pilot severely underrepresents some wide directions. A small
step chosen for the narrow directions moves slowly through the wide ones.
Tested straight interpolations between representative physical trajectories
show no uphill excess over endpoint interpolation. This supports a conditioning
diagnosis; it does not exclude other modes or establish global exploration.

The next development test learns a fixed reference from all3350 discarded
draws, with covariance shrinkage0.1, then starts a new elliptical-slice run.
The physical posterior, Gaussian force prior, guide and feasibility likelihood
stay fixed. The original3350 draws are proposal-training data only. The final
production thresholds remain Rhat<1.01, bulk/tailESS>=400 for every monitored
quantity and at least2000 retained draws/chain. Predictive coverage and proper
scores are evaluated separately after convergence.
