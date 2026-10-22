# Current priority: deformation and numerical mechanics before networks

The [integrated campaign report](../runs/gaussian-campaign-v01/mechanics-priority-v01/REPORT.md) separates three questions: does the population deform measurably, is the stronger nonlinear forward problem resolved, and can a small conditional posterior be checked independently?

All 32 sealed fresh objects were audited across both known programs. After best proper rigid-motion removal, mass-weighted nonrigid coordinate RMS averages **24.75 µm**, versus **1.135 mm** total motion. The mean **objectwise** nonrigid/total RMS ratio is **3.19%**. Compatible strains are small on average but locally nonzero.

At the actual acquired 72 depth pixels per object, mean paired-visible true-versus-rigid contrast is **32.64 µm**, against the stipulated **200 µm** single-observation noise. There are **864 foreground pairs out of 2,304 pixels**, 1,440 shared background pixels and no visibility changes. This is weak elastic contrast at these views, not zero information or a proof of nonidentifiability.

Stronger controlled forces produce visible finite strain, but mesh accuracy remains unresolved. Sparse nonlinear solver parity is **4.16×10⁻¹⁷ m**; the **1,024→8,192-tet difference is 0.38034 mm against a 0.24388 mm tolerance**. The finest 11.52% strain/0.800 mm nonrigid motion also misses the original excitation target. The local mesh video is explicitly **UNCONVERGED**, at true scale with slow playback. No nonlinear fit, network training or probing expansion follows.

Separately, the manufactured **conditional E/K posterior** has independent adaptive integration, eight prior-started MCMC chains (maximum R-hat 1.00220, minimum bulk/tail ESS 2,763/3,095), and six higher-particle SMC runs with all 15 pair checks passing. Fixture center and width are fixed and known: this is not the original four-dimensional marginal posterior. Earlier grid failures, reporting failure and wall cap remain preserved. [Method and limits](GAUSSIAN_COMPLIANT_CONTINUOUS_REFERENCE.md).

Next: diagnose physical central-force/centroid alignment, resolve nonlinear mechanics, then attempt a small nonlinear posterior before networks. There is no calibrated generalized-boundary-learning claim.

The [sealed rotation-aware 32-object report](../runs/gaussian-campaign-v01/point-rotation-analysis-v02/REPORT.md) remains the latest **prediction study**, with its fixed/random-policy and material-coverage limitations unchanged. All raw datasets, particles and media remain local.
