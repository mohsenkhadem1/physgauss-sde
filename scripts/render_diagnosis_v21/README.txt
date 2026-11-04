Rendering diagnosis v21

Eight PhysTwin sequences, seven interventions, four frames, three cameras:
672 scored renders on GPUs 0-2. Future-frame summaries average cameras and
frames within each sequence, then weight sequences equally.

The same initial Gaussian appearance and matched anchors give:
                           PSNR dB   LPIPS    silhouette IoU
Our predicted motion        16.102   0.3263   0.5579
PhysTwin predicted motion   17.652   0.2654   0.6862
Recorded-motion proxy       20.424   0.2178   0.8099
Proxy + affine footprints   19.815   0.2486   0.7701

Replacing our predicted motion with recorded tracks improves PSNR in all eight
sequences (+4.322 dB on average), reduces RGB MSE 61.56%, and reduces LPIPS
33.27%. No appearance parameters were refit. This supports prioritizing motion
and shape prediction over texture capacity. It does not establish a causal
percentage of error attributable to physics. Tested affine/covariance transport
does not help; it can produce visible stretching artifacts.

Limits: reconstructed tracks are incomplete/noisy, not dense truth or a perfect
upper bound. Dinosaur has only 422/4954 valid anchors at the last frame; only
21.7% of Gaussians have an anchor within 10 mm. Local affine clipping is frequent.
Recorded visibility is used for paired diagnostic anchor selection, including
future frames; this is not a deployable forecasting result. Cameras are the
three appearance-training cameras, not held-out novel views. Initial appearance
PSNR is 31.74 dB, but motion, occlusion and mapping errors remain confounded in
the future-frame residual. No old benchmark scores or model weights changed.

Mask audit: object_motions_valid[t] describes transition t -> t+1, whereas the
legacy FEM adapter applied it to positions at t. v21 uses endpoint visibility
and incoming transition validity [t-1], with visibility alone at frame zero.
Legacy fitting masks differ on 4.2-7.5% of entries; the final training mask can
depend on the next frame. Old fits need correction and separate rebenchmarking
before strict causal preprocessing claims. Predictor-array isolation checks do
not establish upstream preprocessing isolation.

Reproduce: runs/deform360-reconstruction-env/bin/python -m
scripts.render_diagnosis_v21.campaign, then the same Python -m
scripts.render_diagnosis_v21.report. The environment name is historical; this
experiment uses PhysTwin only. Unit tests: four passed (tests.txt).
Protocol, source hashes, aggregate/per-case metrics and mask audit are in
reports/render-diagnosis-v21. Raw assets, forecasts and per-render outputs stay
local under runs/. Visual comparison PNGs remain local; metric SVG is published.

Next: correct legacy fitting masks and rerun matched fits, then target forecast
shape/contact error before adding appearance capacity. Preserve prior results
as historical evidence; report corrected comparisons separately.
