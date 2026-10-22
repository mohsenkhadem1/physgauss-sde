# A learned posterior did not match the small physical reference

The fixed neural control failed its declared DEV/CAL gates for all three seeds. TEST remains closed; no neural action selection or TEST performance was measured. The useful result is a measured approximation gap against a numerically qualified reference, rather than another architecture search.

This experiment uses the C38 homogeneous **logarithmic Neo-Hookean** prism, a known uniform normal spring location/law, and ideal scalar endpoint-displacement measurements with 200 µm Gaussian noise. The two unknowns are E and total spring stiffness K. This is a static held-force interpolation experiment, not RGB-D perception, unknown fixture layout, or future-motion forecasting. The physical law differs from the polynomial stable-NH GPU block.

Fresh independent objects were assigned 4096 TRAIN, 256 DEV, 128 CAL and 128 TEST. All eight observation contexts of an object stayed in its role. Inputs contain only acquired displacements, masks, known forces and endpoint identities. A fixed 16-component product-Beta mixture, with two 64-unit hidden layers, predicts the joint density in unit logE/logK coordinates. Each of three seeds trained for exactly 8000 steps; final checkpoints were used without validation selection. There was no architecture or hyperparameter search.

| Role | Seed | Mean joint KL, nat | Maximum joint KL, nat | Maximum field-mean gap, µm |
|---|---:|---:|---:|---:|
| DEV | 2026122410 | 0.5084 | 1.5556 | 635.8 |
| DEV | 2026122411 | 0.5645 | 1.3777 | 482.4 |
| DEV | 2026122412 | 0.4312 | 1.3285 | 470.9 |
| CAL | 2026122410 | 0.5277 | 1.5201 | 899.5 |
| CAL | 2026122411 | 0.5811 | 1.6024 | 846.1 |
| CAL | 2026122412 | 0.5006 | 1.7445 | 573.9 |
| Declared limit | all | 0.10 | 0.50 | 20.0 |

Each role contained 16 continuous-reference cases, exactly two per context. All 32 reference cases passed their numerical checks, including order agreement of the reference-to-network joint KL. Normalized Gauss-Beta rules of orders 12 and 24 agreed on neural field moments to about 1.2e-16 m; inaccurate numerical transport does not explain the observed gaps. Field standard-deviation gaps also exceeded the 20 µm limit. CDF-grid mean errors were 0.13–0.18 rather than the allowed 0.05.

Two seeds passed the conservative marginal CAL coverage/PIT checks, yet failed the joint and predictive checks. Marginal intervals alone therefore did not establish an adequate posterior. Each context has 128 independent CAL objects; its eight contexts are correlated and are not treated as 1024 independent objects. Ordinary and familywise intervals, widths and PIT diagnostics remain in the complete report.

All fits completed in 136.67 s on physical GPU2; peak tensor allocation was 20.2 MB and process RSS 1.63 GB. GPU2 was released immediately afterward. CPU validation took 246.71 s. These are bounded execution costs, not a controlled speed comparison. No TEST release, choices, posterior states, or result files were created after the failed gate.

The 3D field moments come from pushing the density through the exact nonlinear solution at positive material-volume quadrature points. These posterior moments represent uncertainty about physical parameters. They are distinct from the physical shape covariance of material Gaussians. No claim of calibrated real physics, camera uncertainty, or generally intelligent probing follows from this analytical control.

The prospective [protocol](../runs/operator-dev/c38-beta-control-v01/protocol-v01.json), [compact results](../runs/operator-dev/c38-beta-control-v01/result-summary.json), and [complete DEV/CAL report](../runs/gaussian-campaign-v01/c38-beta-control-v01/validation/summary.json) retain every seed. [The numerical teacher](GAUSSIAN_C38_BAR_REFERENCE.md) remains unchanged. A separate fixed Bayesian importance correction is proposed as numerical inference development, not as a repair to these frozen neural results.

Sources are [the density and normalized quadrature](../experiments/campaign_c38_beta_posterior.py), [object-wise simulation pairs](../experiments/campaign_c38_beta_data.py), [reference/coverage metrics](../experiments/campaign_c38_beta_metrics.py), and [the staged runner](../experiments/run_campaign_c38_beta_control.py). Fifteen focused controls passed before generation. Reproduction requires the hash-bound LOCAL datasets/checkpoints; they were not uploaded. Recorded output directories refuse reuse, and the failed gates intentionally prevent the TEST stages.
