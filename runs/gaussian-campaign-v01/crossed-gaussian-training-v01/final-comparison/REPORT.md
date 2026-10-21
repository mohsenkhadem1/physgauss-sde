# Fixed six-fit crossed prediction comparison

The supplied-physics linear baseline is substantially more accurate in every role. Spatial exchange shows no consistent benefit over the identical no-exchange network. Both neural arms improve over static rest while remaining far behind the physical baseline.

All 96 engineering captures and six fixed final checkpoints are included. Inputs are supplied physical hypotheses, known rest/hardware and causal commands. No motion observations or images enter. Scores below average 32 cases per role, excluding the rest frame.

| Role | Exchange mean ± seed SD | No exchange mean ± seed SD | Physical linear | Static rest |
|---|---:|---:|---:|---:|
| fit | 0.1882 ± 0.0223 mm | 0.1885 ± 0.0083 mm | 0.0102 mm | 1.1183 mm |
| development_force_transfer | 0.4687 ± 0.0222 mm | 0.4640 ± 0.0121 mm | 0.0092 mm | 1.0515 mm |
| unseen_middle_material | 0.5730 ± 0.0050 mm | 0.5609 ± 0.0057 mm | 0.0233 mm | 1.3342 mm |

Per-BC/material and joint role/BC/material groups, individual seed scores, folding failures and paired ablation differences are retained in summary.json. Seed SD is optimization variation, not calibrated predictive uncertainty. The cohort uses known layouts and three dependent joint material tuples; no sparse-view inference, native/continuum accuracy, production admission or probing-policy claim follows. Raw predictions and checkpoints remain LOCAL.
