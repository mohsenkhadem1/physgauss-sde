# Hybrid material and boundary Gaussians: implemented prediction pilot

This implements the hybrid representation proposed in the October discussion. It is currently a **known-physical-hypothesis forward experiment**, using independent FEM targets. Sparse images do not yet infer those hypotheses; parameter uncertainty, forward-model discrepancy and active acquisition are unfinished. GitHub Actions stay disabled and raw targets, checkpoints and videos stay local.

## Representation and learned interaction

`campaign_material_gaussians.py` builds one volumetric material kernel per tetrahedron. Its reference center is the centroid, its volume is |det(Dm)|/6, its conserved mass is density times volume, and its reference covariance is the exact uniform-tetrahedron second moment, sum(vertex−centroid)(vertex−centroid)^T/20. These kernels occupy the interior as well as the surface.

Both learned predictors return a shared nodal deformation field. Material centers and deformation gradients come from that same piecewise-affine field: F=Ds*Dm⁻¹ and Σ=FΣ₀Fᵀ. There is no independently predicted F or covariance that can disagree with the motion. The normalized Gaussian mixture integrates to represented body mass. Shape covariance measures physical extent, **not uncertainty about material or supports**. Invalid/inverted predicted tetrahedra are reported rather than repaired.

`campaign_boundary_gaussians.py` defines separate material-coordinate support and load kernels using surface-area quadrature. Each active patch has nonnegative weights summing to one; inactive patches have zero weights. Known hardware/actuator information and candidate material/support hypotheses have distinct roles. Finite compliant K/D laws have explicit energy and dissipation. Exact hard constraints remain exact masks, and rigid attachment patches retain their finite poses. Spreading a force with Gaussian weights can change its moment, so Gaussian load features never silently replace the reference's handle-origin force law. Contact is unsupported until a unilateral law is supplied.

`campaign_hybrid_predictor.py` conditions a Gaussian-weighted spatial network on reference geometry, complete two-handle hardware, material/support hypotheses and causal actuator history. Inputs include each ordered attachment patch, handle mass, full inertia, COM offset and initial orientation. Neither recipe IDs, true solver rates nor measured future motion are model features. The first version uses a causal GRU for force memory. Its zero-drive reference is exact. A finite SO(3) decoder preserves both rigid patches; prescribed fixture positions are enforced directly.

`campaign_hybrid_response.py` is a subsequent exploratory response model. Static geometry/hypotheses determine spatial coefficients; fixed causal force-history filters supply their amplitudes. Latent displacement, handle translation and rotation vectors obey force sign/superposition. The finite rotation decoder remains nonlinear. This is a useful small-response inductive bias, not an exact nonlinear elasticity law or a conservative time integrator. Oscillatory temporal filters use declared quadrature approximations.

Moving physical Gaussians have prior art in [PhysGaussian](https://arxiv.org/abs/2311.12198), which uses MPM. The present implementation uses a shared nodal field rather than implementing MPM. Particle message passing also has precedent in [Graph Network-based Simulators](https://arxiv.org/abs/2002.09405). This pilot makes no novelty claim for Gaussian transport or message passing.

## What the comparison measures

The local dataset has 16 three-second captures: four already-used TRAIN hardware/layouts crossed with four force families, with 61 states at 20 Hz. Fifteen pairs use the highest available independently verified original capture; the point-pinned twist uses the independently verified fresh optimizer-only trajectory. The incomplete original temporal qualification remains incomplete. These are engineering captures, not a newly admitted production or inference dataset.

Twelve pairs fit each model. Four fixed pairs are withheld from fitting: point-pin/pull, fixed-patch/shear, two-patch/twist, moving-patch/push. Every hardware/layout is seen in fitting. Consequently this measures developmental **cross-action transfer on known layouts**, not generalization to new objects or material distributions. Existing action/BC and hardware-selection correlations also prevent treating this as a factorized inverse-inference benchmark.

The predictor receives the true physical hypothesis and known rest state, plus planned controls. It sees **zero motion images or future measured positions**. Its entire motion is an open-loop forecast from rest. The "after 1 s" metric uses later targets only; it is not a posterior updated by an observed one-second prefix.

Both first-version neural arms use 76,111 parameters, identical inputs, fixed seed, 1,500 optimization steps and the final checkpoint. One exchanges spatial messages; the other disables exchange. No withheld score selects a checkpoint. Training-only response scales balance the fitting loss and never enter the predictor. Later V02 design follows inspection of V01 diagnostic scores and is explicitly development, not fresh confirmation.

A stronger cheap reference fits an eight-mode linear AR2/DMDc state model separately for each complete public hardware fingerprint, using fitting targets only. It rolls out from rest under the same controls, rejects unknown hardware and reports force-excitation limits. It cannot establish new-layout generalization either. All comparisons include static-rest error and deformation validity, with complete per-pair results retained.

## Reproduction

The source builder `prepare_campaign_hybrid_pilot.py` produces the local dataset and a compact manifest. `train_campaign_hybrid.py` runs either first-version arm; `train_campaign_hybrid_response.py` runs the response version. Fixed configurations are in `configs/gaussian_hybrid_prediction_pilot_v01.json` and `..._v02.json`. All model sources are frozen into each local run before fitting, with data/source hashes and the final checkpoint recorded.

`render_campaign_hybrid_video.py` makes matched actual-versus-predicted orbit views, including every withheld pair. Its labels identify independent FEM targets, supplied physics, prediction from rest and zero image observations. Physical kernel ellipsoids use the transported covariance. The local first-version clip is `runs/gaussian-campaign-v01/hybrid-pilot-v01/exchange-v01/actual-vs-predicted.mp4`; it preserves failure cases rather than presenting only fitting success.

The next inference stage must learn a distribution over material/support hypotheses from acquired views, use the same normalized conditional prior as reference inference, and include calibrated forward discrepancy. A narrow parameter distribution or a small Gaussian rendering footprint is not evidence of calibrated predictive uncertainty. Active probing remains gated on those checks.

## Recorded pilot results

Coordinate RMSE, averaged equally over pairs; units are millimetres. Four reused TRAIN layouts only.

| Predictor | Fit pairs | Held cross-action pairs | Held targets after 1 s |
|---|---:|---:|---:|
| gru_spatial | 0.403 | 3.573 | 4.350 |
| gru_no_exchange | 0.685 | 3.345 | 4.059 |
| force_linear_spatial | 0.839 | 2.761 | 3.353 |
| force_linear_no_exchange | 0.909 | 2.699 | 3.269 |
| linear_state_baseline | 0.142 | 1.000 | 1.191 |
| Static rest | — | 4.051 | — |

The structured response model improves the neural mean over the initial GRU, but spatial exchange does not beat its matched no-exchange ablation in either version. The simple linear state model is substantially better than all four neural arms. All reported predictions remain finite with positive tetrahedron determinants over this window; this does not certify physical balance or long-horizon stability.

These are exploratory results from one seed and four held pairs, with material/support truth supplied. There is no demonstrated Gaussian-network advantage, calibrated posterior or active-policy improvement. In particular, fixed-patch shear and two-patch couple query directions are poorly excited by their fitting controls. More expressive neural structure alone cannot establish a response that the training experiments do not identify. The appropriate next step is a better-excited, factorized training distribution and discrepancy-aware inference, rather than reporting narrow uncertainty from this pilot.
