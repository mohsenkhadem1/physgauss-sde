# Handover

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

Repository created and first mechanics milestone verified on CPU. Seven tests pass: reference mass/volume, zero rest stress, force-energy consistency, rotation objectivity, internal force/torque balance, stochastic impulse balance, time-step refinement, and differentiable rendering (some share one test). Synthetic benchmark and observation-only inference are implemented; the end-to-end demo and results are being completed. No reconstruction-performance claims yet.

## Decisions

- Start with fixed-topology deformable solids.
- Use genuine volumetric mechanics rather than independent splat motion.
- Separate Gaussian shape covariance, scene-parameter uncertainty, observation noise, and process noise.
- Use all available observations for offline smoothing; never access withheld images during fitting.
- Compare a deterministic physical model with parameter uncertainty against the same model with a correlated neural diffusion field.
- Label local Gaussian/Laplace approximations explicitly; they cannot establish multimodal posterior correctness.
- Do not redistribute the user's paper or credentials.

## Next milestones

1. Implement and verify the physical renderer, synthetic benchmark, inference, and metrics.
2. Benchmark seed and gap sweeps; assess coverage and interval width together.
3. Add inferred geometry and camera uncertainty.
4. Add contact events, stronger multimodal smoothing, and real videos.
5. Compare against PIDG, GP-4DGS, ODE-GS, and uncertainty-aware reconstruction methods under matched observations.

