# Research protocol

## Hypothesis

Coupled material dynamics reduce ambiguity across missing frames. A structured stochastic model is useful only if it improves held-out motion or probabilistic scores beyond uncertain deterministic dynamics.

## First experiment

Known elastic rest shape and camera orbit. Fit only observed RGB images. Withhold a contiguous temporal block and render a separate camera path for novel-view evaluation. Never build a pseudo-measurement from a withheld image. Train the diffusion network on a separate synthetic conditional-innovation calibration task, and disclose this supervision.

Use identical observation schedules, physical laws, rendering, and observation-noise settings for the two inference methods. Report parameter/state posterior approximations, sample count, prior assumptions, runtime, and numerical failures. Keep calibration and test seeds separate.

## Metrics

- Held-out PSNR and RGB error.
- 3D position RMSE and sample-based energy score.
- Componentwise CRPS, central interval coverage, and interval width.
- Separate missing-gap and observed-frame statistics.
- Mass conservation, internal net force and torque, minimum deformation determinant, and time-step convergence.
- Never interpret a low PDE residual, good PSNR, or wide confidence interval as proof of physical identification.

## Related work

- PIDG: https://arxiv.org/abs/2511.06299
- PhysGaussian: https://arxiv.org/abs/2311.12198
- PAC-NeRF: https://arxiv.org/abs/2303.05512
- Lagrangian particle optimization: https://arxiv.org/abs/2406.04155
- ODE-GS: https://arxiv.org/abs/2506.05480
- GP-4DGS: https://arxiv.org/abs/2604.02915
- USplat4D: https://arxiv.org/abs/2510.12768
- Structured constitutive energies: https://arxiv.org/abs/2302.02403
- Latent SDE inference: https://proceedings.mlr.press/v108/li20i.html

These are references and planned baselines, not implemented comparisons.
