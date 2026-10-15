# Full nonlinear likelihood audit

## Why this is necessary

The adaptive-noise posterior is exact for a common local linearized image model, not the full nonlinear simulator and renderer. A small average pixel disagreement does not guarantee a small likelihood error. There are over ten thousand scalar observed RGB values in the default experiment.

Let `r` be the whitened linear-model image residual and `delta` the whitened difference between nonlinear and linear predictions. Then:

```
log L_nonlinear - log L_linear = dot(r, delta) - 0.5 * norm(delta)^2
```

Even small per-pixel deltas can accumulate into a substantial log likelihood change. Material nonlinearity, perspective projection, and Gaussian depth sorting can all affect the approximation.

## Importance-overlap diagnostic

The proposal is the posterior under the linearized likelihood. It uses the same scene prior, noise-increment prior, scale prior, and image-noise variance as the nonlinear target. For either a fixed-scale control or the adaptive mixture, the unnormalized target/proposal ratio is therefore:

```
w(z, alpha) proportional to L_nonlinear(observed_RGB | z, alpha)
                            / L_linear(observed_RGB | z, alpha)
```

Draw independent proposals, run the full nonlinear simulator, and evaluate this ratio using observed RGB only. Normalize weights and record:

```
ESS = 1 / sum(normalized_weights^2)
relative_ESS = ESS / proposal_count
```

We also record the largest normalized weight, weight entropy, log-ratio range, and invalid physical proposals. A relative ESS below 0.1 is a predeclared warning. This threshold is diagnostic, not a formal acceptance test. High finite-sample ESS cannot detect a mode never visited by the proposal.

These diagnostics do **not** establish a corrected posterior or calibrated intervals. We deliberately do not promote fragile reweighted intervals into improved benchmark results. The next inference step should use an iterative nonlinear smoother or a properly checked nonlinear posterior sampler.

## Physical failures and data isolation

An inverted physical proposal has zero nonlinear target density. It keeps zero importance weight and is counted; it is not quietly replaced. A programming or camera-model error stops the audit. No withheld image, material-point target, or benchmark score is used in the ratios.

New study checkpoints save the actual floating-point observed images, their indices/noise level, and the diffusion-network weights. Historical milestone-2 checkpoints lacked those images; they can only be regenerated in the matching numerical environment. The audit labels that legacy path explicitly. It additionally checks the stored reference render and selected stochastic Jacobian columns against current forward calculations.

## Source provenance

`--source-ref` must identify a local committed implementation whose full source fingerprint matches the original study. The audit checks that its core model, simulation, rendering, diffusion and inference modules remain byte-equivalent after newline normalization. This permits adding audit code without silently evaluating an old posterior under altered physics.

The audit has its own source fingerprint and an atomic manifest/record layout. Changed source, changed original measurements, or changed sampling settings cannot silently resume into the same output.

## Commands

For the historical milestone-2 local study:

```sh
physgauss audit --study runs/scale-study --source-ref d225760 \
  --output runs/likelihood-audit --samples 256
```

For a newly generated study from a clean committed checkout:

```sh
physgauss study --quick --output runs/new-scale-smoke
physgauss audit --study runs/new-scale-smoke --source-ref HEAD \
  --output runs/new-audit --samples 32
```

Windows PowerShell users should put each command on one line. Use `--resume` to continue a matching audit. Generated `.pt` checkpoints stay in ignored run directories; compact JSON/CSV/HTML/plot evidence is versioned in `examples/`.
