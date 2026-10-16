# Nonlinear posterior sampling

The local Gaussian mixture is now a proposal reference, rather than the final posterior. Every Metropolis decision evaluates the full nonlinear simulator and the likelihood of the actually observed RGB frames. This addresses the approximation failure found in the importance audit. A correct invariant distribution is not sufficient evidence that a finite run has converged.

## Target and change of coordinates

Let `k` index the declared discrete process-scale grid and `z` contain the eight standardized scene coordinates and the standardized white increments. The target is:

```
p(k,z | observed_RGB) proportional to prior(k) * N(z;0,I) * L_nonlinear(z,k)
```

The common linearized model gives a normalized Gaussian reference `q(z|k)=N(m_k,H_k^-1)` and log component weights `log w_k = logsoftmax(log prior(k) - negative_log_evidence_k)`. Log weights are recomputed without taking the log of possibly underflowed probabilities. With `H_k=L_k L_k^T`, write:

```
z = m_k + L_k^-T u
r(k,u) = log L_nonlinear(z,k) - log L_linear(z,k)
p(k,u | RGB) proportional to w_k * N(u;0,I) * exp(r(k,u))
```

The component-specific Jacobian is included by this Gaussian change of coordinates. This identity requires unchanged priors, image-noise level, observed images, and forward model. The command verifies historical source compatibility using the same safeguards as the likelihood audit.

## Three reversible kernels

Each chain independently chooses one kernel at each transition:

1. Probability 0.70: within-component pCN, `u' = sqrt(1-beta^2)*u + beta*epsilon`, with `epsilon~N(0,I)`. Accept with log ratio `r'-r`.
2. Probability 0.15: independent component/coordinate refresh, `k'~g` and `u'~N(0,I)`.
3. Probability 0.15: component transport, `k'~g`, retaining the same `u` and decoding through the new component's affine map.

For the last two kernels accept with log ratio:

```
r' - r + log(w_k'/g_k') - log(w_k/g_k)
g = 0.95*w + 0.05*prior
```

The defensive component mass permits proposing every declared scale even if its linearized probability underflows. It does not guarantee efficient discovery of a nonlinear mode. Transport proposals can leave the component unchanged; nontrivial scale changes have their own attempt/accept counters.

Each chain adapts only its pCN beta during discarded warmup, in windows of 25 transitions towards acceptance 0.35, within `[0.03,0.98]`. Adaptation freezes for retained transitions. Warmup and retained acceptance/invalid counts are recorded separately. There is no annealing, likelihood tempering, or truth-dependent tuning.

With `--langevin`, the within-component kernel becomes gradient-guided pCNL. Write `a=sqrt(1-beta^2)` and `h(u)=gradient_u r(k,u)`. The proposal is `u'=a*u+(1-a)*h(u)+beta*epsilon`. The acceptance log ratio additionally includes `log phi(u')-log phi(u) + log N(u; a*u'+(1-a)*h(u'), beta^2 I) - log N(u'; a*u+(1-a)*h(u), beta^2 I)`. This reverse-proposal term is essential; the gradient is a proposal aid and does not approximate the target. Gradients use the full nonlinear likelihood and are transformed by `L_k^-1` into whitened coordinates. The independent and component-transport kernels are unchanged. Transition counters retain the name `pcn` for either within-component variant; the `langevin` setting distinguishes them. A finite-difference check tests scene and process-noise gradients through the actual physics and renderer; a separate numerical-quadrature test checks nonlinear scale probabilities.

Inverted-element proposals have zero target probability and remain counted rejections. A nonfinite initial target stops the run rather than silently restarting or selecting the best chain. NaN renderer results stop with an error. Rejected proposals retain the previous state in the chain.

## Diagnostics and evaluation

Four independently initialized chains are the default. Every scene coordinate, every white increment, observed log likelihood, and (for the adaptive model) process scale is monitored. The implementation uses averaged ranks for ties, Blom rank normalization, split chains, folded rank Rhat, FFT autocovariances, and the initial positive/monotone sequence estimator of multichain ESS. Tail ESS is the minimum for the empirical 5% and 95% indicator functions.

The predeclared diagnostic screen requires rank/folded split Rhat below 1.01, bulk ESS at least 400, tail ESS at least 400, and no frozen chain for every monitored quantity. Undefined diagnostics, including constant adaptive-scale traces, cannot pass automatically. These are convergence warnings, not a proof of posterior correctness or calibration. Diagnostics based on finite samples cannot exclude unvisited modes, including a rare scale component.

All retained states enter diagnostics. To limit rendering memory, an equal evenly spaced subset from each chain is used for truth-based predictive scores; this is a cost decision, not a claim of independent samples. The local reference is evaluated with the same rendering sample count. Scores remain provisional when diagnostics fail. ESS counts are Monte Carlo efficiency, not independent scene or realization replicates. This selected-case experiment must not be combined with the 18-case local study as a claim about repeated-realization coverage.

No fitting code receives held-out truth. Only after sampling is complete does the evaluation layer regenerate the benchmark, verify equality with the actual saved observations, and score its withheld positions and views. The prior still assumes known rest geometry, camera, appearance, supports, density, material family and diffusion network.

## Reproduction

From a clean committed checkout, first create a study containing the local reference and saved observations:

```sh
physgauss study --quick --output runs/new-local-reference
physgauss sample --study runs/new-local-reference --source-ref HEAD --output runs/nonlinear-smoke --chains 4 --warmup 20 --draws 32 --evaluation-samples 16
```

That short run tests integration only; it cannot satisfy an ESS requirement of 400. The historical full local study has source ref `d225760` and requires the original numerical environment because its observations were not persisted:

```sh
physgauss sample --study runs/scale-study --source-ref d225760 --case stochastic__long_gap__3101 --output runs/nonlinear-stochastic --warmup 1000 --draws 2000
```

Repeat `--case` to select additional exact labels. Omit it to run every study case. `--method physical` and `--method fixed_sde` condition on those scales instead of sampling scale. `--resume` resumes at completed-case boundaries with unchanged settings, full source fingerprint, observation/reference checkpoint hashes, and diffusion checkpoint hash. It does not resume an interrupted individual chain; that case restarts with its recorded seed. Large chain tensors stay under ignored `runs/`; compact evidence is copied to `examples/`.

## References

- Cotter, Roberts, Stuart and White, [MCMC Methods for Functions](https://arxiv.org/abs/1202.0709): Gaussian-reference pCN proposals.
- Vehtari, Gelman, Simpson, Carpenter and Buerkner, [Rank-normalization, folding, and localization](https://arxiv.org/abs/1903.08008): multichain convergence diagnostics.
- [Stan reference manual: posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html): autocorrelation and ESS definitions.

The particular Gaussian-mixture correction/transport implementation here is derived above. No claim of algorithmic novelty or a published-method reproduction is made.
