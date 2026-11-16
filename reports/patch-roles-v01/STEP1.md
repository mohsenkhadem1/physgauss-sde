# Patch roles v01, step 1: K Gaussian surface patches with learned roles

Research prototype plus a synthetic recovery study. This is not paper text.
Code: `scripts/patch_roles_v01/`. Outputs: `runs/patch-roles-v01/` (per-scene `result-s*.json`, cached SMC particles `*.npz`, `summary.json`, `tests.json`, logs).
The published code in `scripts/joint_field_identification/` is imported and was not modified.

## 1. Model as implemented (`model.py`)

**Body.** The same small-strain block as the joint-field study, with bounding box [-.06,.06]x[-.03,.03]x[0,.06] m. It is quasistatic with Poisson ratio 0.3 and no gravity. `mechanics.LinearGaussianBody` provides the mesh, the element stiffness, the surface quadrature, the published footprints, the interpolation and the material field.
- Material: log E(X) = m0 + m1·chi(X), the published two-kernel, volume-centred field.
- Inference mesh: (8,4,4) cells, i.e. 225 nodes and 768 tets.

**Patches.** There are K exchangeable patches. Each has a role in {force, support, off}. Two families are implemented.
- `published` (regression family). A force patch is exactly the published load: top -z and side -y footprints at a shared centre x, width 18 mm, 2 amplitudes per time. A support patch is the published bottom footprint at x, with stiffness k. The published bounds are kept.
- `general`. The centre lies on the four lateral faces, parametrised as (x, phi), where phi is arc length around the y-z cross-section. Each patch has a log width s in [6, 25] mm. The footprint w_j is a normalised Gaussian of ambient distance on the surface quadrature points; it integrates to 1, has the same form as `footprint()`, and can wrap around edges. A force patch has 3 Cartesian modes (x, y, z), so its amplitude a_j(t) is a 3-vector in N. A support patch is an isotropic bilateral spring with energy ½ k_j ∫ w_j |u|² dA.
- Bounds: log k in [ln 1500, ln 24000]; x in [-.05, .05] m.

**Forward model**, for role counts (nF, nS) and theta = (m0, m1, [log tau], force (x, phi, log s) × nF, support (x, phi, log s, log k) × nS):

    K_theta = K_el(E_theta) + Σ_supports k_j diag(w_j ⊗ I_3)
    K_theta u_t = B_theta a_t + Σ_c c_{c,t} f_c          (f_c: known commands, θ-independent columns)
    y_t = M_t (G_theta a_t + D_theta c_t) + eps,   eps ~ N(0, σ² I),  σ = 0.02 mm

- G and D come from one batched Cholesky solve per particle (GPU, float64). The solve is followed by P1 interpolation and orthographic camera rows: front camera (x,z) on 10 front-face points and top camera (x,y) on 10 top-face points, giving 40 rows per time at 8 times.
- 20 back-camera rows (x,z) on 10 back-face points exist only as candidate "look" actions. M_t is the per-time row mask.

**Amplitude prior.** a = vec_t(a_t) ~ N(0, Λ), with Λ = τ² σ_F² Matérn32(t,t'; ℓ=0.30) ⊗ I_nA, σ_F = 0.08 N, and the time-major ordering of `surface.load_cov`. τ ≡ 1, or in the hierarchical option log τ ~ U[ln .5, ln 2.5] is a continuous parameter. The prior scale is the same for every force patch; per-patch τ_j would be a one-line change.

**Exact marginal**, computed in amplitude space and never forming C. Its dimension is T·nA ≤ 96, against 320 observations:

    P = Λ⁻¹ + Σ_t e_t e_tᵀ ⊗ Gᵀ M_t G / σ²,     b = vec_t(Gᵀ M_t r_t)/σ²,  r_t = y_t − D c_t
    log p(y|θ) = −½[ rᵀr/σ² − bᵀP⁻¹b + log|Λ| + log|P| + n log 2πσ² ]       (= log N(y; d, R + HΛHᵀ))
    μ_θ = P⁻¹ b,   S_θ = P⁻¹                                             (conditional force posterior)

Reset/probe episodes are known loads only: y_p ~ N(0.08 N · g_{θ,p}, σ² I). They update the weights of the physical hypotheses and never the force conditionals. The time-mask form handles looks, i.e. partial rows at one time.

**Role prior (sparsity).** Each patch is iid with P(off)=.5, P(force)=.25, P(support)=.25. Role vectors with nS = 0 are excluded: the stiffness is singular because nothing anchors the rigid modes. Continuous priors are uniform in the bounds above, as in `continuous.run_smc`.

## 2. Inference (`run_scene.py`) and why this route

The patches are exchangeable a priori: they share iid role and continuous priors. The posterior over the 3^K labelled role vectors therefore depends only on the counts (nF, nS). The multinomial gives that mass exactly:

    p(nF,nS | y) ∝ K!/(nF! nS! nOff!) pF^nF pS^nS pOff^nOff · Z_(nF,nS),   Z = ∫ p(y|θ) p(θ) dθ

So instead of 81 role sets at K=4, I enumerate the admissible count configurations: 6 at K=3 and 10 at K=4.
- Within each configuration I run the published adaptive tempered SMC (`continuous.run_smc`, unchanged): 512 particles, 6 RW-MH moves per stage, conditional-ESS 0.8.
- Patches within a role are ordered by x. This removes label switching; the likelihood is -inf when the order is violated, and log nF! + log nS! is added back to the SMC evidence.
- I chose enumeration over role-flip (birth/death) SMC for three reasons. With exchangeability there are only O(K²) configurations. Each configuration is a fixed-dimensional problem the published sampler already handles. The role posterior then comes directly from evidences, with no trans-dimensional mixing to diagnose.
- The cost is that role posteriors inherit SMC evidence noise. Seed replicates (§5) show it is not negligible.

Everything downstream works on the configuration-weighted particle mixture: each particle carries its exact (μ_θ, S_θ).
- (a) Hidden target: displacement of the 10 back-face points (3D, 8 times) under the original unknown loads. This is a Gaussian mixture.
- (b) New known commands C1–C4 at 0.08 N under reset conditions. These are deterministic per particle.
- Threshold risk: r = P[RMSE(q, q̂) > δ | D], with δ = 20% of the predicted response RMS, a prediction-time quantity.
- Design: J(proceed) = r; J(look) = 0.02 + E r'; J(probe) = 0.05 + E r'. The expectation uses 128 common-random-number predictive draws.
- Looks update both the weights and the force conditionals; this is an exact per-particle Gaussian update. Probes reweight the particles only.
- Ties in threshold risk are broken by the squared-error risk of the decision target, C2 (top push at x = +30 mm).

**Exact vs approximate.**
- Exact, given θ and the configuration: the amplitude marginal, μ, S, the hidden-target mixture moments, the look/probe updates, and the reduction of role vectors to counts.
- Approximate:
  - SMC within each configuration, which also makes evidence and role posteriors approximate (§5);
  - the 128-draw preposterior MC;
  - FEM discretisation on the coarse inference mesh;
  - the Gaussian Matérn force prior, which is a modelling choice.

**Identity tests** (`tests.py`, results in `runs/patch-roles-v01/tests.json`, all passing):
- The published-family forward model equals `LinearGaussianBody.solve` to a relative error of 1.3e-14.
- The Woodbury log-likelihood, μ and S equal `continuous.MarginalGaussianLikelihood` (dense C) to within 4e-9 (absolute, log-lik) and 1e-12 (relative, μ and S). This holds for the published 1F1S and 2F1S-hierarchical cases and for the general 2F1S, 1F2S-hierarchical and 0F2S cases, with masked rows, a known command and a probe episode.
- Footprints integrate to 1.

## 3. Truth generation (avoiding the inverse crime)

`truth.py` uses a separate (16,8,8) mesh: 1377 nodes, 6144 tets, 8× the elements of the inference mesh.
- Footprints are re-integrated on the fine surface quadrature.
- log E is shifted for the fine mesh's volume centring, as in `fine_reference.py`.
- Seeded noise of 0.02 mm is added to the camera rows. Probe responses and held-out command responses are computed on the same fine mesh.
- Mismatch options: a top-hat (uniform-disc) traction footprint, pressure-only histories with a short correlation length, and a stiffening support k(u) = k0(1 + |u|²/u0²) solved by damped Picard iteration.

This is still the same P1-tet family. It is a finer numerical reference, not an independent solver. As a control, four scenes were re-run with truth on the inference mesh itself (`*-truth8x4x4`, an intentional inverse crime). This separates discretisation error from the effects of the role model.

## 4. Scenes (`scenes.py`)

True log E0 = ln 16000 and m1 = 0.3 (0.4 in scene i). Force histories are Matérn draws (σ_F · τ) unless stated otherwise.

| scene | truth | K | notes |
|---|---|---|---|
| i_published | published family: 1 force (x=-20 mm, top/side) + 1 bottom support (x=12.5 mm, k=8000) | 3 | regression |
| ii_two_forces | F top x=-30, F front x=+30 z=35, S bottom centre (w 20 mm, k 8000) | 3 | |
| ii_h_scale | as ii with force scale τ=1.8; hierarchical τ in the model | 3 | probe stage |
| iii_a_support_truth | known command 0.3 N top x=+35; S bottom x=-20 (k 8000) + compliant S bottom x=+35 (w 12, k 3000) | 3 | designed probe + fixed push P top -z x=+40 |
| iii_b_force_truth | same command; S bottom x=-20; unknown force bottom x=+35, upward, = 0.45 × command (synchronous "holding hand") | 3 | designed probe + fixed push |
| iv_decoy | F top x=+20, S bottom x=-10 (k 6000) | **4** | sparsity |
| v_a_tophat_short | as iv, but top-hat footprint (r 15 mm), pressure-only, ℓ=0.10 | 3 | mismatch |
| v_b_nonlinear_support | as iv, support stiffening u0=15 µm | 3 | mismatch, probe stage |

## 5. Results

Seed 0, fine-mesh truth unless marked. The location error is the posterior-mean 3D centre distance under the true configuration, with patches matched by x-order; for the published family it is the x distance.
- P(F/S near) is the posterior probability, over all configurations, that some patch of that role lies within 15 mm of each true patch.
- Coverage columns (cov) are 90% equal-tailed intervals.
- Hidden and new-command errors are RMSE in µm, with the predictive sd and coverage alongside.

| scene / stage | P(true cfg) | MAP | P(F near) | P(S near) | E[#spurious] | loc err F / S mm | abs dlogE0 (90% cov) | abs dlogk (cov) | force cov | hidden RMSE/sd µm (cov) | new-cmd RMSE/sd µm (cov) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| i_published | **0.94** | 1F1S | 1.00 | 1.00 | 0.00 | 1 / 1 | 0.14 (y) | 0.13 (y) | 0.94 | 5.5/4.8 (0.73) | 17.8/15.2 (0.78) |
| i_published seed 1 | 0.93 | 1F1S | 1.00 | 1.00 | 0.00 | 1 / 1 | 0.12 (y) | 0.12 (y) | 1.00 | 5.4/4.8 (0.74) | 17.1/16.2 (0.81) |
| i_published, coarse truth | 0.92 | 1F1S | 1.00 | 1.00 | 0.00 | 1 / 2 | 0.22 (y) | 0.14 (y) | 1.00 | 4.6/4.8 (0.88) | 12.9/14.4 (1.00) |
| ii_two_forces | **1.00** | 2F1S | 1.0/1.0 | 1.00 | 0.00 | 2,3 / 5 | 0.30 (**n**) | 0.08 (y) | 0.69 | 14.8/7.2 (**0.46**) | 6.0/6.1 (0.78) |
| ii_two_forces, coarse truth | 1.00 | 2F1S | 1.0/1.0 | 1.00 | 0.00 | 2,3 / 7 | 0.09 (y) | 0.00 (y) | 0.75 | 11.4/7.2 (0.55) | 5.7/5.0 (0.56) |
| ii_h_scale initial | 1.00 | 2F1S | 1.0/1.0 | 1.00 | 0.00 | 1,3 / 4 | 0.24 (y) | 0.57 (n) | 0.98 | 19.9/7.1 (0.34) | 8.4/12.4 (0.82) |
| ii_h_scale after probe | 1.00 | 2F1S | 1.0/1.0 | 1.00 | 0.00 | 1,2 / 4 | **0.00** (y) | 0.86 (n) | 0.81 | 18.5/6.7 (0.35) | **4.3**/2.4 (0.46) |
| iii_a initial | **0.77** | 0F2S | – | 0.11/0.07 | 1.77 | – / 27,27 | 0.18 (y) | 0.52,0.13 (y,y) | – | 7.3/4.0 (0.64) | 12.5/4.9 (0.54) |
| iii_a seed 1 | 0.98 | 0F2S | – | 0.33/0.16 | 1.49 | – / 19,24 | 0.22 (y) | 0.03,0.30 | – | 7.5/4.3 (0.70) | 12.8/5.7 (0.51) |
| iii_a after designed probe (top x=-40) | 0.06 | 0F3S | – | 0.89/0.65 | 0.89 | – / 8,7 | 0.12 (y) | 0.03,0.29 | – | 5.9/5.2 (0.93) | 7.2/4.7 (0.59) |
| iii_a after push top x=+40 | 0.95 | 0F2S | – | 0.51/0.03 | 1.48 | – / 17,23 | 0.16 (n) | 0.47,0.16 | – | 8.3/4.0 (0.39) | 11.5/3.7 (0.35) |
| iii_b initial | **0.004** | **0F1S** | **0.00** | 1.00 | 0.06 | 62 / 6 | 0.14 (y) | 0.13 (y) | 0.79 | 4.9/3.2 (0.60) | 20.8/2.9 (**0.22**) |
| iii_b seed 1 | 0.14 | 0F1S | 0.00 | 1.00 | 0.16 | 62 / 5 | 0.15 (y) | 0.01 (y) | 0.79 | 4.5/5.2 (0.77) | 21.2/2.5 (0.23) |
| iii_b, coarse truth | 0.015 | 0F1S | 0.00 | 0.97 | 0.16 | 58 / 7 | 0.09 | 0.10 | 0.79 | 4.1/3.8 (0.84) | 18.2/3.6 (0.28) |
| iii_b after designed probe (top x=-40) | 0.009 | 0F1S | 0.00 | 1.00 | 0.01 | 65 / 3 | 0.07 | 0.35 | 0.79 | 4.9/3.0 (0.60) | 20.6/1.6 (0.20) |
| **iii_b after push top x=+40** | **0.954** | **1F1S** | **0.82** | 1.00 | 0.22 | **10** / 2 | 0.08 (y) | 0.06 (y) | 0.71 | 8.1/9.4 (0.89) | **4.5**/3.6 (0.77) |
| iv_decoy (K=4) | **0.78** | 1F1S | 1.00 | 0.99 | 0.22 | 2 / 4 | 0.26 (y) | 0.12 (y) | 1.00 | 8.6/8.0 (0.92) | 7.7/12.3 (0.96) |
| iv_decoy seed 1 | 0.87 | 1F1S | 1.00 | 0.99 | 0.15 | 2 / 5 | 0.21 (y) | 0.21 (y) | 0.96 | 9.2/7.7 (0.93) | 10.7/11.5 (0.91) |
| iv_decoy, coarse truth | 0.89 | 1F1S | 1.00 | 0.98 | 0.18 | 3 / 6 | 0.08 (y) | 0.27 (y) | 1.00 | 5.6/7.9 (0.98) | 9.3/12.9 (1.00) |
| v_a top-hat + short ℓ | 0.90 | 1F1S | **0.00** | 0.60 | 1.50 | **60** / 17 | 0.01 (y) | 0.68 (y) | **0.42** | 17.8/8.3 (0.72) | 54/25 (0.49) |
| v_b nonlinear support | 0.49 | 1F2S | 0.99 | 0.38 | 1.14 | 6 / 17 | 0.26 (y) | 1.15 (**n**) | 0.83 | 6.4/5.3 (0.80) | 9.3/5.4 (0.49) |
| v_b after designed probe (top x=+40) | 0.30 | **2F1S** | 1.00 | 0.71 | 0.98 | 4 / 11 | 0.46 (n) | 1.04 (n) | 0.71 | 5.6/7.5 (0.93) | 5.7/4.5 (0.59) |

Configuration evidences (log Z) and all role posteriors are listed in `runs/patch-roles-v01/summary.json`.

### Regression against the published 2-patch case (scene i)

- With roles learned at K=3, 0.94 of the mass is on 1F1S. The remainder is 1F2S 0.05 and 2F1S 0.01; every zero-force configuration has log Z ≈ −4234, against 2952 for 1F1S. The force and support centres are within 1 mm of the truth.
- Within 1F1S, an independent run of the published dense `MarginalGaussianLikelihood` with published `body.solve`, `surface.load_cov`, `bank.BOUNDS` and `run_smc` gives the same posterior, both for seed 0 and seed 1. The truth is (9.68, 0.40, 12.5 mm, 8.99, −20.0 mm).

| quantity | patch model, seed 0 | published code, seed 0 | patch model, seed 1 | published code, seed 1 |
|---|---|---|---|---|
| log E | 9.81 ± .19 | 9.82 ± .21 | 9.79 | 9.80 |
| m1 | 0.21 ± .15 | 0.23 ± .17 | 0.21 | 0.21 |
| support x (mm) | 12.98 ± 1.75 | 13.18 ± 1.78 | 12.83 | 13.02 |
| log k | 9.11 ± .22 | 9.12 ± .22 | 9.11 | 9.11 |
| load x (mm) | −20.98 ± 1.33 | −20.86 ± 1.44 | −21.16 | −21.04 |
| log Z | 2951.70 | 2951.18 | 2951.61 | 2951.51 |

- The behaviour also reproduces the published caveat: fine-mesh truth lowers coverage. Hidden coverage is 0.73 here and 0.88 with coarse truth; new-command coverage is 0.78 and 1.00.

### What works
1. **Role counts and locations when the model is right and there is no structural confound.**
   - i, ii, ii_h and iv all put ≥0.78 mass on the true configuration.
   - In those four scenes, force and support patches are found within 1–5 mm with P(near) ≥ 0.98.
   - Force-history 90% coverage is 0.94–1.00 in i and iv.
2. **Sparsity.** In iv (K=4, truth 1F1S), the extra patches are switched off.
   - The true configuration gets 0.78 and 0.87 over two seeds.
   - The residual mass is 2F1S/1F2S with E[#spurious] = 0.15–0.22.
   - log Z of 3F1S, 1F3S, 2F2S and 0F4S is 3–15 nats below 1F1S. With pOff = .5 the Occam factor of the extra amplitude dimensions does most of the work; the prior contributes only about 1–2 nats per patch.
3. **Stiffness-vs-force-scale ambiguity and the calibrated probe** (ii_h, hierarchical τ, true τ = 1.8).
   - From the unknown-load episode alone, corr(log E0, log τ) = 0.93 and τ ∈ [0.96, 2.35], with new-command risk 0.65–0.68.
   - The design ranks probes far above looks for the new-load target: J = 0.051 for P front +y x=+40 against 0.60–0.66 for looks. Looks are best for the hidden target, with hidden sd 6.2 against 6.8 µm.
   - After the executed probe, corr = 0.40, τ ∈ [1.71, 2.34], log E0 error falls from 0.24 to 0.00, new-command RMSE halves from 8.4 to 4.3 µm, and risk drops to ≤0.05.
   - This reproduces the paper's "views for hidden motion, probes for new loads" split inside the role model.
4. **Force-vs-support confound broken by one push** (iii_b, a synchronous unknown force).
   - The unknown-load data put 0.93 on 0F1S, i.e. no force at all, at both seeds (0.82 at seed 1). The force is absorbed by the material field: m1 is 0.60 against a true 0.30, a stiffer right half. log Z(1F1S) − log Z(0F1S) = −5.5 nats.
   - One 0.08 N calibrated push above the confound region (top −z, x=+40 mm) moves this to **+39.5 nats**. P(1F1S) rises from 0.004 to 0.954, the force is located within 10 mm (P(near) = 0.82), new-command RMSE falls from 20.8 to 4.5 µm, and coverage rises from 0.22 to 0.77.
   - The mechanism: in the reset episode the hidden force is absent, so the material explanation predicts the wrong compliance.

### Failure modes found (honest list)
1. **The design cannot target an ambiguity the posterior does not contain.**
   - In iii_b the posterior is confidently wrong. Every action had expected C2 risk ≈ 0, so the rule chose *proceed*, and the designed probe (tie-broken by squared error) was top x=−40 mm. That probe did not help: P(1F1S) went from 0.004 to 0.009, and coverage stayed at 0.20.
   - The breaking push in item 4 above was **chosen by me from knowledge of the scene**, not by the planner.
   - The realised C2 error exceeded the tolerance while the posterior risk was 0.00.
2. **The marginal likelihood does not "tie" force vs support; the Occam factor picks one, sometimes wrongly.**
   - iii_a (support truth): support explanations beat the force explanation 1F1S by about 9 nats, which is correct.
   - iii_b (force truth): the wrong, simpler explanation wins by 5.5 nats.
   - So the classic confound shows up as a confident mis-ranking that depends on prior volumes, not as a 50/50 tie.
   - Among support-count configurations in iii_a the evidences are within about 3 nats (0F1S/0F2S/1F2S/0F3S), and the result is seed-sensitive: true configuration 0.77 against 0.98.
   - The compliant second support is poorly localised: 19–27 mm error, P(near) ≤ 0.33. A designed probe sharpened its location to 7–8 mm but shifted mass to 0F3S. The push above it raised P(0F2S) to 0.95 but worsened localisation.
3. **Model mismatch produces confident, wrong geometry (v_a).** With a top-hat footprint and short correlation length, the role count is right (0.90 on 1F1S), but the force patch jumps to the unobserved bottom face: z ≈ 2 mm, 60 mm from the true top centre. Force coverage is 0.42 and new-command coverage 0.49.
4. **Nonlinear support produces spurious patches (v_b).**
   - The model adds a second support (1F2S 0.50).
   - After the probe, whose small-amplitude response sees a softer support, it adds a spurious *force* patch (2F1S 0.44) to reconcile the stiffened original episode.
   - log k is not covered (error 1.0–1.15) and log E0 coverage is lost.
   - The risk is at least flagged: 0.5–0.8 risk on all commands, all of which exceeded tolerance.
5. **Overconfident uncertainty generally.**
   - Hidden-displacement coverage is 0.34–0.73 in i, ii, ii_h and iii, against the nominal 0.90. New-command coverage is 0.22–0.82.
   - The coarse-truth control (i, iv) restores 0.88–1.00. Fine-mesh discretisation error (a few µm against posterior sds of 3–8 µm) is therefore a main cause there, as in the paper's 86% fine-mesh result, made worse here because 320 observations shrink the posteriors further.
   - **In ii the coarse-truth control does not restore coverage** (hidden 0.46 → 0.55; log E0 error 0.30 → 0.09). A 4× sampler-budget check rules out SMC under-sampling (§5a). The cause is unresolved.
   - The threshold risk is ordinal but not calibrated. Over 56 command predictions (fine truth, seed 0, all stages):

     | predicted risk | n | mean predicted risk | realised exceedance |
     |---|---|---|---|
     | < 0.1 | 20 | 0.02 | 0.45 |
     | 0.1–0.5 | 22 | 0.28 | 0.59 |
     | ≥ 0.5 | 14 | 0.68 | 0.79 |

6. **SMC evidence noise matters for the roles.**
   - log Z of the same configuration differs by up to 7.8 nats between seeds in 15–17 dimensions: iii_a 1F2S 2981.2 against 2973.4. In 5–9 dimensions the difference is ≤ 1.6 nats.
   - Role posteriors moved by 0.1–0.2 between seeds (iii_a, iii_b, iv). The qualitative conclusions did not change.

### 5a. SMC budget check (scene ii, 1024 particles, 10 moves)
Seed 2 with 4× the budget (1024 particles × 10 moves, 508 s) reproduces seed 0. log Z(2F1S) is 2845.4 against 2845.1. Hidden RMSE/sd is 14.7/7.1 µm, with coverage 0.45 against 0.46. log E0 error is 0.28 (not covered) against 0.30. The 1F2S evidence is unstable (662 against 836), but that configuration has negligible mass.

So the undercoverage in ii is **not** SMC under-sampling. The likely causes, not isolated yet:
- Discretisation: partly. The coarse-truth control gives 0.55.
- Single-realisation tail behaviour: possible, because the 240 hidden coordinates are strongly correlated, so the per-coordinate coverage of one case is not an independent-sample estimate.
- Weak identifiability of the m1/E0 split with two force patches on two faces.

Settling this needs a multi-case cohort (≥16 truths per scene). That was out of scope for this single-case study.

## 6. Compute
- Hardware: GPU A (2080 Ti) and GPU B (180 W cap), float64, 2 CPU threads per process, memory-capped `pr-roles-*` units.
- One configuration takes 35–75 s: 512 particles, 20–50 tempering stages, 65k–150k batched FEM solves at about 0.5 ms each. Zero-force configurations collapse in 2 s.
- A K=3 scene takes 3–5 min per inference stage; K=4 takes about 4 min. Truth generation takes 0.2 s (75 s for the nonlinear support).
- The published dense-likelihood reference SMC took 190 s on CPU, against 68 s for all six configurations of the patch model.
- All runs reported here sum to about 62 min of process time, run on 2 GPUs in parallel (about 50 min elapsed), plus 8.5 min for the §5a check.

## 7. Recommendation for step 2 (hidden-command variants of Fig. 1 S2 and S1)

**What carries over.** Role inference works when the unknown loading is not proportional to a known load and the model family is right. A push on a known region resolves the force/support/material confounds. The planner, however, cannot be relied on to find that push.

**Prerequisites before using `scripts/fig1_v01` truths:**
1. **Geometry and supports.**
   - S2 is a 0.24×0.10×0.12 m block with y up, a glued bottom, a 30 N press over a 4×4 cm window, an 8× stiff inclusion, nonlinear neo-Hookean T10, and implicit Newmark.
   - S1 is a clamped stem with a 20 N push.
   - Needed: a body class for these geometries (`fig1_v01/block.py` `mesh()` plus the `stem_trial_v01/fem` mesh), with the patch surface parametrisation over all faces, not only the lateral ones.
   - The glued base can be a fixed known support, with nS ≥ 0 extra patches allowed, which also removes the singular nS = 0 exclusion.
2. **Material family.** The S2 truth is a piecewise-constant inclusion (left, centre or right). The two-Gaussian field will be mismatched, just as v_a's footprint was. Use the Fig. 1 block bank's inclusion parametrisation (position, contrast, background E, ν), so that material and patch roles are not confounded through m1. In iii_b, m1 absorbed the force.
3. **Hidden command.** Hide D1's press magnitude and history; S2 D1 is the press at x=+0.05.
   - Keep D0 (gravity release) known in d_θ.
   - Score D2 (press at x=−0.07) as the new known command.
   - The S2 press is a box-window body load, not a Gaussian traction. Expect a v_a-type footprint mismatch: allow width up to 30 mm, and check against a top-hat footprint variant.
   - Large-strain neo-Hookean at 30 N on 60 kPa is outside the affine-in-force assumption. Run first at reduced press amplitude, so that the truth stays in the linear regime, or linearise about the gravity state.
4. **Dynamics.** The Fig. 1 truths are Newmark with a 0.25 s ramp. The quasistatic marginal applies only if inertia is negligible over 25 frames at 48 fps. Check the quasistatic-vs-dynamic residual on the truth first. Otherwise H must include the modal or Newmark response to unit amplitude histories; it stays affine and the marginal is unchanged.
5. **Observations.** Use the sealed tracks from the Fig. 1 RGB pipeline (27 top-surface probes; 4 stations for S1), with noise estimated from D0 as in eq. (tracking).
   - Expect far fewer, noisier rows than the 320 here, so role posteriors will be broader.
   - Do not touch the sealed predictions. Write a new namespace and treat it as a post-hoc study.
6. **Sampler.** Raise the SMC budget (≥1024 particles, ≥10 moves) or add per-configuration seed replicates whenever role posteriors decide the outcome.
7. **Probe and risk.** Include a fixed calibrated push near each MAP force patch as a candidate. In iii_b the planner's candidate set never contained the decisive push, because none was near the hidden patch.
   - Add a model-discrepancy noise term, from the coarse/fine mesh difference or the Fig. 1 COMSOL check (0.08–0.25 mm), before trusting threshold risks. The calibration table above shows they are optimistic.

**First runs, in order:**
1. S2 with D1 hidden in the linear regime (low press), against the oracle known-command fit.
2. S2 at full press.
3. S1, hiding the 20 N push and keeping the clamp known.
