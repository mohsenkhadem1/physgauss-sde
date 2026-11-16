# Patch roles, step 2: hidden-command variants of Figure-1 S2 (block press)

Post-hoc research study. Not paper text and not part of the sealed Figure-1 benchmark.
- Code: `scripts/patch_roles_v02/`.
- Outputs: `runs/patch-roles-v02/`, containing `bank/`, `truth/`, `rgb/`, `scenes/`, `cases/<case>/result-s<seed>.json` with cached SMC particles, `summary.json`, `tests.json`, `discrepancy_diag.json` and `logs/`.
- Nothing under `scripts/fig1_v01`, `scripts/joint_field_identification`, `scripts/stem_trial_v01` or `runs/fig1-v01` was modified. Those files were imported or read only; the sealed predictions were not touched.
- Variant (c), S1 stem with the push hidden, was **not run**. There was not enough time.

## 1. Question and variants

Figure-1 S2 setup:
- Block 0.24 x 0.10 x 0.12 m with the bottom glued, gravity switched on at t = 0, and a hidden 8x stiffer inclusion at xc = -0.06, 0 or +0.06 (worlds Bleft, Bcentre, Bright).
- Each episode is a 30 N press on a 4x4 cm, 1 cm deep box, ramped over 0.25 s. Over 0.5 s, 25 frames are recorded.
- D0 is a central press, D1 a press at x = +0.05, and D2 a press at x = -0.07 (held out).
- Observations are the 20 RGB-tracked, two-view-triangulated markers, using the unchanged Fig-1 pipeline.
- Scores: the Fig-1 metric, i.e. the 3-D RMS over 27 top-surface probes and all frames, for **D2**; and the same metric on the probes for the **hidden D1 response** (the "hidden target").

| variant | episodes given to the model | configurations |
|---|---|---|
| `oracle` (commanded load, the non-hidden comparison) | D0 known + D1 known | 0F0S |
| `hidden` | D0 known + D1 with the command **hidden**: roles, location and 3-D force history are inferred, with amplitudes marginalised | all (nF, nS) with nF + nS <= K = 2 |
| `d1only` | D1 hidden only. Material can then come only from gravity and the hidden episode | same |
| `oracleK` | as oracle, with support patches allowed (control for spurious roles) | 0F0S, 0F1S, 0F2S |
| `+probe` | hidden, plus one executed calibrated press (known command), also tracked by RGB | as hidden |
| `robust` | hidden, after markers flagged as tracking outliers by a D0-only commanded-load fit are dropped | as hidden |

Amplitudes:
- **(b) full**: the existing Fig-1 truths and RGB tracks at 30 N.
- **(a) low**: new truths at 10 N (`truth.py`). These use the same nonlinear neo-Hookean T10 (18,8,9) mesh, Newmark pipeline, renderer and tracker (`obs.py`), with only the press magnitude scaled. All new truths are in `runs/patch-roles-v02/truth/`.

**The force-linearity worry from step 1 is mostly unfounded at the observation level** (`truth.py --linear` on the truth mesh, Bright, D1):
- At 30 N, the press part of the nonlinear response differs from linear by 0.025 mm RMS at the markers, against a press signal of 0.60 mm.
- Gravity's own nonlinearity is 0.039 mm, and the total is 0.040 mm.
- At 6 N the press-part nonlinearity is still 3.4%. It is dominated by press-gravity coupling and dynamic phase, not by the press amplitude.
- So "reduced amplitude" does not make the problem much more linear. It mainly lowers the signal-to-tracking-noise ratio. The low amplitude was set to 10 N (scale 1/3), so that the press part stays about 2-3x the tracking noise.
- Full amplitude (b) was run with the same linear model, with no linearisation. Its nonlinearity is below the tracking noise.

Tracking pipeline check:
- `opencv` is not installed in `.venv`. The tracker was run with the cv2 package of another local Python 3.12 venv, exposed through a path shim (`runs/patch-roles-v02/cv2shim`). Nothing was installed.
- Re-rendering the Fig-1 D1-Bright truth and re-tracking it this way reproduces the sealed Fig-1 track file to 1e-12 mm. The pipeline is therefore identical.

## 2. Model (`model.py`, `bank.py`)

**Body and material.**
- Linear T10 on the Fig-1 ours mesh `block.mesh((12,5,6))`, with 9750 free dofs and the bottom glued (a known support). Density is known. Gravity is a known load in every episode.
- Material follows the Fig-1 ours inclusion family. xc takes values in {-.08, ..., +.08} in 0.02 steps, contrast in {2, 4, 8, 16}, or there is no inclusion; nu is in {.3, .4, .45}.
- That gives 111 grid materials. The soft modulus E is **continuous** (log-uniform on [2e4, 2e5]) and enters exactly as K = E Khat. The SMC moves over continuous coordinates that are binned to the grid, and the prior matches Fig-1's uniform-over-hypotheses prior.

**Exact elastodynamics, by mode acceleration.**
- The Fig-1 truths ring strongly: the gravity release from rest gives about 0.65 mm undamped oscillations at roughly 9-12 Hz. A quasistatic model is therefore not usable.
- Per grid material, `bank.py` stores the 80 lowest M-orthonormal modes, the static adjoint Khat^-1 H^T for the 141 observation and probe rows at the surface dofs, the surface-to-surface static compliance (float32, 2919^2), and the known loads. These are gravity and 1 N presses on the protocol box at 9 x positions.
- Response: u(t) = Khat^-1 f a(t)/E + sum_k phi_k phi_k^T f [eta_k(t) - a(t)/(E lam_k)]. The modal eta_k use the truth's own Newmark recursion (dt = 1/480).

**Patches.**
- K exchangeable patches on all five non-glued faces. The centre (u, b) lies in [0,1]^2, mapped area-uniformly to left/top/right/front/back. Log width s is in [6, 30] mm. The footprint is a normalised Gaussian.
- **Force patch:** a 3-D traction amplitude, piecewise linear on the 24 frame knots, with a(0) = 0 because the episode starts at rest. Prior N(0, (10 N · tau)^2 Matern32(0.15 s) x I3), with hierarchical log tau in [ln .1, ln 10].
- **Support patch:** a rank-3 spring k on the footprint-averaged displacement, with log k in [ln 1e2, ln 1e6]. It is present in every episode. The reaction is solved exactly at every Newmark step inside the modal recursion.
- An earlier frame-knot collocation for the support was 25% wrong and was replaced.
- Role prior and enumeration are as in step 1: P(off, F, S) = (.5, .25, .25), with x-ordering within roles. nS = 0 is now admissible because of the glued base.

**Noise and discrepancy.** The Woodbury marginal is exact after time-whitening.
- **disc** (main, pre-declared before any run): white sigma_w (inferred, log-uniform 0.01-1 mm) plus a model-discrepancy term sigma_d = 0.15 mm Matern32(l = 0.05 s) in time. The discrepancy is independent across marker coordinates and episodes. The level is the COMSOL-check scale.
- The predictive for the true response adds the discrepancy.
- **white**: sigma_w only.

**Sampler.** The published `continuous.run_smc`, unchanged: **1024 particles, 10 RW-MH moves**, conditional ESS 0.8, run per configuration. Seed replicates are listed in Section 4.

**Identity tests** (`tests.py`, `runs/patch-roles-v02/tests.json`, all pass). Comparison is against full sparse linear Newmark on the same mesh:
- Known loads: max relative error 0.8%. A force patch: 1.4%. A force patch plus a support spring in K: 1.0%. The residual is attributed to the 80-mode truncation.
- The Woodbury marginal equals the dense Gaussian log density (white + discrepancy + force covariance) to 5e-12.

## 3. Results

Abbreviations and conventions:
- Truth configuration: 1F0S, i.e. one force at the D1 window (centre (+50, 630, 0) mm, a 4x4 cm box load), with no extra support.
- "P(F@press)": the posterior probability that some force patch is centred within 20 mm of the window centre.
- "loc err": the mean distance from the MAP-configuration force patch nearest the window to the window centre.
- Force coverage: 90% coverage of the resultant 3-D force history (24 frames x 3 components) against the truth [0, -F ramp(t), 0]. Fy(0.5 s) is the resultant vertical force at the last frame.
- D2 and "hidden": the Fig-1 probe RMS in mm. sd is the predictive sd; cov is 90% coverage per entry.
- All rows use the disc noise model, K = 2, 1024 particles, 10 moves, unless the row says otherwise.

### 3a. Low amplitude (10 N), D0 known + D1 hidden

| world | seed | MAP cfg (mass) | P(1F0S) | P(F@press) | loc err mm | force cov | Fy(0.5 s) [90%] (true -10) | dlogE (90% cov) | P(xc, c=8, nu true) | D2 rmse/sd (cov) | hidden rmse/sd (cov) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bleft | 0 | 1F1S (1.00) | 0.000 | 1.00 | 4.8 | 0.72 | -8.9 [-9.7, -8.2] | 0.004 (n) | 1, 1, 1 | 0.045/0.150 (1.00) | 0.174/0.155 (0.97) |
| Bleft | 1 | 1F1S (1.00) | 0.000 | 1.00 | 3.2 | 0.76 | -8.8 [-9.6, -8.1] | 0.005 (n) | 1, 1, 1 | 0.045/0.150 (1.00) | 0.165/0.155 (0.97) |
| Bcentre | 0 | 1F1S (1.00) | 0.000 | 1.00 | 7.5 | 0.50 | -8.2 [-9.6, -6.7] | 0.005 (n) | 1, 1, 1 | 0.062/0.150 (1.00) | 0.206/0.154 (0.97) |
| Bcentre | 1 | 1F1S (1.00) | 0.000 | 1.00 | 6.0 | 0.35 | -7.3 [-8.8, -5.9] | 0.004 (n) | 1, 1, 1 | 0.061/0.150 (1.00) | 0.184/0.155 (0.98) |
| Bright | 0 | 1F1S (1.00) | 0.000 | 0.84 | 16.9 | 0.74 | -9.7 [-12.2, -7.2] | 0.012 (n) | 1, 1, 1 | 0.103/0.150 (0.99) | 0.088/0.156 (1.00) |
| Bright | 1 | **0F2S (0.99)** | 0.000 | **0.01** | - | 0.67 | **-0.1 [0, 0]** | 0.019 (n) | 1, 1, 1 | 0.087/0.150 (1.00) | 0.211/0.151 (0.92) |
| Bright, white noise | 0 | 1F1S (1.00) | 0.000 | 1.00 | 5.2 | 0.78 | -9.6 [-10.7, -8.6] | 0.010 (n) | 1, 1, 1 | 0.098/**0.007 (0.14)** | 0.067/0.017 (0.61) |
| Bright, **d1only** | 0 | 2F0S (1.00) | 0.000 | 0.99 | 9.3 | 0.92 | -9.4 [-11.8, -7.0] | 0.002 (n) | 1, 1, 1 | 0.115/0.150 (0.99) | 0.059/0.157 (1.00) |
| *oracle* (commanded) Bleft / Bcentre / Bright, seeds 0 and 1 | | 0F0S | - | - | - | - | - | 0.009 / 0.002 / 0.003 (n) | 1, 1, 1 | 0.048 / 0.062 / 0.113 (1.00/1.00/0.99) | 0.071 / 0.049 / 0.095 |
| *oracle, white* Bright | 0 | 0F0S | - | - | - | - | - | 0.003 (n) | 1, 1, 1 | 0.114/**0.002 (0.02)** | 0.095/0.002 (0.02) |

### 3b. Full amplitude (30 N; existing Fig-1 truths and tracks)

| world | seed | MAP cfg (mass) | P(1F0S) | P(F@press) | loc err mm | force cov | Fy(0.5 s) [90%] (true -30) | dlogE (cov) | P(xc, c=8, nu) | D2 rmse/sd (cov) | hidden rmse/sd (cov) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bleft | 0 | 2F0S (1.00), both patches at the press | 0.000 | 1.00 | 8.9 | 0.56 | -28.2 [-29.1, -27.3] | 0.006 (n) | 1, 1, 1 | 0.074/0.150 (1.00) | **0.573**/0.160 (0.82) |
| Bcentre | 0 | 2F0S (1.00), both patches at the press | 0.000 | 1.00 | 4.0 | 0.99 | -25.9 [-30.1, -23.4] | 0.004 (n) | 1, 1, 1 | 0.190/0.150 (0.95) | **0.531**/0.171 (0.89) |
| Bright | 0 | 1F1S (1.00) | 0.000 | 1.00 | 3.5 | 0.83 | -27.5 [-29.9, -25.1] | 0.016 (n) | 1, 1, 1 | 0.248/0.150 (0.92) | 0.089/0.155 (1.00) |
| Bright | 1 | 1F1S (1.00) | 0.000 | 1.00 | 3.5 | 0.83 | -27.5 [-29.9, -25.1] | 0.016 (n) | 1, 1, 1 | 0.254/0.150 (0.91) | 0.087/0.155 (1.00) |
| Bleft | 1 | 2F0S (1.00), both patches at the press | 0.000 | 1.00 | 9.8 | 0.57 | -27.8 [-28.7, -27.0] | 0.006 (n) | 1, 1, 1 | 0.074/0.150 (1.00) | **0.578**/0.163 (0.82) |
| Bcentre | 1 | 2F0S (1.00), both patches at the press | 0.000 | 1.00 | 5.0 | 0.26 | -25.2 [-27.4, -23.3] | 0.004 (n) | 1, 1, 1 | 0.190/0.150 (0.95) | **0.526**/0.165 (0.88) |
| Bright, robust (markers 2 and 7 dropped) | 0 | 1F1S (1.00) | 0.000 | 1.00 | 2.8 | 0.93 | -28.3 [-30.9, -25.8] | 0.017 (n) | 1, 1, 1 | 0.253/0.150 (0.91) | 0.092/0.156 (1.00) |
| Bright + executed press at x = -0.03 (near the MAP support) | 0 | 1F1S (1.00) | 0.000 | 1.00 | 3.9 | 0.82 | -27.4 [-29.9, -24.9] | 0.016 (n) | 1, 1, 1 | 0.226/0.150 (0.93) | 0.088/0.156 (1.00) |
| Bright + executed press at x = +0.05 (near the MAP force) | 0 | 1F1S (1.00) | 0.000 | 1.00 | 3.9 | 0.92 | -28.1 [-30.5, -25.6] | 0.014 (n) | 1, 1, 1 | 0.256/0.150 (0.92) | 0.090/0.155 (1.00) |
| *oracleK* Bright (commanded loads, supports allowed) | 0 | **0F2S (1.00)** | - | - | - | - | - | 0.019 (n) | 1, 1, 1 | 0.227/0.150 (0.94) | 0.094/0.150 (1.00) |
| *oracle* Bleft / Bcentre / Bright, seeds 0 and 1 | | 0F0S | - | - | - | - | - | 0.005 / 0.003 / 0.005 (n) | 1, 1, 1 | **0.078 / 0.190 / 0.253** | 0.227 / 0.131 / 0.112 |
| *Fig-1 sealed ours* (commanded, grid bank, plug-in noise) | | - | - | - | - | - | - | - | MAP xc true, c=4 | **0.365 / 0.339 / 0.296** | - |

Material is recovered in every run, including d1only: P = 1.00 on the true inclusion position, contrast 8 and nu = 0.4. The soft-modulus error is 0.2-1.9% (dlogE 0.002-0.019).

### 3c. Reading

**1. Load location and history are recovered when hidden; the role count is not.**
- In 16 of the 17 hidden runs a force patch sits on the D1 window: P(F@press) = 0.84-1.00 and location error 3-17 mm, within the 4x4 cm window in all but one run.
- The resultant press is recovered to 5-27% at the last frame: -7.3 to -9.7 N against -10 N, and -25.2 to -28.3 N against -30 N.
- The estimate is **biased low**, by 3-27%. The 90% interval misses the true final force in 10 of the 16 runs that found the force. Whole-history 90% coverage is 0.26-0.99, with median 0.76.
- The likely cause is the surface-Gaussian traction standing in for the truth's 1 cm-deep box body load (a footprint/representation mismatch). Not isolated.
- **Posterior mass on the true role set (1F0S) is 0.000 in every hidden run.** The MAP configuration is always one of three wrong sets:
  - 1F1S, with a spurious weak support (k = 150-2500 N/m, about 3-40% of the block's local stiffness) on a face away from the press;
  - 2F0S, with the press split into two force patches. This is a footprint-shape effect: a 4x4 cm box is fitted by two Gaussians. Both patches sit at the press, so it is not a wrong role, but the count is wrong;
  - in one seed, **0F2S, with no force at all**. Low Bright, seed 1: P(F@press) = 0.01, and the 10 N press is explained by two supports. Seed 0 of the same data gives 1F1S, with 0F2S 12 nats behind.
- So at 10 N the force-versus-support decision is **seed-dependent**, the step-1 failure mode again.

**2. The spurious roles are model-discrepancy absorption, not a consequence of hiding the command.**
- With the commanded loads known (oracleK, full Bright), two supports are still added: 0F2S beats 0F0S by **220 nats**.
- The noise-free error of the coarse linear model at the true parameters against the nonlinear fine truth (`discrepancy_diag.py`, a diagnostic that uses the truth and is not used by any inference) is 0.02-0.16 mm RMS per episode. It is spatially coherent across neighbouring markers and smooth in time.
- The pre-declared discrepancy term is independent across marker coordinates. It therefore cannot absorb this structure: with 2 x 1440 observations, a 4-parameter weak spring buys 100-250 nats.
- Dropping the two markers that a D0-only commanded fit flags as tracking outliers (`--robust`) changes nothing. Those are markers 2 and 7 under the central press, with 0.68 and 0.77 mm residuals; the flagging procedure does not use the truth. The support stays at x = -40 mm, 175 nats ahead.
- An executed calibrated press at the spurious support's location (x = -0.03) **strengthens** it, from 177 to 248 nats. That is consistent with a persistent model error rather than a structure that the press could refute.

**3. Prediction is as good hidden as commanded.**
- D2 RMSE, hidden against oracle on the same data: low 0.045/0.048 (Bleft), 0.061-0.062/0.062 (Bcentre), 0.087-0.103/0.113 (Bright); full 0.074/0.078, 0.190/0.190, 0.248-0.254/0.253.
- All are below the sealed Fig-1 ours (0.30-0.36 mm), because of the continuous E and exact modal dynamics. That improvement is a model change, not a hiding effect.
- Material and D2 are driven by D0 and gravity, which is why hiding D1 costs nothing for D2.
- Even **d1only** (no commanded episode at all) matches the oracle on D2: 0.115 against 0.113 mm. The material is fully identified from gravity, through the known load rho g and the ringing frequency, plus the hidden episode.
- The **hidden D1 response** is close to the commanded fit when one patch is found (low Bright s0: 0.088 vs 0.095 mm). It is 2-4x worse when the press is mis-represented:
  - full Bleft and Bcentre with the 2-patch split: 0.53-0.58 mm against 0.13-0.23 mm commanded, with coverage 0.82-0.89;
  - low Bright s1 with no force: 0.211 mm.

**4. Discrepancy term and overconfidence.**
- The disc term fixes the *predictive* overconfidence of step 1. D2 coverage is 0.91-1.00 with disc, against **0.14** (hidden) and **0.02** (oracle) with white noise only on the same data. White-noise D2 sd is 0.002-0.007 mm, against errors of 0.10-0.11 mm.
- With disc the predictions are conservative in the low and Bleft cases: sd 0.15 mm against RMSE 0.05 mm, coverage 1.00. The hidden target is near nominal (0.92-1.00), except in the mis-represented full Bleft and Bcentre runs (0.82-0.89).
- It does **not** fix *parameter* overconfidence. The log E 90% interval is about 0.002 wide and misses the truth in all 31 runs, by 0.2-1.9%, against an interval of about 0.2%.
- Force-history coverage stays below nominal.
- It does not stop spurious roles (point 2). The cause is the independent-per-coordinate structure, not the level.

**5. Planner and probes.**
- The candidate set included a calibrated press at the bank site nearest each MAP force patch (x = +0.05 or +0.06), and nearest each MAP top-face support (x = -0.03).
- In every run the pre-posterior D2 threshold risk was 0.000 for "proceed" and for every probe, so the planner chose **proceed** every time.
- There was no ambiguity for a probe to break. The force was already found, and D2 was not exceeded in any run.
- Executing the near-MAP-force press (x = +0.05) and the near-spurious-support press (x = -0.03) anyway did not change the role posterior. 1F1S stayed at 1.00, and the support was not removed.
- So the step-1 confound-breaking test is **not demonstrated** here. The relevant confound in S2 is the persistent model discrepancy, which a known push cannot break. The low-amplitude seed-dependent 0F2S/1F1S case (low Bright) is the one where a probe at the hidden window could matter. That probe stage was not run (no low-amplitude probe truth was made).

**6. Risk calibration.** The threshold rule is "RMS error > 20% of the predicted response RMS", with the discrepancy included in the predictive, as in step 1. It is badly calibrated in both directions.
- Over 31 D2 predictions: predicted risk >= 0.5 in 14 cases, realised exceedance 0/14; predicted risk < 0.1 in 14 cases, exceedance 0/14.
- Over the hidden target: risk < 0.1 in 10 cases, exceedance 4/10 (all full Bleft and Bcentre); risk >= 0.5 in 21 cases, exceedance 3/21.
- With a 0.15 mm discrepancy floor, a relative threshold makes small-signal targets (Bleft D2, about 0.6 mm) always "risky". The design rows use the particle mixture only, so their risk is 0 where the reported target risk is 0.45-1.0. These two risk definitions are inconsistent and should be unified.

**7. SMC evidence noise.**
- Seed-to-seed log Z differences are at most 6 nats for force-only configurations, and seeds agree on the MAP configuration in 5 of the 6 replicated worlds. For support configurations they reach 6-55 nats (low Bright 0F1S 55, 0F2S 19), where the posterior over spring location is multimodal.
- With 1024 x 10 this is the dominant source of role-posterior instability at 10 N.

## 4. Compute

- Hardware: GPUs A and B (2080 Ti), float64, 4 CPU threads per process, capped `pr2-*` units, at most 3 processes at once.
- Bank: 111 materials x about 11 s on CPU, about 25 min total over 2 units. Disk: 4.4 GB.
- Truths: 8 nonlinear truth-mesh runs (low D0/D1/D2 x 3 worlds; D2 Bleft and Bcentre made later) plus 2 full-amplitude probe presses, each 7-10 min on CPU, plus diagnostics. Render plus RGB tracking: about 1-2 min per episode.
- Inference: a K = 2 hidden case (6 configurations) takes **37-44 min** at 1024 x 10. Per configuration: 0F0S 1 min, 1F0S 3 min, 2F0S 5 min, 0F1S 7 min, 0F2S 9 min, 1F1S 16 min. Support configurations cost the most, because of the per-Newmark-step reaction solve.
- An oracle takes about 1 min; oracleK about 16 min.
- All SMC in this report: 11.7 h of GPU process time over about 7.5 h elapsed on two GPUs.

## 5. What failed or was not done
- **True role set never recovered.** P(1F0S) = 0 everywhere, because of spurious supports or split patches. One seed lost the force entirely (low Bright s1).
- Force amplitude is biased low by 3-27%, and its history is under-covered.
- log E is never covered, because the discrepancy model is independent per coordinate.
- The threshold risk is uncalibrated.
- The planner never chose a probe, and executed probes did not remove the spurious support.
- Variant (c), the S1 stem, was not run.
- Low-amplitude probe stages were not run.
- Only 3 worlds per amplitude; no multi-truth cohort.
- **Choices made with knowledge of the truth**, all in step-2 setup and none tuned on held-out D2:
  - the 10 N low amplitude (chosen after the linearity diagnostic on Bright D1);
  - the probe x positions -0.03 and +0.05 (chosen from the posterior MAP patches, using the bank grid);
  - the robust and oracleK controls, which were added after seeing the spurious supports.
- sigma_d = 0.15 mm and l_d = 0.05 s were fixed before any run.

## 6. Recommended next step
Replace the independent-per-coordinate discrepancy with a **spatially correlated** discrepancy GP: a smooth field over the marker positions times Matern in time, with its amplitude inferred, or estimated from the commanded D0 episode.
- Re-run oracleK first. The test passes if 0F0S wins with the commanded loads known.
- Then re-run the hidden variants.
- At the same time, represent a force patch as a **box/volume load family** matching the press class, or a top-hat with an inferred size, to remove the 2-patch split and the amplitude bias.
- Only then is the planner/probe test meaningful. Run it on the low-amplitude Bright case, where the force-vs-two-supports decision is seed-dependent, with a press at the MAP window as a candidate.
- Raise the sampler budget or add replicate-averaged evidence for support configurations.

