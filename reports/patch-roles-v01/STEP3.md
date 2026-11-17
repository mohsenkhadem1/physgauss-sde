# Patch roles, step 3: correlated discrepancy, volume force patches, reruns, probe test

This is a development-mode study, so settings were tuned with the truths in view. Every tuned choice is listed in `runs/patch-roles-v02/step3/TUNING-LOG.md`, so that a later prospective run knows what to freeze.

- Code: `scripts/patch_roles_v02/step3_*.py`, plus `probe_chain.sh` and `obs.py` from step 2.
- Outputs: `runs/patch-roles-v02/step3/`, containing `cases/`, `disc/`, `bankvol/`, `summary*.json`, `calibration*.json`, the diagnostics, `TUNING-LOG.md` and `logs/`.
- The three new low-amplitude probe truths and tracks went into `runs/patch-roles-v02/{truth,rgb,scenes}`.
- The Fig1, joint_field_identification and stem_trial code and runs, and the sealed runs, were not modified. Nothing was committed or installed.
- **S1 (stem with the push hidden) was not run.**

## 1. Gate: does a correlated discrepancy remove the spurious supports? **No.**

The gate test is oracleK on Bright: D0 and D1 with their commands known, and K = 2 support patches allowed. The table gives log Z relative to the no-extra-patch configuration (0F0S); the gate passes only if both differences are ≤ 0.

| discrepancy model | full (30 N): 0F1S / 0F2S | low (10 N): 0F1S / 0F2S |
|---|---|---|
| step 2 (independent per marker coordinate, 0.15 mm) | +220 (0F2S) | – |
| v1: Matern_t × SE_space over markers; hyperparameters from a commanded-D0 fit (ℓ_s came out at 12 mm, i.e. effectively independent) | **+191 / +232** | **+219 / +281** |
| v2: v1 with hyperparameters inferred jointly | 0F0S log Z changed by only +12 / +0.3; run stopped | – |
| v3: per-marker term plus smooth term, fit on D0+D1 | 0F0S log Z +27 / +6; run stopped | – |
| **v4: an episode-shared term plus an episode-specific term** (each a²I + b²·SE, common time kernel), fit on commanded D0+D1 | **+186 / +224** | **+44 / +71** |

**Diagnosis** (`step3_diag.py` and `discrepancy_diag`; both use the truth):
- At 10 N, about 85% of the coarse-linear model's error is common to all episodes of a world (the gravity-response error); at 30 N it is 40–60%.
- The shared term therefore cuts the low-amplitude gap by a factor of 4, but cannot close it.
- With the commanded loads known, two springs reduce the noise-free D1 model error from 0.060 to 0.050 mm. The springs model a genuine structural error of the coarse linear model; they are not fitting noise.
- A ±5% correction of the lowest modal frequencies removes almost nothing, so this was not pursued.
- **v4 was used for every rerun.**

## 2. Volume force patches

`step3_forcediag.py` uses the true material. A surface traction of the same total force moves the markers more than the truth's 1 cm-deep box load (κ = 0.90–0.96 at h = 20 mm, and 0.75–0.89 at h = 15 mm), so the model needed less force to match.

The new force patch is a uniform body load in a 1 cm layer under a soft square of half-size h, with h inferred in [5, 40] mm. Its static and modal responses come from a new volume bank (`step3_bank.py`, 111 materials, about 10 min CPU). It reproduces the body-box press: κ = 0.99–1.01 against the model and 1.00–1.02 against the truth. The inferred h was 22–29 mm (truth 20 mm).

A second source of the force bias was prior shrinkage: one shared force scale τ, with posterior τ ≈ 0.2, pulled the vertical force toward zero. Separate scales for x, y and z were added after a truth-informed check. In that check the conditional force moved from −8.9 to −9.7 N as τ was relaxed.

**Identity tests** (all pass, against dense Gaussians):
- the correlated and shared Woodbury marginals, to 1e-9;
- the box footprint integrates to 1, with area 16.0 cm² at h = 20 mm.

## 3. Hidden reruns (v4 discrepancy + volume force patches, K = 2)

Sampling budget: support configurations 2048 particles × 12 moves, others 1024 × 10; two seeds each. Truth: one force at (50, 630, 0) mm, 4×4 cm, and no extra support. D2 and the hidden D1 response are reported as the Fig-1 probe RMS in mm.

| case | seeds | MAP config (P = 1.00 in every run) | loc err (mm) | Fy at 0.5 s [90%] | truth covered | D2 s3 / s2 / oracle | D1 hidden s3 / s2 / oracle |
|---|---|---|---|---|---|---|---|
| low Bright | 0, 1 | 1F1S, support at top x ≈ −11 mm, k ≈ 230 N/m | 4.8, 4.9 | −8.9 [−10.1, −7.8]; −9.0 [−10.2, −7.8] | y, y | 0.112 / 0.087–0.103 / 0.113 | 0.093 / 0.088–0.211 / 0.096 |
| low Bleft | 0, 1 | 1F1S, support near base (back / front) | 1.4, 1.6 | −10.4 [−10.9, −9.9]; −10.5 [−11.0, −10.0] | y, y (edge) | 0.043–0.045 / 0.045 / 0.049 | 0.115–0.122 / 0.165–0.174 / 0.071 |
| low Bcentre | 0, 1 | 1F1S, support on front face, x ≈ 84 mm | 4.1, 4.0 | −9.3 [−10.1, −8.4]; −9.1 [−9.9, −8.2] | y, n | 0.062 / 0.061 / 0.062 | 0.162–0.168 / 0.184–0.206 / 0.049 |
| full Bright | 0, 1 | 1F1S, support at top x = −40 mm, k ≈ 330 N/m | 3.8, 4.0 | −28.9 [−31.2, −26.6]; −28.8 [−31.1, −26.5] | y, y | 0.260–0.261 / 0.248–0.254 / 0.253 | 0.093–0.095 / 0.087–0.089 / 0.112 |
| full Bleft | 0, 1 | 1F1S, support on front face near base | 1.0, 1.0 | −31.9 [−33.0, −30.8]; −32.0 [−33.2, −30.8] | n, n | 0.087 / 0.074 / 0.080 | 0.351–0.359 / 0.573–0.578 / 0.227 |
| full Bcentre | 0, 1 | 2F0S. s0: the press split into two patches (34 and 53 mm). s1: a spurious second force on the right face | 4.1, 3.0 | −28.4 [−31.2, −25.8]; −24.1 [−29.2, −18.8] | y, n | 0.190 / 0.190 / 0.190 | 0.397–0.433 / 0.526–0.531 / 0.131 |

**Role set.**
- P(1F0S) = 0.000 in all 12 runs. The force is always found at the window (P(F@press) = 1.00). The extra patch is a weak spring, or in full Bcentre a split or spurious second force.
- Step 2's failure where the force was lost entirely (low Bright seed 1, which chose two supports) is gone: 1F0S now beats 0F2S by about 570–600 nats.
- The spurious-support margin over 1F0S is +35 to +62 nats at 10 N (step 2: +136) and +123 to +209 at 30 N.

**Force.**
- Location error is 1.0–4.9 mm (step 2: 3–17 mm).
- Final-force bias is −11% to +7%, except full Bcentre seed 1 at −20%. Step 2 was −3% to −27% and always low.
- The truth is covered in 7 of 12 runs (step 2: 6 of 16 force-finding runs). Whole-history coverage is 0.54–0.83.

**Seeds.**
- Both seeds give the same MAP configuration in every case.
- MAP-configuration log Z agrees within 0.4–9 nats.
- Other configurations differ by up to 35 nats (full Bcentre 2F0S) and 29 nats (low Bright 0F2S).

**Prediction.**
- D2 equals the oracle within 0.01 mm everywhere, as in step 2.
- The hidden D1 response improves where the press was mis-represented in step 2: full Bleft 0.57 → 0.35 mm, full Bcentre 0.53 → 0.40–0.43 mm, low Bleft 0.17 → 0.12 mm, low Bcentre 0.19–0.21 → 0.16–0.17 mm.
- It is still 1.5–3× the oracle in those worlds.

**Predictive uncertainty.**
- With only the smooth discrepancy parts in the predictive (the pre-chosen setting), sd is 0.008–0.076 mm. That is too narrow: D2 coverage is 0.45–0.93 and hidden-D1 coverage 0.46–0.97.
- Adding the per-marker parts as well, a post-hoc reanalysis of the same particles (`--pred-sd all`, `summarypredall-.json`), gives D2 coverage 0.86–1.00 and hidden-D1 coverage 0.92–1.00, with sd 0.07–0.21 mm.

## 4. Probe test (low Bright) and unified risk

- **One risk definition** is used for both the report and the planner: risk = P(RMS₃D > 0.25 mm), evaluated on a 512-particle resample with 32 common discrepancy draws.
- The candidates were 7 grid presses plus the bank presses nearest the MAP force window (x = +0.05) and the MAP support (x = 0.0).
- **The planner chose "proceed"**: every action had expected risk 0.000 for both seeds, because the posterior D2 error is about 0.11 mm against a 0.25 mm threshold.

I executed two presses anyway:

| low Bright, seed 0 | log Z(1F1S) − log Z(1F0S) | force loc | Fy [90%] | D2 | D1 hidden |
|---|---|---|---|---|---|
| no probe | +53 | 4.8 mm | −8.9 [−10.1, −7.8] | 0.112 | 0.094 |
| + press x = +0.05 (at the MAP force window) | +47 | **2.6 mm** | **−9.6 [−10.7, −8.6]** | **0.097** | **0.077** |
| + press x = −0.03 (near the MAP support) | +55 | 6.1 mm | −8.5 [−9.6, −7.4] | 0.113 | 0.101 |

- The press at the window sharpens the force estimate and improves both predictions by about 15%, but it does not remove the spurious support. The press near the support does neither.
- The planner cannot value the useful press because its risk is already 0.

**Risk calibration** (20 runs × D2 and D1 targets):

| predictive | risk < 0.1: realised exceedance | risk ≥ 0.5: realised exceedance | Brier |
|---|---|---|---|
| smooth discrepancy parts only | 7 of 40 exceeded, all at risk ≈ 0 | – | 0.17 |
| all discrepancy parts | 3 of 28 | 4 of 12 | 0.23 |

The risk is essentially a per-world constant set by the discrepancy level. It does not separate the cases that exceed 0.25 mm (full Bright D2; full Bleft and Bcentre hidden D1) from the cases that do not. It is not calibrated.

## 5. Compute

- About 20.0 GPU-process hours of SMC across the 24 step-3 results, plus about 1 h of post-hoc reanalysis and about 1.5 h of stopped or superseded runs.
- A hidden case takes 72–81 min. The one-force-plus-one-support configuration alone takes 37–45 min at 2048 × 12.
- Discrepancy fits take 1–3 min per world; the volume bank about 10 min CPU; the new probe truths about 8 min CPU each.
- Hardware: GPUs A and B only.
- The 3-process limit was kept throughout. Twice, a cancel command of mine (`kill` on a `ps | grep` match) hit my own shell instead of the intended waiter script. No job was affected.

## 6. Next step

The remaining blocker is structural error in the forward model, not the noise model: a coarse linear mesh against a nonlinear fine truth. Four things to try:

1. **Shrink the model error itself.** Options are a finer or p-refined inference mesh near the top, or linearising about the gravity-loaded state, since the nonlinearity is mostly gravity–press coupling. Then re-run the oracleK gate.
2. **Alternatively, make the discrepancy physical:** a GP prior on a smooth log-E field perturbation pushed through the linearised model, which can absorb local-stiffness errors that look like springs.
3. **Make risk informative:** use relative or episode-specific thresholds, kriging of the shared discrepancy into D2, and calibration on a multi-truth cohort.
4. **S1** remains to be done.
