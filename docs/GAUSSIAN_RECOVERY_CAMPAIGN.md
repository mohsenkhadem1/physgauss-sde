# Known-force prediction and active probing campaign

Status: active, starting from commit `0ce8dab`. The goal is recorded in the ongoing agent thread. [Machine-readable protocol](../configs/gaussian_campaign_v01.json).

The [diagnosis](../reports/gaussian-forward-inverse-diagnosis-v01/index.html) found major forward under-response even with correct physical parameters, and genuine stiffness ambiguity in some views. The campaign starts with known-force prediction. It does not assume that more confident material estimates or a larger architecture will improve probing.

## Milestones

1. Audit force commands, time envelopes, hardware and support conventions on TRAIN data. Establish fit and generalization errors separately. Compare three training arms on the existing Green operator: common 0.4 mm loss floor, 0.04 mm control-response floor, and the smaller floor with balanced support-type sampling. No old TEST data selects a model.
2. Collect four actual calibrated controls on 32 additional TRAIN layouts and 16 VALIDATION layouts with the unchanged verified Isaac worker. Eight validation layouts select checkpoints; eight separate layouts gate known-force generalization. Reserve 16 further TRAIN layouts for discrepancy calibration. Independent diagonal responses are never training targets. Keep all failures.
3. Require useful forward improvement, plausible response amplitude/phase/direction, positive volumes and limited passive-motion regression before active inference. Compare against both static and the original frozen teacher. If the simple loss/data changes fail, record that outcome before changing representation or architecture.
4. Compare inference with a numerical posterior using the same prior and likelihood. Progress from stiffness-only to nuisance marginalization. Calibrate discrepancy on reserved TRAIN responses and assess coverage alongside proper predictive scores. Do not claim the density-slice comparison isolates full inference error.
5. Freeze the selected model, likelihood and policies before predictive evaluation on 32 fresh layouts. Compare one-poke selection with fixed and random policies at the same cost. A paired confidence interval that includes no benefit is an inconclusive or negative result, never a success claim.

Acceptance thresholds and exact role IDs are in the protocol. The new confirmation recipes are frozen in [the holdout plan](../configs/gaussian_campaign_holdout_v01.json). Existing 64 TEST layouts and eight poked TEST objects remain diagnosis/development evidence after having been examined; they are not fresh confirmation data.

## Execution

GPU0 runs the first fits, GPU1 and GPU2 run one Isaac case at a time, and CPU workers handle checks and statistics. Existing display work on GPU3 is left running. Physics, fixed recipes and sensor acceptance tolerances are retained. Bird remains stopped.

Local state lives under `runs/gaussian-campaign-v01/`, including simulation coordinator ledgers, training protocols, source snapshots, histories and checkpoints. Progress summaries will be published with source and result hashes; large data and models use the existing artifact publication workflow. The earlier sealed report and frozen models stay intact.

The active campaign is research work: the milestones describe tests to pass, not results already achieved. It remains active while experiments and follow-up diagnoses are outstanding.

## First completed experiments

[First-trial report](../reports/gaussian-recovery-first-trials-v01/index.html): all three initial Green-model loss/sampling pilots completed. Better fitting on TRAIN did not transfer to the four pilot VALIDATION layouts, and none passed the forward gate. Independent TRAIN-only force-combination checks exposed a substantial mismatch with exact force superposition. Four additional unchanged-physics Isaac cases measured zero force and reproduced the earlier axis response bit for bit; zero-load drift does not explain the mismatch in the three stationary-support cases.

The next [matched nonlinear phase](../configs/gaussian_campaign_nonlinear_v01.json) compares the Green map, the existing nonlinear coordinate operator, and that same operator referenced to zero input. It adds no layers or parameters to the nonlinear candidate. Fixed training waves add the first eight, first sixteen and all thirty-two declared new TRAIN layouts as they finish verification. Calibration, reserved gate responses and confirmation data remain excluded from fitting.

The reserved control gate uses new responses on earlier validation layouts; those layouts' passive trajectories were already used for checkpoint selection. It is therefore a held-out response check, not a wholly unseen-layout confirmation. The thirty-two new `h` layouts provide the independent final comparison. TRAIN diagonal data were opened only for model-class diagnosis; they remain excluded from all weight fitting and discrepancy calibration.

[The first eight-layout selection comparison](../examples/gaussian-neural-boundary/campaign-selection-pilot.json) retains every tested checkpoint and freezes their identities before opening these responses. The zero-referenced final candidate improves mean error over static by 14.7%, but the original controlled teacher remains stronger on this set (0.469 mm versus 0.546 mm). The new candidates have not passed the gate or been adopted.

The [force-angle curriculum](../configs/gaussian_rich_controls_v01.json) adds off-axis TRAIN pushes at two amplitudes and measured zero force on four declared TRAIN layouts. Earlier axis recordings are retained as replays; the independent diagonal remains excluded from fitting. This addresses an identified coverage gap without changing the active-action menu or its comparison budget.

## Expanded data and reproducibility

The [first-eight-TRAIN wave](../examples/gaussian-neural-boundary/campaign-selection-first8.json) also falls short of the original controlled teacher on the same eight development layouts: 0.499 mm for the final zero-referenced candidate versus 0.469 mm for the teacher. The [warm transfer phase](../configs/gaussian_campaign_transfer_v01.json) uses sixteen added TRAIN layouts and equal passive/control sampling. Its [complete comparison](../examples/gaussian-neural-boundary/campaign-selection-transfer-first16.json) likewise retains a stronger original teacher; no candidate has passed the forward gate. These are known-parameter development results, not image-conditioned physical inference or fresh confirmation.

The separate [campaign archive index](../examples/gaussian-neural-boundary/campaign-artifacts.json) publishes checkpoints, full captures, source snapshots and negative results through immutable artifact commits. All 2,564 phase05 files and 9,686 phase06 files were downloaded, restored and hash verified. The [replay proof](../examples/gaussian-neural-boundary/campaign-restore-proof.json) reproduces two TRAIN predictions bit for bit for each restored nonlinear checkpoint and checks exact rest under zero force and zero prescribed motion. Earlier sealed results and their archive index stay frozen.

The [calibration plan](../configs/gaussian_campaign_calibration_v01.json) separates eight discrepancy-fit interventions from eight predictive-audit interventions. These are not forward-weight training targets. Data capture may proceed before model selection; discrepancy fitting requires a frozen forward checkpoint, and predictive audit requires a frozen likelihood. Fresh confirmation remains separate.

## Numerical reliability now takes priority

A repeatable simulator response is insufficient for physical inference. The new TRAIN-only timestep audit on `t0001` reproduces the old 960 Hz positions, velocities and commands bit for bit, yet the diagonal response changes substantially across 480, 960 and 1,920 Hz. Adjacent-rate coordinate differences are 181 and 275 micrometres, respectively; they are comparable to the response itself and increase under refinement. Force impulses agree with the analytic pulse and independent sensor/clock checks pass. This establishes failure of the tested timestep refinement, without identifying its cause or establishing which rate is accurate.

The [new numerical reliability protocol](../configs/gaussian_campaign_numerical_gate_v01.json) takes execution priority while leaving the original campaign protocol frozen. Calibration capture stopped cleanly after t0119, retaining eight fit-only cases; none of the eight audit cases started. Further fitting, bulk data capture and the active-policy comparison wait for a controlled audit of solver iterations, world-coordinate translation and rate sensitivity. All four TRAIN boundary families must meet the declared engineering checks before large-scale replacement capture; a one-layout diagnosis cannot pass that gate. Mesh and attachment fidelity remain separate limitations even if these checks pass.

The [completed numerical audit](GAUSSIAN_CAMPAIGN_NUMERICAL_AUDIT.md) retains all six iteration/coordinate cases, including the compression-guard failure at128 iterations. Recentered iteration differences grow from137 to839 micrometres. A crossed replay reproduces each coordinate condition bit for bit on the other GPU, so GPU assignment does not explain the233-micrometre coordinate discrepancy at32 iterations in this case. Removing intermediate capture produces the same trajectory bit for bit. The reference still has no accepted operating point; solver settings are being investigated before further fitting.

## Matching-prior reference prototypes

The conservative atlas [prior prototype](CAMPAIGN_REFERENCE_PRIOR_DESIGN.md) excludes some valid supports and has not been adopted. A separate [joint face/UV rejection prior](../experiments/campaign_rejection_prior.py) keeps BC probabilities uniform and samples the full exactly feasible continuous support domain conditional on the complete public attachment geometry. Its geometry-only audit includes all108 noncalibration TRAIN supports. This is a newly designed prior, not the exact conditional law of the seeded simulator generator.

Its unknown normalization depends on BC and public hardware. The [tempered SMC prototype](../experiments/campaign_tempered_reference.py) therefore permits local kernel ratios only within the same normalization stratum; a whole-prior independence proposal cancels the entire prior. Invalid geometry is rejected before strict decoding. Bounded rejection failures abort sampling and are reported with a conservative probability bound; they cannot silently redraw a BC or become null MH moves. Synthetic tests check joint categorical/continuous targets, separated modes and differing category normalization constants. These implementations establish neither physical posterior calibration nor a converged numerical posterior on videos. They will share the same frozen prior and likelihood with a future amortized comparison after numerical reliability and known-force prediction pass their gates.
