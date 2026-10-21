# Separate conditional 960 Hz evidence audit

`experiments/campaign_dynamic_extension_audit.py` is a new read-only auditor. The published initial audit, failed 480 Hz temporal objects, physical core and live runner were not edited. No simulations or reserved-data reads were performed.

The auditor pins the published base-auditor revision, exact frozen baseline manifest, revised conditional runner, temporal helper, temporal declaration, unchanged core/static fixture/parent gate, and immutable failed-trigger audit. It delegates complete-cell clocks, force/impulse, source/input snapshots, residuals, positive volumes, geometry and work bookkeeping to the frozen base auditor. For each new cell it additionally requires the exact conditional metadata/provenance and independent held-force work reconstructed from complete stable per-step handle-center increments. The unchanged mechanics prerequisite must pass.

It reopens only the six declared TRAIN t0003 temporal evidence cells and, when available, the separate two-cell conditional manifest. It recomputes initial120/240/480 full-response metrics and requires agreement with the immutable failed comparisons. The output retains those original objects and `initial_moving_temporal_gate_passed=false` even if an extension later passes.

The extension output lists all four actual rates and all three adjacent full-coordinate trajectory RMSEs, both measured contraction flags/ratios, and the 960 Hz diagonal-minus-zero force signal. Zero subtraction determines only the common `max(2 micrometres, 5% force signal)` tolerance. The final 480-to-960 difference must meet tolerance and contract from the preceding difference, except for the original two-micrometre floor exception. No coarse rate or failed criterion is hidden.

Two valid passing extension cells can produce only `conditional_960_temporal_candidate_passed=true`. Matched operating-point checks, moving960 reference qualification, overall independent reference qualification and native qualification remain false. Existing 480 Hz accuracy/translation/replay/floor checks retain their 480 Hz scope. The separate published 480 Hz moving-zero absolute-floor study remains required evidence.

## Conditional manifest interface

Use a separate JSON with these exact top-level fields:

```json
{
  "parent_gate_sha256": "962457955d72a97dd0d8e48a41aff0c6ead821931031343e081cf837c561b2a0",
  "temporal_protocol_sha256": "e6c9efb31c34ea1b6ffe476da192edecad3da5697c0cebaf185ddc5d8813df6f",
  "baseline_manifest_sha256": "688f89e1de05e98ce91783e1d53c1ad775fa5630c607c288c1ecd444f93c56db",
  "trigger_audit_sha256": "3733850713231a5d750a74e8d454475fe30f579f6b4e638699a53e6268872f43",
  "approved_runner_source_sha256": ["a0fefae860eef4f7442c019045a72b2544d7708bd3fc47e737a7b3a649b0d87f"],
  "temporal_dependency_sha256": {
    "campaign_dynamic_temporal_protocol.py": "dd9e43aa0bc8e570133fc21b6d54e1a85e678fb395bba660e6ab544ee43ea64a"
  },
  "cells": []
}
```

The two cells must each have case `t0003`, kind `temporal_extension`, hz960, relative_tolerance1e-7, absolute_tolerance_N1e-10, translation_m[0,0,0], one of the two declared responses, and distinct fresh `output` paths. Missing evidence remains incomplete. Unapproved source revisions or slots are rejected rather than silently accepted. A future runner revision requires an explicitly reviewed audit revision; this version does not accept an arbitrary manifest whitelist.

Run with `--baseline-manifest`, `--temporal-manifest` and a NEW `--output` path. Omitting the temporal manifest reports pending conditional evidence. Never overwrite the published original/trigger audit.

## Minimal matched 960 Hz follow-up proposal

`configs/gaussian_campaign_dynamic_960_validation_v01.json` is explicitly `PROPOSED_ONLY_NOT_EXECUTION_AUTHORIZATION`. After both temporal candidates pass, parent approval/publication and a separately reviewed exact-request guard would be required before these eight new cells:

| Check | New cells | Reuse |
|---|---:|---|
| Diagonal relative accuracy | 1e-5 and1e-9 | Temporal1e-7 baseline |
| Moving-zero effective accuracy | Absolute floors1e-8 and1e-12 N | Temporal1e-10 N baseline |
| Coordinate translation | Zero and diagonal at[0,0,-.25] m | Original temporal pair |
| Fresh reset | Zero and diagonal | Original temporal pair |

This is ten operating-point cells including two reused temporal responses. Each runs the same declared t0003 mechanics, 960 Hz, three seconds/31frames and250-iteration cap. The proposal uses the 960 Hz signal for its trajectory tolerance, reports both adjacent accuracy differences/effective floors, and uses a separate 1e-12 numeric reset threshold with bitwise diagnostics. The original480 absolute-floor study is not replaced by these proposed960 cells. No current request guard authorizes this follow-up matrix.

Passing this factorized follow-up would still cover only the declared fixed discretization and .01 N diagonal/zero envelope. It would not establish continuum/native equivalence, exhaustive refinement interactions, original0.12–.24 N recipe amplitudes, trustworthy inference or improved active probing.

## Validation and pending real evidence

85 tests passed in 2.77 s with two CPU threads: 30 new extension-auditor tests plus the existing 15 base-auditor, 27 temporal-request and 13 floor-request tests. New tests cover exact slot/source/role rejection, stale or reused output paths, immutable initial decisions, full-response cancellation traps, four-level metrics and candidate-only output. Orchestration tests stub the separately tested complete-cell verifier and use tiny analytic arrays; they are not physics evidence.

The read-only pending report `runs/operator-dev/independent-dynamic-temporal-extension-audit-pending-v01.json` verifies all six real original moving-case temporal cells, recomputes both initial failed comparisons, and passes the31-test mechanics prerequisite. The conditional manifest is missing, so temporal candidate and all reference qualifications remain false. No optional960 trajectory has been generated by this task.
