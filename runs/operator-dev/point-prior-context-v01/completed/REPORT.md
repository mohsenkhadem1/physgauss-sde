# Posthoc rest/prior context on the sealed 32-object test

This is a separately frozen descriptive context comparison, not a new preregistered primary endpoint. Source/design/archive hashes froze BEFORE any reserved target values were opened. The original C26 prior-generation source, exact H0 hardware and all six known-control arrays were verified. Existing observation-independent prior means use p(mask)/Nmask and 2048 systematic samples/seed20261118; old case posterior likelihoods/weights are not reused. No forward, inference or physical queries were made.

All 32 complete held-out objects pass the final aggregate/release/state/source gate. Targets are A31..60 and B0..60 for every comparison. All objects remain included.

| Predictor | A pooled RMSE | B pooled RMSE | A mean-object RMSE | B mean-object RMSE |
|---|---:|---:|---:|---:|
| Static rest |2.14105mm|1.33361mm|1.85438mm|1.16718mm|
| Preexisting C26 prior mean |1.62705mm|1.03887mm|1.55834mm|0.99958mm|
| Sealed no-extra-view posterior |0.67490mm|0.46295mm|0.55687mm|0.38583mm|
| Sealed active-view posterior |0.31009mm|0.22411mm|0.25490mm|0.18476mm|

The context prior-minus-no-view mean object RMSE difference is 1.00147 mm for A, descriptive paired 95% bootstrap interval [0.83371,1.17445], and 0.61375 mm for B,[0.52127,0.70888]. Prior-minus-active differences are 1.30344 mm A,[1.14385,1.47070], and 0.81482 mm B,[0.71752,0.91348]. Positive means the context baseline is worse. Static-minus-policy intervals also exclude zero. Full MSE differences and all 32 object contributions are retained in summary.json. Bootstrap unit is the complete object, with 10,000 replicates/seed2026118000; frames and rays are not independent units.

These controls confirm an observation-conditioning benefit relative to rest/prior prediction. They do not distinguish intelligent view selection from a simple extra view: the sealed primary active-versus-random/fixed intervals include zero. Neither finite-model posterior calibration nor real-world/unknown-BC performance follows. The prior archives are the preexisting C26 evaluator outputs first hash-bound for this posthoc diagnostic; no regeneration or held-score tuning occurred. Raw trajectories, prior NPZs and posterior state arrays remain LOCAL.

Frozen design: ../design.json. Final sealed aggregate SHA256: 51fcbd0e93e92e2a6f80ab996c45c30b21e73c57c797cd436bca4549b733240d.
