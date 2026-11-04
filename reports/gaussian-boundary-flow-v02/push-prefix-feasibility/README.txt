Prefix-only real-push plane feasibility (no future-object evaluation)

Implemented isolated PrefixPlanePushBackend. It retains the corrected input package, initial filtered geometry, uniform material, gravity, Gaussian table term, response-ratio prior and full-command quasistatic continuation. Only the single tool attachment operator becomes unilateral. The old bilateral backend and frozen eight-case campaign are untouched.

A straight-approach plane hypothesis is reproducibly constructed from the SAME 12 selected prefix controller-centroid positions. The sign follows net prefix motion and must point toward the initial object centroid. The estimator rejects small displacement, poorly linear paths or outward initial approach instead of guessing. No object observations beyond the permitted prefix, contact/force labels or true surface normals are used. The 48 selected object points are used only for prefix discrepancy reporting.

This case has an identifiable COMMAND AXIS: net displacement195.89mm, first PCA component99.907%, net-axis alignment0.999961, initial inward cosine0.982925. Normal=[-0.173994,-0.984712,0.008237]. This does NOT establish that the command axis equals the hand surface normal. Hand/controller points need not lie on the actual contact surface; fixed standoff hypotheses of -10,0,+10mm along that normal are explicitly declared. No offset was selected as a winner or fitted.

Frozen feasibility: three standoffs, shared median-prior logits z=0, CPU, prefix frames0..46 only (12 observations, all47 command states solved). All three complete with maximum residual<=8.64e-8 and minimum J>=0.4696.

Standoff(mm)   Absolute prefix RMSE(mm)   Relative prefix RMSE(mm)
-10             36.182                    35.520
  0             35.506                    34.912
+10             34.919                    34.685

Errors remain substantial. This establishes a feasible numerical hypothesis, not a fitted or improved real model. No neural training, suffix scoring, contact-discovery claim or posterior-calibration claim was made. Three tests verify reproducible prefix-only direction, future prescribed action affecting only future plane locations, standoff convention, degenerate/outward rejection and forbidden target fields.

Sources: push_backend.py; push_prefix_feasibility.py; frozen plan.json and results.json in this directory. Plane/geometry assumptions remain the main limitations. A subsequent matched prefix-only bilateral/unilateral oracle would be required to separate the boundary-law benefit from parameter choice and plane standoff before any new inference bank is justified.
