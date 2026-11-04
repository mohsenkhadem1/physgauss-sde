Matched prefix-only bilateral/unilateral oracle — completed bounded diagnostic

Frozen plan before execution; corrected single_push_sloth package, same12prefix
frames/48points, five response logits bounded[-6,6], identical geometry/table/
uniform material and relative displacement objective. Each law used exactly200
forward calls split67/67/66 across three predetermined starts/hypotheses. The
unilateral offsets were fixed at[-10,0,+10]mm; normals from the previously frozen
prefix-command-axis hypothesis. No neural training, suffix scoring or future
object data was used. Existing eight-case conclusions remain unchanged.

Best attained prefix errors (mm):
                 relative objective   absolute coordinates   aligned shape
bilateral             23.408               24.822                 19.447
unilateral            33.862               34.972                 21.368

Hypothesis results: initial relative -> best relative; initial frame-zero error
bilateral start0: 45.280 ->25.738; initial raw frame0 6.073mm
bilateral start1: 41.556 ->23.408; initial raw frame0 2.955mm
bilateral start2: 30.608 ->27.143; initial raw frame0 14.981mm
unilateral -10mm:35.520 ->35.003; initial raw frame0 8.388mm
unilateral   0mm:34.912 ->34.512; initial raw frame0 8.388mm
unilateral +10mm:34.685 ->33.862; initial raw frame0 8.388mm

The selected best fits STILL have raw frame-zero mismatch11.656mm(bilateral)
and10.763mm(unilateral). Relative displacement fitting does not solve unknown
initial equilibrium/prestress. Do not hide this mismatch through rebasing.

All six Powell runs reached the declared function-call cap, not convergence.
An important optimization limitation was detected after execution: the ordered
coordinate searches never reached Gaussian-width or tangential-table logits;
these last two coordinates remained zero in EVERY evaluated call. The nominal
five-dimensional parameterization was thus not fully explored. This is a fair
reported equal-call bounded attempt, not a global optimum or decisive rejection
of unilateral contact. There is no basis to adopt this unilateral family from
these results. A more balanced five-coordinate search would require a separately
frozen supplemental diagnostic; no extra calls were silently added.

Failures preserved: bilateral45/200 and unilateral34/200. Failed calls count
against the same budget and receive the predeclared finite penalty; no selection
replacement. Successful full-prefix paths: maximum residual9.992e-6/9.500e-6,
minimum J0.1862/0.2379 (bilateral/unilateral). These satisfy the unchanged OBJECT
backend admission criterion1e-5; this is distinct from the stricter synthetic
block qualification gate. At the respective best fits, residual8.999e-8/1.221e-6
and minimum J0.3713/0.4151. All paths solve every command frame0..46.

GPU runtime245.5s(bilateral GPU0),291.4s(unilateral GPU1), running concurrently.
Both workers completed and GPUs are released. Every call, its parameters, error
or solver failure, and optimizer stop reason is retained in *-results.json and
per-hypothesis JSONs. summary.json includes visited coordinate ranges. No new unit
tests were added for the runner; the isolated operator/backend tests remain as
previously reported.

Interpretation: the current simple plane hypothesis has not demonstrated better
real prefix adequacy. Known synthetic release correctness alone does not identify
real tool geometry or solve real shape mismatch. Preserve both this bounded
negative result and the synthetic success; neither is a real forecast claim.
