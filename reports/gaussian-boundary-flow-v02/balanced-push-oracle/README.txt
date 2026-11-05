Balanced finite-difference prefix diagnostic — finished; no model adoption

The new plan was frozen separately before execution. It preserves the Powell
experiment and eight-case benchmark. Same corrected push package, same12prefix
frames/48points, same5logits[-6,6], same3starts/offsets and exactly200forward calls
per law. Explicit finite-difference step0.001 avoids tiny default steps at zero.
Every gradient reserves base plus ALL FIVE perturbations before evaluation.
Residual calls are predeclared full-coordinate local audit probes, not partial
gradients. All source hashes, calls, failures and stops are retained.

Both laws completed33full five-coordinate gradient blocks plus2audit probes.
Every coordinate was explored. All6optimizations stopped at the fixed budget;
none established convergence/global optimality. Bilateral:0failed calls,210.9s
onGPU0. Unilateral:20failed calls,468.3s onGPU1;7gradient blocks include failures
(19failed calls there, plus1failed audit probe). Penalty gradients across failed
solves can be unreliable; this is disclosed rather than treated as convergence.
Both workers are now terminal.

Best attained errors, mm:
                    relative-prefix   absolute-prefix   frame-zero   shape
bilateral                25.537            26.237           5.724     21.795
unilateral               27.010           664.711         663.918     22.125

The unilateral relative-score improvement is NOT a physically acceptable fit:
it exploits a huge initial equilibrium displacement that cancels under frame0
subtraction. Best logits approach bounds[5.999,6,-6,-6,-6], i.e. high gravity/
attachment relative response and weak floor/table. Raw initial shape is roughly
0.664m away from its observed position. Rebased relative motion cannot certify
physical consistency of that raw state. No retrospective rejection criterion
was added; this bad result is reported in full.

Per-start best relative errors:
  bilateral starts0/1/2:25.537,26.050,26.048mm
  unilateral offsets-10/0/+10mm:33.238,27.010,29.799mm
Successful paths' maximum residual:9.95e-8(bilateral),2.07e-7(unilateral), within
unchanged object admission1e-5. Minimum J across successful paths:.3203/.1777.
At the respective best fits, residual8.42e-8/9.48e-8 and J.8852/.8821. A converged
force-balance solve can still be in the wrong physical configuration.

The earlier Powell best was23.408mm bilateral and33.862mm unilateral, but did
not explore width/table. The balanced test fixes that coverage flaw; it does
not supersede or conceal the earlier minima. Different finite-budget searches
find different local points, reinforcing the absence of global-optimum claims.

Qualitative contact evidence
Root inspected public PREFIX camera0 frames0,25,46 and reported that the hand
appears to pinch the nose/face while moving the toy. This is qualitative image
evidence, not ground-truth contact, but it makes an adhesive/held bilateral
hypothesis plausible. A case name containing 'push' is not enough to impose a
nonadhesive plane. The unilateral block withdrawal test remains valid for its
specified mechanics; it does not establish this real video's contact type.

Decisive next experiment
Keep the existing bilateral/grasp hypothesis and restore the old v22-style
ABSOLUTE observed-coordinate prefix objective (or a formally specified joint
initial-state/relative likelihood), with the same observation budget and fully
reported initial mismatch. Freeze this as a separate objective-consistency
ablation before execution; do not silently change current results. It must
prevent improvement through unobserved initial drift, not add another density
network or larger contact search. If a raw force-balanced initial state still
cannot explain the prefix, then diagnose natural shape/initial boundary preload
or material/geometry, rather than recentering away the discrepancy. Only after
raw prefix adequacy is demonstrated should a new inference bank/forecast test
be justified. This is a model-consistency gate, not a request to tune suffix
metrics or force unilateral adoption.

Tests:4new tests pass against the exact frozen gradient block: five-coordinate
coverage, analytic quadratic forward-difference accuracy, no partial gradients
under hard budget/cache, and bounded inward perturbation at+6. No suffix/object
evaluator or new dataset trajectories were accessed.
