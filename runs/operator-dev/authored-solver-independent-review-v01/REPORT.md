# Independent ABC/D solver review

No blocking mechanics issue was found for one newly declared fresh-from-rest t0000/twist/120Hz full-trajectory **diagnostic**, conditional on the controls below. No simulation or reserved response was opened by this review; only source, frozen TRAIN failure and completed ABC/D evidence were read. All original source/input pins match and the ABC/D worker PIDs are absent. This is not production adoption or a relabeling of the original failure.

## Objective and reconstruction

The ABC wrapper restores all saved float64 body/auxiliary point-chord positions/velocities, rigid centers/rotations, stable reduced coordinates and clock directly into DynamicState. It does not reinterpret saved chord velocities as rigid tangent velocities. It assembles the original physical model, builds the initial-rest P0 cache, then restores the exact preceding state. A reproduced all251 original history entries and final optimizer q exactly. This is strong evidence that the previously missing rest-first cached metric is reconstructed correctly.

The diagnostic `original_objective` matches the frozen core's finite-pose body/auxiliary inertial terms, log-J elasticity, body-only drag and force work. Zero nodal force is valid for this exact failure, and the failed command has zero handle force. Repeat gradients and recorded objective differences are exactly zero in the saved curvature evidence. Rotation increments use the same left exponential map and old base rotations; the final state resets local rotational coordinates in the original step code.

B, allowing1000 diagnostic iterations, meets the unchanged1e-10N absolute floor at275 iterations. A fails at250 with6.2384e-10N. C rebuilds the cached mass/kinematic projection at the current state but retains the original rest K; it is not a finite-strain Newton Hessian. C still fails at250 with1.8900e-10N. All three final J are about.999985. Thus this evidence supports a slow optimizer tail, not a large-deformation/inversion or reconstructed-state failure. The large printed relative residual at zero force divides by1e-30 and is not the acceptance test; the physical absolute floor is.

## Curvature interpretation

At A's terminal q, the residual-direction Rayleigh quotient of L0^-1 H L0^-T is1.92803762. In a local quadratic gradient iteration, the corresponding signed multiplier is1−lambda≈−.92803762. The observed median norm ratio over the last50 iterations is.928037707 for A and.928038050 for B; all last50 accepted step lengths are1. C has matching ratio.923925577 and Rayleigh about1.923925 under its refreshed metric. Roughly half of these tail accepts invoke the residual-falling roundoff rule.

This is a consistent local explanation for the tail. Scalar residual norms alone do not directly prove a signed vector oscillation, and24-step Lanczos Ritz values/residual estimates are not certified global Hessian eigenvalue bounds. No global convergence, unique minimum or all-state conditioning result follows.

## D change and result

Reviewed D wrapper `experiments/campaign_solver_step_damped.py` SHA87d0f269c217655d63a49183a230cc4033e06ece3e7699fe479ba9afdbc5ce48 binds one private instance method. Its captured original bound method returns the original cached factor; multiplication by sqrt2 creates a new tensor. It does not scale the stored cache in place, repeatedly compound scaling, mutate the core class or affect another instance. Therefore the optimizer metric is2P0 and the computed descent direction is approximately half of the original direction, up to float64 solve roundoff.

The physical objective, physical residual scaling, positive-J line search, cap250 and1e-10N floor are unchanged. This is **not** a claim of exact equivalence to setting the original initial line-search alpha to.5: D retains alpha1, including the alpha-dependent residual-falling roundoff condition.

D converged in69 iterations to9.00146e-11N. Compared with accepted diagnostic B, its maximum q difference is1.42e-13m, point velocity difference1.70e-11m/s and rotation-matrix difference7.47e-13. Objective difference is2.34e-18J, minimum-J difference2.93e-14 and handle/support work differences zero. The predeclared mixed-unit1e-9 state-array agreement is a numerical regression threshold only, not a physical uncertainty norm. Elastic/kinetic differences are reported separately. D's original P0 factor hash equals A/B; its scaled-factor hash differs as intended. The source/protocol pins include every B artifact actually consumed.

D has not completed a full trajectory, and the old250-iteration failed cell remains failed. Tight agreement at this single rest-to-step transition does not prove that a changed optimizer follows the same basin over every nonlinear step.

## Controls required for the next single full trajectory

1. Freeze a NEW wrapper, exact protocol, source/input hashes, audit adapter and fresh output. Declare only t0000/twist/120Hz,3s,360 steps and61 states at20Hz. Start from the exact initial rest/zero velocity state. A restored D step or continuation from the old failure cannot be called a fresh trajectory. Preserve old failure/ABC/D outputs.
2. Keep exact old mesh/material/body drag, both complete ordered handles/mass/full inertia/COM, point fixture, known compiled command arrays, accumulated clocks, gravity/contact assumptions, cap250/relative1e-7/floor1e-10 and original core byte hash. Apply the same instance-only2P0 metric at every step. Build the unchanged unscaled cache at rest and record base/scaled factor hashes; repeated calls must not alter the base factor. No class/global patch, rate/floor/cap fallback, restart or case sweep is included.
3. Reuse the immutable cell verifier through a transparent NEW declaration/provenance adapter, not the old C18 prelaunch seal. Verify full float64 canonical61-frame arrays, every actual interval/held force and compiled metadata, finite positive-J histories, all accepted physical residuals within the unchanged threshold, accepted endpoint J>=.1, fixture/patch/rotation guards, full context roles, independent saved-q J/stretches/physical mass/energy/momentum and F dot stable-center work/moving-origin moments. Save failures and last accepted state without source edits or outcome replacement.
4. Pin the original partial-trajectory.npz, last-accepted-state.npz and240 accepted step ledger. Compare the new first41 frame states through step240 with both the original partial trajectory and the saved last accepted physical state. Parent-predeclared thresholds are maximum position difference<=2e-7m, body/auxiliary point-chord velocity difference<=2.4e-5m/s and.24m times maximum rotation-matrix difference<=2e-7m. Require equal command/actual-clock prefixes and report maxima/RMSE separately. These are optimizer-regression controls against already accepted states, not a reference for the failed endpoint or later tail. Stable reduced coordinates are internal parameterization, while physical poses and point velocities are the primary comparison.
5. Run exactly one bounded CPU worker with2threads, no GPU, and stop/preserve a failure. Independently hash/verify the resulting evidence and worker termination. A complete capture is not automatically a passing audit. Report this one diagnostic separately from the original51-slot matrix; all broader qualifications remain false.

The independent audit adapter is being coordinated with the owner of the new full-trajectory wrapper. No old core, runner, gate, proof or observation loader is edited.

## What is still needed before adoption

A passing one-case diagnostic establishes only that the proposed optimizer can finish this previously failing trajectory under the same discrete equations and acceptance criteria. Adoption needs a prospective numerical revision/provenance policy and explicit reconciliation of the actual-force matrix: all four boundary families, authored force families/rates and matched nonlinear-accuracy/floor, translation and fresh-reset checks at accepted operating points. Do not splice the new endpoint into the old failed cell or infer that an unchanged objective guarantees identical local-minimum selection in every nonlinear case.

Finite rigid rotations, free/point-pin modes and nonzero-COM/full-inertia mechanics tests must continue to pass; the optimizer must remain instance-isolated across step sizes and repeated calls. Record iteration/backtracking/floor-dominance distributions and final physical state/response differences. Any reuse of old successfully converged cells needs an explicit reviewed source-revision/equivalence rule, rather than a silent hash exception. The old failure remains visible under its original solver.

The prospective common-command-clock/matching-prior production admission is a later distinct gate. Neither this optimizer diagnosis nor a full authored-force matrix validates the native Isaac mechanics, a continuum model, contact, an amortized uncertainty posterior or an active policy.
