# Physical-time reference and a verified numerical repair

The fixed 96-tetrahedron stable-Neo-Hookean block now passes the declared **temporal** refinement check with BDF2: the final 3,840→7,680 Hz maximum query difference is 1.819 µm, below 20 µm. All original implicit force, Jacobian, and exact-boundary checks pass. This supplies a physical-time numerical reference for a same-mesh Newton comparison. It does not qualify the spatial mesh, a GPU solver, contact, or real-world prediction.

The experiment uses exactly the matching Newton material law, 240 × 80 × 80 mm geometry, E = 5 kPa, ν = 0.35, density 625 kg/m³, physical P1 tetrahedron-lumped masses (total 0.96 kg), a fully fixed left face, zero gravity and damping, and a known uniform right-face traction stepping to 1 N at t = 0. Every trajectory starts independently at rest. The duration is 0.1 s, with all physical states saved and 25 common output times at 240 Hz. Force impulse is 0.1 Ns. No force, material, support, or physical damping was retuned.

## Unchanged physical equations

Backward Euler solves E(q) + ½Σ M/h² ||q−qn−hvn||² − F·q and sets vn+1 = (qn+1−qn)/h.

BDF2 uses one backward-Euler startup step. Later steps use h_eff = 2h/3 and target y = (4qn−qn−1)/3 + (2h/9)(4vn−vn−1), minimizing E(q) + ½Σ M/h_eff² ||q−y||² − F·q. The physical velocity is (3qn+1−4qn+qn−1)/(2h). This follows by eliminating velocity from the two first-order BDF2 equations. It is physical time, unlike the earlier h = 1 s static proximal optimizer. No artificial damping is introduced; the integrators' numerical dissipation differs.

Each accepted step must have original objective free-force residual ≤10^-10 N, independently reconstructed physical BE/BDF2 residual ≤10^-9 N, J > 0.1, finite state, and exactly fixed support displacement/velocity. The full unmodified elastic tangent plus the physical inertial diagonal is solved by sparse Newton with positive-J Armijo acceptance. No stiffness projection or regularization is used.

## Refinement results, including failures

All errors below are maximum Euclidean position differences over **all 25 common times and 122 fixed volume/surface/edge/corner queries**. Earlier failures are preserved.

| Adjacent rates, Hz | Backward Euler | BDF2 |
|---|---:|---:|
| 240→480 | 175.799 µm | 138.807 µm |
| 480→960 | 110.161 µm | 76.513 µm |
| 960→1,920 | 72.388 µm | 27.695 µm |
| 1,920→3,840 | Not run | 7.321 µm |
| 3,840→7,680 | Not run | **1.819 µm** |

Both original four-rate methods failed the 20 µm finest-adjacent gate. BDF2 improved the result but was not then qualified. The separately declared finer BDF2 continuation retains a monotonically decreasing sequence of all five errors. The final pair changes by about a factor of four, consistent with second-order behavior. An adjacent-difference check is an engineering temporal convergence indicator, not a mathematical bound on the exact continuum solution.

## Numerical failures and their bounded repair

The first 7,680 Hz run failed its unchanged 10^-10 N objective threshold: world-coordinate subtraction stagnated at 1.398 × 10^-10 N. A separately frozen displacement/step-increment representation reproduced the successful 3,840 Hz field to 4.62 × 10^-14 m, but its 7,680 Hz run still stalled at 6.50 × 10^-10 N while Armijo reduced the step size. Both failures and their residual histories remain saved. Their thresholds were not relaxed.

The final, separately frozen repair retains displacement u, physical velocity, and accepted step increments internally. It computes compatible F = I + Σu_a⊗∇N_a directly and calls the **same frozen stable-material energy, stress, and full tangent**. The Newton variable is the step increment z, so the inertial difference avoids subtracting global positions. These are numerical reparameterizations of the same equations.

Armijo's finite-energy roundoff branch uses the previously validated static solver scale 128ε max(E_young V, |E|, |potential|). Here E_young V = 7.68 J and the material-scale window is 2.183 × 10^-13 J. That branch additionally requires strictly decreasing original residual. It does not change the objective, force tolerance, or final stationarity requirement. Direct displacement assembly is checked against the old energy/force/full tangent at moderate deformation and by finite differences before the new trajectories run.

The final 3,840 and 7,680 Hz trajectories both pass. Their maximum original objective residual is below 1.95 × 10^-14 N; maximum physical residual is below 5.40 × 10^-14 N. Minimum J is 0.99548/0.99548 and all support defects are zero. The 3,840 Hz field again matches the frozen earlier success within 4.62 × 10^-14 m, far below its prospective 10^-8 m parity threshold. Thus the repair changes arithmetic, not the simulated response or physical parameters.

Global-position output necessarily rounds rest + u. Maximum displacement roundtrip error is 1.39 × 10^-17 m, below the separately declared 10^-14 m guard. Reconstructing BDF2 acceleration from the saved global positions produces maximum force residual 2.09 × 10^-10 N at 3,840 Hz and 8.47 × 10^-10 N at 7,680 Hz. These readback diagnostics were prospectively separated from stationarity in the retained increment representation. Raw local outputs include displacements, increments, physical velocities, and independently reconstructed world-chord velocities so this distinction is inspectable.

## Validation, resources, and scope

Three base tests passed in 1.79 s: a mass/known-force backward-Euler impulse/position/work oracle; an independent 2×2 first-order oscillator BDF2 discrete oracle; and nonlinear original momentum residual/constraint checks. Two relative-representation tests passed in 1.69 s. Two final repair tests passed in 1.74 s, including moderate-deformation energy/force/full-Hessian parity and finite differences. An initial unit-test sparse-format error was corrected before benchmark launch; its log is retained.

All jobs used two CPU threads. The original eight trajectories completed in 6.28 s; the first finer continuation completed in 3.41 s with the 7,680 Hz failure; relative v02 completed in 3.79 s with its retained failure; final v03 completed both trajectories in 9.73 s. Every stage had a frozen pre-run protocol and bounded wall budget. All workers were reaped. No GPU call or new finite physics beyond those declared trajectories was performed by this worker.

The 96-tetrahedron mesh still has unresolved spatial error. Newton GPU validation must compare the same physical method/rate/lumped masses/known force and the same material law, inspect original native and independent momentum residuals, and retain its float32 quantization floor. This CPU temporal success does not transfer the earlier C36 log-J static qualification to a different law or validate gravity, dropping/contact, uncertain boundaries/material inference, RGB forecasting, probing, or calibration. No inverse or learned model was fitted.

See [final compact results](compact-summary.json), [v03 protocol](../../operator-dev/stable-transient-relative-v03/protocol.json), and [terminal receipt](../../operator-dev/stable-transient-relative-v03/terminal.json). Earlier compact records preserve the [base comparison](../stable-transient-v01/compact-summary.json), [first finer failure](../stable-transient-refinement-v01/compact-summary.json), and [relative v02 failure](../stable-transient-relative-v02/compact-summary.json). Raw trajectories and complete per-step ledgers remain local.
