# Compatible P2 contact: independent controls pass, first drop screen unresolved

GPU1 float64 contact force agrees with independent exact barycentric polynomial integration plus CPU stable-NH elasticity to 1.77e-13 N; the manufactured solution differs by 3.17e-14 m. Full-consistent-mass freefall agrees with the independent scalar BE-start/BDF2 reference to 9.73e-15 m.

Five complete drop cells retain the same .24×.08×.08 m domain, E300 kPa, ν=.35, ρ625 kg/m³, g=−9.81z and 5 cm initial gap. The model uses true six-node P2 surface traces, positive reference-area Duffy quadrature, κ=10⁹ N/m³, zero friction/damping and full T10 mass. It is a compliant plane, not a rigid nonpenetrating floor. Every complete cell satisfies original physical R<1e-9 N and objective R<1e-10 N.

The common-120Hz/81-material-query differences are 1.655 mm (240→480 Hz), 4.120 mm (480→960 Hz), 1.774 mm (6→96 tetrahedra) and 0.486 µm (surface Q8→Q16). Only the surface quadrature comparison passes the 20 µm screen. The time steps are not yet in a convergent regime around impact; no drop-data qualification is claimed.

The κ=10¹⁰ cell stopped at Newton iteration25 with original residual 1.09775e-10 N, above the unchanged 1e-10 N criterion. Its last accepted state is saved. The planned E600 kPa cell was not launched after that failure. All prior native BE/contact failures remain unchanged.

The compact summary additionally reports full-consistent-mass discrete BDF2 impulse bookkeeping and mechanical energy. The telescoped BE-start/BDF2 impulse is (3P_N−P_N−1−P_1−P_0)/2, not ordinary rectangle-quadrature ΔP. No unconditional per-step mechanical-energy monotonicity is claimed for switching contact.

Worker and owned controller terminated after89.46 s; raw trajectories, per-step ledgers and checkpoints stay local. Higher-rate refinement uses a separately frozen exact analytic freefall initializer and cannot relabel these BE-start outcomes.
