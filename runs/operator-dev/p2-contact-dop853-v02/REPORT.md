# P2 contact: completed adaptive accuracy pair, velocity not qualified

Both fixed-physics integrations completed. The reused rtol1e-7 run and fresh rtol1e-9 run agree in position to **0.154231µm**, passing the 20µm gate. Their maximum all-node velocity difference is **0.515519mm/s**, failing the unchanged 0.1mm/s gate. Thus this pair does **not** qualify the temporal reference.

The maximum velocity difference occurs at t=0.133333s, corner node38, with the largest component in z. Its consistent-mass coordinate RMS difference is29.445µm/s, while the maximum COM velocity difference is1.052µm/s. These supplementary summaries describe a local elastic velocity discrepancy; they do not replace the failed maximum gate.

The tighter run completed14172 accepted steps and316271 RHS calls in757.998s (whole worker760.625s). Accepted-state energy error was15.501nJ, below470.88nJ; momentum balance7.33e-15Ns and mass-solve residual2.02e-13N also passed. Pair output-energy difference422.957nJ passed470.88nJ. These conservation checks do not independently certify temporal accuracy.

The model remains96 P2 tetrahedra, exact consistent mass, E300kPa, rho625kg/m³, nu0.35, gravity9.81m/s², initial gap0.05m, frictionless undamped finite compliant plane kappa1e9N/m³, true quadratic boundary Q8. The analytic precontact initializer and fixed120Hz output times are unchanged. No rigid-floor or spatial/quadrature qualification is claimed.

The manual public SciPy step loop retains the identical DOP853 method while avoiding dense interpolation except at requested queries. Every accepted state receives energy/momentum checks. Atomic512-step snapshots preserve the complete committed prefix; a hard kill can lose up to511 uncommitted tail states. Controller and worker are reaped. All raw states/checkpoints stay local. The earlier900s capped phase remains preserved.
