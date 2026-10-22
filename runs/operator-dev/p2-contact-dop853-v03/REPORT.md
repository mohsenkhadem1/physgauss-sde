# Fixed-mesh compliant contact: adaptive temporal reference passes

The completed rtol1e-9 and fresh rtol1e-10 integrations agree to **4.292nm** at all81 fixed material queries and **0.024753mm/s** across all nodal velocities. These pass the unchanged20µm and0.1mm/s gates. Output-energy difference13.197nJ passes470.88nJ. The previous1e-7/1e-9 velocity failure remains preserved.

The fresh integration completed25,606 accepted steps and708,875 force evaluations, with1709.66s whole-worker runtime (1711.02s controller). All accepted states and9 fixed120Hz outputs were checked: energy error2.610nJ, momentum error4.68e-15Ns, mass-solve residual2.15e-13N, and minimum stage Jacobian0.925322. The complete committed prefix and raw arrays remain local; controller and worker are reaped.

The model and physics are unchanged:96 compatible P2 tetrahedra, exact full consistent mass, stable Neo-Hookean E300kPa/nu0.35/rho625kg/m³, gravity9.81m/s²,0.05m initial bottom gap, no friction/damping, static finite-compliance plane kappa1e9N/m³, positive Q8 quadrature on every true quadratic boundary face. Both runs start from the same exact precontact freefall state at1/12s and end at0.15s.

This passes the declared numerical temporal check for the **fixed mesh/Q/contact law**. It is not a rigorous exact-error bound, continuum spatial qualification, surface-quadrature qualification, hard nonpenetration, or drop-data admission. Momentum uses the same RK quadrature as the auxiliary force integral, so is an algebraic consistency check. Surface and mesh refinement remain necessary.
