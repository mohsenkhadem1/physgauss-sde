# Graded64 Q3 resource-limited attempt

The predeclared 64×32×32-cell (393,216-tetrahedron) Q3 solve from rest reached its **2400 s wall cap** during CG. Worker elapsed time was 2399.72 s; the controller completed termination and reaping in 2401.77 s. Peak observed RSS was 1,563,463,680 bytes, below the 12 GiB guard. The controller records timeout=true; the memory watchdog records reason=null. The traceback location inside a curvature check does **not** establish a negative-curvature or inversion failure.

No converged or accepted intermediate displacement was saved by this old frozen driver. There is therefore **no 64-mesh field, no new spatial comparison and no convergence qualification**. All previous failed spatial checks remain. No physical parameters, quadrature, forces, supports, residual tolerances or live source bytes were changed.

Session 34734 was reaped with exit 1; controller 2226328 and worker 2226400 were absent. GPU0 was released. This is a retained resource-limited attempt, not an undisclosed retry or a failed constitutive-law test. The separately validated new accepted-Newton checkpoint helper is intended for future declared phases and was not inserted into this live run.
