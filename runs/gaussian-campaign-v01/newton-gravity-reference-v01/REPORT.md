# Matched P1 gravity equilibrium references

All four declared CPU equilibria passed their original free residual, positive-J, exact clamp and global force/moment balance gates. The physical block is 0.24 x 0.08 x 0.08 m with rho=625 kg/m3, nu=0.35, gravity (0,0,-9.81) m/s2 and the full left face fixed in all axes. There are no rigid handles or external tractions. Newton's stable Neo-Hookean continuum law is used with standard E/nu mapping. Each solve starts from rest.

| E (kPa) | P1 tets | Maximum displacement (mm) | Original free residual (N) | Minimum J |
|---:|---:|---:|---:|---:|
| 300 | 96 | 7.999324 | 1.65e-12 | 0.989504 |
| 300 | 768 | 12.704539 | 4.67e-12 | 0.981694 |
| 600 | 96 | 3.984883 | 2.56e-12 | 0.994772 |
| 600 | 768 | 6.342599 | 4.97e-12 | 0.990896 |

Total mass is 0.96 kg and the gravity resultant is -9.4176 N in z, balanced by the clamp reaction. The physical P1 lumped nodal masses and common 122 material probes are saved for exact-discretization comparisons with Newton GPU readbacks. Native float32 masses, rest poses, Lamé parameters and gravity can differ slightly and must be explicitly audited by the GPU comparison; these CPU solves do not establish native equivalence by themselves.

All four solves took 0.289 s in total with two CPU threads. These are strict fixed-discretization reference equilibria, not a spatially qualified gravity model. In particular, coarse P1 bending remains under-resolved. No temporal, contact, training, inference or prediction claim follows.

Evidence: [protocol](../../operator-dev/newton-gravity-reference-v01/protocol.json), [summary](summary.json), [terminal receipt](../../operator-dev/newton-gravity-reference-v01/terminal.json). All frozen source and local capture hashes matched after completion; session 98178 was reaped with exit 0.
