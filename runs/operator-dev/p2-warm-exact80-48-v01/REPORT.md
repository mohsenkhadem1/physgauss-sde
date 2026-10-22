# Exact80 equilibrium correction at the graded48 mesh

The predeclared warm-start quadrature continuation passed in **652.26 s**. It starts from the immutable completed Q3 displacement, uses the same physical domain, material, force, clamp and geometry, and changes bulk quadrature from Q3 to the positive degree-six-exact 80-point rule. It makes no independent basin-of-attraction claim.

One Newton step, with 277 original-operator CG iterations and no regularization, reduces free residual from **4.8303e-8 N to 1.0629e-12 N**. Force balance defect is 1.0223e-12 N, actual moment defect 4.31e-14 N m and clamp error zero. Minimum sampled J is 0.45111845; the whole-element Bernstein lower bound is 0.45111845, while the older conservative norm bound remains 0.00917405.

Q3-to-Q80 maximum position change is **1.204e-9 m over all nodes**, and **1.027e-10 m over the original 63 fixed queries** (0.0001027 µm), passing the unchanged 20 µm quadrature query gate. This does not establish spatial convergence: all earlier failed spatial checks remain. Maximum strain is 61.61%, volume-weighted 99th-percentile strain 6.135%, and nonrigid coordinate RMS 1.477 mm; the original 2–8% maximum-strain excitation goal remains unmet.

Peak CUDA allocation was 2.003 GB. The one focused independent CPU test passed in 3.15 s: streamed exact80 continuation agrees with a monolithic exact80 solve initialized from rest on a small control, while preserving the supplied Q3 field. Session 95014 was reaped with exit zero, both owned PIDs were absent, and GPU2 was released to the root-owned gravity stage. No current model, dynamics, contact, inference or dataset admission is claimed. Raw fields remain local.
