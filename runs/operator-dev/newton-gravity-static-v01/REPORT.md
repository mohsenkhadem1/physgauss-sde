# Newton gravity equilibrium: matched discrete validation

All five fixed-law/static numerical solves passed the prospective original-force, stationarity, clamp, positive-J and 20 µm CPU comparison gates. Four 512-iteration cases span E=300/600 kPa and 96/768 tetrahedra; their maximum CPU position differences were 0.120, 2.023, 0.180 and 2.105 µm. A separate 128-iteration coarse control differed from 512 iterations by 1.291 µm.

The largest original elastic-plus-gravity residual was 2.202 mN, below the declared 10 mN threshold (total weight 9.418 N). Native quantized-rest force floors were 0.566–1.344 mN; they are explicitly reported rather than asking FP32 to meet CPU float64 residuals. Fixed nodes remained exact and minimum J exceeded 0.989.

This is zero-velocity proximal STATIC relaxation with a one-second pseudo step and the previously verified accepted-initial-iterate repair. It is not a physical gravity transient. Comparison uses the identical P1 mesh and installed stable Neo-Hookean material, not a continuum-converged reference. All five starts were fresh rest states; physical damping and contacts were zero. Native masses, inverse masses, Lamé arrays, rest poses and gravity were read back. Both owned processes terminated; GPU1 was used exclusively. Raw arrays, detailed histories and logs stay local.
