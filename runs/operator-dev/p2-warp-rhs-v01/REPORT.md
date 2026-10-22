# Fused FP64 P2 force evaluation: same-state checks pass

The independent Warp1.18 prototype reproduces the full compatibleT10 stable-NH and true quadratic surface-contact energy/gradient/Jacobian checks. It preserves every bulk and check quadrature point, six-node face trace, positive weights, and the same frictionless finite-compliance law. It is a force-only backend; no integrator or mass approximation is introduced.

Manufactured CPU checks cover independent standard-form material/force, rigidtranslation, mixedcontact, energygradient/fullJacobian finite differences, float64 type enforcement, and an inversion visible only at retained guard points. The tests first exposed an aliased CPU returnbuffer; the newwrapper now copies its returned gradient and the fixed tests pass. No package was installed or changed.

The bounded GPU1 check used all9 saved outputs of the completed96tet temporal reference plus its accepted state nearest0.102s. Bothactive and inactive contact were present. Maximum forceL2 difference was2.075e-13N, energy difference2.78e-17J and minimumJ difference6.66e-16. Both backends reused the exact same originalTorch fullM Cholesky factor; mass-weighted acceleration difference was2.094e-13N. Four repeated evaluations of every state passed the strict atomic-roundoff envelope.

Five paired timing rounds alternated backend order with equal work. Inclusive median time was2.236ms per evaluation forTorch and0.987ms forWarp: **2.265×** on these states, including transfers, fullmasssolve and outputtelemetry. The6.807s firstWarp JIT/evaluation is reported separately. Controller/worker are reaped and GPU1 released. This is **not** a complete-trajectory speedup or backend qualification; a paired trajectory control is still required before any dynamics switch.

The fresh process appended the pinned isolatedWarp prefix to its own importpath after loading the existingNumPy/Torch. Shared environments and frozen simulation sources remained untouched. Compiledcache/logs/rawstates remain local; compact source/protocol/results are reproducible.
