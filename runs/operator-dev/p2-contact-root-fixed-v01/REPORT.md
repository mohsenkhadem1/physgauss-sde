# Fixed outer contact controls: preserved partial failure

A separate CPU-only protocol tested topology-split outer Gauss16/32 with the endpoint mapping r=lo+(hi−lo)sin²(πt/2) and its Jacobian. The same exact polynomial inner integration and finite reference-area plane law remain. Manufactured constant/affine/quadratic, tangency, near-linear and PSD controls passed (one test, 0.67 s).

At saved t=0.108333 s, tightening all adaptive component tolerances tenfold changed energy by 2.71×10⁻²⁰ J, global force L2 by 1.22×10⁻¹⁵ N and maximum stiffness entry by 5.82×10⁻¹¹ N/m. Fixed32 agreed with that refined adaptive result within 1.08×10⁻¹⁹ J, 4.78×10⁻¹⁵ N and 2.04×10⁻¹⁰ N/m. However, fixed16→32 failed the frozen stricter controls: force 6.08×10⁻¹⁰ N exceeded 10⁻¹⁰ N and stiffness 8.73×10⁻⁵ N/m exceeded 10⁻⁷ N/m; energy 9.55×10⁻¹⁶ J passed.

At the second impact state (nearest 0.102 s), tenfold tighter adaptive integration reported roundoff status2 and stopped the diagnostic after 3.607 s. Its estimated component-error ratio was 0.365<1 but the helper additionally requires solver success at its internal 0.2 oversolve target. No result was silently accepted or tolerance relaxed; the entire partial failure remains. No fixed32 claim is made for this unfinished state.

The fixed-rule result's initialized `estimated_errors` zeros mean **not available**, not zero error. Only adaptive estimates and the explicit inter-rule differences are meaningful. This receipt clarifies the frozen result without changing its bytes. No GPU, contact trajectory, hard-floor, or data qualification follows from these CPU controls.
