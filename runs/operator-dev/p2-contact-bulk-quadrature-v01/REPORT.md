# Fixed-state bulk quadrature bias

The original contact model used Duffy3×3×3 (27-point) bulk integration, which is not degree-six exact for this constant-material compatible P2 stable-NH energy. A new CPU-only, hash-bound diagnostic compared that rule with the published positive Keast24 and positive Duffy5×4×4 exact80 rules on the same four completed Q16 fields. No trajectories or GPU calls occurred.

Keast24 and exact80 passed the original 10⁻¹² J / 10⁻¹⁰ N L2 parity gates at all four states: worst differences were 2.36×10⁻¹⁶ J and 9.31×10⁻¹³ N. Q3 differed at the three deformed states by **48.0–128.6 nJ** and **0.269–0.538 mN** in bulk gradient L2 (relative force differences 3.57×10⁻⁶–1.26×10⁻⁵). Before impact the values are at roundoff and relative differences there are not physically meaningful.

Rest geometry, connectivity, gravity forces and exact full consistent mass remained byte-identical. All rules used a common union of 141 original/new integration and nodal guard points (minimum saved J 0.92643), and individual integration minima are reported separately. The finite field comparison took 0.768 s. A manufactured degree-six/compatible-P2/affine-patch test passed in 1.77 s.

A new reference operator should therefore explicitly correct **both bulk quadrature to Keast24 and surface integration to root64**, supported by separate same-state controls. It must not be described as a contact-only change compared with old Q3 trajectories. The old temporal checks remain valid for their original discrete Q3 operator; neither this diagnostic nor quadrature exactness establishes spatial/contact/time convergence or dataset admission.
