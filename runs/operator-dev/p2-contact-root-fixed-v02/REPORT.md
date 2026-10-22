# Fixed outer32→64: energy/force pass, stiffness gate fails

The new CPU-only phase completed all four hash-bound saved states in 4.293 s, using the original successful adaptive reference without waiving solver status. The earlier tenfold-tighter adaptive failure remains preserved and was not reused as truth. Both rules retain the same root splits, finite plane law and reference metric.

All energy and force comparisons passed the original 10⁻¹² J / 10⁻¹⁰ N L2 gates. Worst fixed32→64 differences were 3.82×10⁻¹⁷ J and 2.291×10⁻¹² N. Fixed64 agreed with the original adaptive reference for energy, force and full stiffness at every state; worst stiffness entry difference was 8.73×10⁻¹⁰ N/m.

The aggregate nevertheless **fails**: at the accepted state nearest 0.102 s, fixed32→64 stiffness differs by 7.649×10⁻⁷ N/m, exceeding the unchanged 10⁻⁷ N/m gate. Fixed32 versus adaptive fails the same stiffness gate. The other states passed. No threshold was relaxed, no extra quadrature order or GPU trajectory was launched, and no contact/dataset qualification is claimed. Fixed-rule error estimates are explicitly unavailable (`null`); actual inter-rule changes establish these checks.
