# Exact24 versus exact80 fine-field operator parity

The positive Keast 24-point rule and the existing 80-point degree-six rule pass the predeclared operator check at the completed graded48 exact80 equilibrium. No nonlinear or CG solves were performed. Energy differs by **1.73e-18 J**, gradient by **6.68e-15 N maximum** (relative L2 3.09e-12), and the prescribed public smooth full-Hessian action by **1.95e-16 N maximum** (relative L2 3.42e-13). Node-block maximum difference is 4.55e-12 N/m and full consistent mass-action maximum difference 6.62e-24.

Original free residuals are 1.136e-12 N with exact80 and 1.140e-12 N with exact24, both below the unchanged 1e-10 N gate. Minimum J is unchanged at 0.45111845. Exact24 preserves all 80 old integration guard points and the 10 vertex/edge guard points, in addition to its 24 integration points. This is fixed-field implementation parity for constant-material polynomial stable NH on straight P2 elements. It does not admit a new displacement, spatial, temporal, contact or inference dataset.

The bounded GPU2 stage completed in 33.87 s, peak CUDA allocation 1.952 GB. Session 4651 was reaped with exit zero, both owned PIDs were absent and GPU2 was released. Raw operator arrays remain local.
