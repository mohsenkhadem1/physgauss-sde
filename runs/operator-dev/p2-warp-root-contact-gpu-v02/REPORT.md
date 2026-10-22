# Root64 contact force/energy: GPU same-state implementation control

The new standalone root-split surface operator passed GPU1 versus CPU on four frozen 96-tet states, with no new dynamics. Maximum gradient L2 difference was 3.946×10⁻¹⁴ N and energy difference 1.735×10⁻¹⁸ J, below the original 10⁻¹⁰ N / 10⁻¹² J gates. Repeated GPU atomic sums varied by at most 3.923×10⁻¹⁴ N, with identical energies. No production Hessian is supplied.

Five rounds alternated CPU/GPU order on the same four states. Mean complete `.evaluate()` cost, including input transfer and energy/gradient output, was 2.990 ms on CPU and 0.5872 ms on GPU (5.093× for this interface). First CPU/GPU JIT took 21.900/61.273 s and is separately recorded/excluded. This is not a full-RHS, nonlinear-solver or trajectory speed claim. No old Duffy kernel timing comparison was made because its contact approximation differs.

The controller completed/reaped in 87.63 s under the 180 s cap; both owned PIDs are absent and GPU1 released. CPU30/31 is disjoint from the other live jobs. All resource checks passed. The unexecuted v01 CPU34/35 preparation was superseded explicitly; no v01 GPU call occurred.

The finite reference-area penalty law and FP64 weights remain unchanged; root boundaries and integration points depend on geometry. CPU finite-difference sweeps control energy/force agreement numerically, so later accepted-state trajectory energy checks remain essential. The earlier Q8→Q16 field/velocity failure and fixed32→64 stiffness failure are preserved. No surface/time/spatial qualification, hard-floor limit, trajectory or drop-data admission follows from this implementation control.
