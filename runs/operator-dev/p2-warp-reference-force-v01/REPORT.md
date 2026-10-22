# Corrected combined reference backend: CPU validation

The new backend explicitly corrects two integrations: constant-material compatible-P2 bulk energy/force uses degree-six-exact Keast24, and finite compliant plane contact uses root-split outer64 surface integration. It reuses frozen bulk and Jacobian kernels, adds the independently tested root64 kernel, and never launches/subtracts the old Duffy contact kernel. Rest geometry, body force, exact full mass, material and κ remain unchanged. All 141 original/new Jacobian check points are retained.

The manufactured expansion/compression, energy/force, full-mass gravity, mass residual and inversion-rejection test passed (2.36 s). On four hash-bound saved states, combined WarpCPU versus independent NumPy Keast24 plus adaptive plane passed original gates: maximum energy difference 5.55×10⁻¹⁷ J, gradient L2 2.21×10⁻¹³ N, Jacobian 7.77×10⁻¹⁶, full-mass residual 5.97×10⁻¹⁴ N. Torch versus SciPy full-RHS acceleration differed by at most 2.07×10⁻¹⁰ m/s². The finite check took 3.065 s.

An earlier manufactured test exposed a reporting bug in the independent reference helper: a minimum initialized at one capped expanding-state J. The actual backend reported 1.0185126 correctly. That failed test, old source and unexecuted CPU protocol remain preserved. The new independent reference starts its actual minimum at infinity; the physical J>0.1 guard is unchanged. All previous four saved states had J≤1, so their old reported minima remain valid.

This phase made no GPU or trajectory calls and supplies no production Hessian. Subsequent DOP trajectories require new source/protocol review and fresh time/surface/spatial controls. Earlier Q3/Duffy results retain their original discrete-operator scope.
