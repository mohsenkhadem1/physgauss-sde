# Geometry and surface integration diagnosis

This CPU2 geometry-only audit reads no trajectories, observations or latent labels and performs no dynamical solves. Both original and new mesh families preserve the same 0.0005 m³ domain, original six boundary-face areas/planes, positive volumes, shared-face opposition, closed boundary edge incidence, one boundary component and Euler characteristic two. No duplicate coordinates or unmatched internal boundary facets were found.

The fixed central-octahedron diagonal is a concrete quality problem, not a topological hole: repeated eightfold cell-count refinement does not imply halving of mesh size.

| Tetrahedra | Old / new maximum edge, mm | Old / new minimum mean-ratio quality |
|---:|---:|---:|
|2|141.42 / 141.42|0.83995 / 0.83995|
|16|100.00 / 100.00|0.68723 / 0.68723|
|128|61.24 / 50.00|0.50397 / 0.68723|
|1024|39.53 / 25.00|0.32868 / 0.68723|
|8192|26.52 / 12.50|0.19383 / 0.68723|

At8192 cells, old maximum rest-edge-matrix condition is25.96 and minimum radius ratio3r_in/R_circum is0.04551; the new shortest-diagonal family gives5.83 and0.62299. These ordering-independent quality measures confirm that condition changes are not just vertex numbering. The new refiner uses globally shared edge midpoints and64eps scale-relative geometric tie handling with invariant vertex-ID ties. Four tests pass, including passive rotation/translation/uniform-scale connectivity, positive volume/mass, surface conformity, and affine deformation/energy invariance. Original sources remain unchanged. Geometry improvement does not establish dynamical convergence.

## Gaussian surface law and fixture discretization

The independent reference uses adaptive Gauss–Kronrod21 integration of Gaussian-weighted1/X/XXᵀ on the original six triangles. Positive Duffy Gauss–Legendre P1 assembly at order16 agrees with its centroids within3.4e−17m in this smooth test; this numerical agreement is not a rigorous universal error bound. Four tests cover analytic polynomial triangle integrals, normalization, Gram PSD/row sums, affine-field fixture energy, refinement and frame covariance.

The continuous normalized patch centroids are approximately(22.643,32.659,19.527) and(74.007,70.233,73.169)mm. They are not the prescribed Gaussian centers. A balanced1N pair aligned with the prescribed centers therefore has a **continuous rest moment of9.31227mN·m**, even under accurate integration. Zero resultant is not zero couple; no zero dynamic-torque claim follows.

Current vertex-lumped centroid errors for the two patches fall from26.04/46.92mm on2tets to0.770/1.639mm on128 and0.0457/0.0994mm on8192. The old normalized rule is an explicitly documented approximation, not algebraically equal to continuous surface integration. The surface rule is the same in old and new interior-refinement families at each level, preserving the intended mesh-only control.

The NEW assembly exposes normalized load vectors l_i=∫psi N_i/Z and consistent scalar fixture Grams M_ij=∫psi N_i N_j/Z. Full world tensors use kron(M,K/D). Replacing M with diag(l) is a separate mass-lumping approximation. For one declared affine fixture field, old patch1 energy error is+240% at2tets,+9.79% at128 and+0.591% at8192. Even accurate row-sum weights followed by diagonal lumping add+161.7%,+9.15% and+0.569% respectively. These probe-specific percentages are not universal bounds or explanations of all trajectory error.

All existing physical results remain frozen. A matched quality-mesh-only control retains the old lumped law. Any subsequent consistent-assembly or centroid-aligned tension experiment requires its own coherent objective/gradient/work formulation and prospective protocol; no matrix-only substitution into the old hardcoded step is valid.

Evidence: [protocol](protocol.json), [complete compact geometry/quadrature report](../../gaussian-campaign-v01/mesh-surface-quadrature-v01/report.json). The diagnostic completed in2.55s without dynamics or GPU use.
