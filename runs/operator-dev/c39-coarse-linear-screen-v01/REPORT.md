# C39 public-action linear coarse screen

This prospectively frozen screen used supplied candidate physics only: E = 75 kPa, ν = 0.35, density 625 kg/m³, gravity, and the same graded block/support geometry. No C38 observations, object labels, nonlinear response fields, or new noise draws were opened. New action compilers accept public geometry/action only. All actions retain gravity; the three candidates have equal 2 N surface resultants.

An independent NumPy rest-tangent sparse matrix was assembled using StableP2Block.elastic with authored T10 rows/columns. Exact constrained elimination and an original Torch rest-HVP check passed for all 20 responses. Maximum residual was below 4.8×10⁻¹³ N; constraints were exactly zero. Two focused tests passed. Mechanics took 6.03 s (9.33 s including process startup), peak RSS 0.633 GiB; the CPU-only worker was reaped normally.

| Support | Mesh | Down Fisher | Side Fisher | Left-down Fisher | Preferred / runner-up ratio |
|---|---|---:|---:|---:|---:|
| Both ends | 4×2×2 | 1574.27 | 1336.33 | 1113.38 | Down, 1.178 |
| Both ends | 8×4×4 | 1771.93 | 1485.60 | 1285.22 | Down, 1.193 |
| Separated bottom patches | 4×2×2 | 171.64 | 523.89 | 112.75 | Side, 3.052 |
| Separated bottom patches | 8×4×4 | 228.03 | 663.94 | 150.17 | Side, 2.912 |

Fisher is ||∂u₉/∂log E||²/σ² = ||u₉||²/(200 µm)² for the total gravity-plus-action response at the same nine top material points. It is local material information conditional on support in a linear model, not a complete unknown-support utility or evidence of active-policy advantage. Gravity prefix and canonical +1 N responses, plus all 63 displacement modes, remain in result.json.

The predeclared screen criterion passes: different preferred actions by support, consistent across both meshes, each with at least a 5% margin. Left-down remains in the coarse evidence and is omitted only from the prospective fine linear confirmation after ranking third. No nonlinear case, inference dataset, or observation is admitted by this coarse result.
