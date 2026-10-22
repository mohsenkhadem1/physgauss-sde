# Zero-copy full elastic HVP: paired linear control

Both 165,888-tetrahedron cases used the same Keast24 physical Hessian, immutable Q3 initial field and original uniform4 coarse preconditioner. Only the elastic Hessian action implementation changed. Warp recomputes compatible T10 gradients inside each quadrature/element thread, retains all cofactor cross terms, aliases Torch buffers and launches on the current Torch CUDA stream. Original Torch energy, gradient, Jacobian guards, mass, node blocks, preconditioner, CG acceptance and physical residual remain inherited.

The paired CG times were **150.727 s / 277 iterations** for Torch and **9.81068 s / 270 iterations** for Warp (15.36× in this control). First Warp JIT was 7.808 s, separately recorded and excluded. Different atomic summation order can change finite-precision CG paths.

The original Torch Hessian independently checked both directions: true residuals were 1.059×10⁻¹² and 9.293×10⁻¹³ N. Original candidate gradient residuals were 1.066×10⁻¹² and 9.365×10⁻¹³ N. Directions differed by at most 2.189×10⁻¹⁴ m; the new candidate was within 2.190×10⁻¹⁴ m of the immutable exact80 field. Clamps and Bernstein-positive Jacobian gates passed. Both processes were reaped and GPU2 released.

This is one paired, sequential fixed-linear-system control, not a nonlinear/trajectory speed guarantee or spatial/contact qualification. V01's pre-CG bitwise assertion on two identical Torch gradient evaluations failed at 4.34×10⁻¹⁹ N and is preserved. V02 shares one original accepted cache/R between both implementations, preserving the original tolerances and objective. Raw correction arrays/logs/caches remain local.
