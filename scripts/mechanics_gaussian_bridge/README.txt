Paper mechanics -> PIDG rasterizer
=================================
bridge.py reuses the paper's material-coordinate Gaussian binding and Q2 basis.
It computes positions and F from mechanical nodes, transports full covariance
as F Sigma F^T, and evaluates spherical-harmonic radiance in the local polar
rotation frame. PIDG's DeformModel is not instantiated or evaluated. Fixed
opacity and appearance coefficients remain shared with the existing paper.

Torch operations retain gradients to nodal state, Gaussian covariance and
appearance. Polar rotation uses Newton iterations rather than an SVD backward,
which is singular at identity/repeated singular values. Nonpositive Jacobians
and unconverged polar factors are rejected. Binding is fixed in reference space.

render_rollout.py validates sealed physical states and renders the existing
learned Q2 mechanics predictions with the official PIDG CUDA rasterizer. It
also backpropagates an image functional to mechanical nodes and checks finite,
nonzero gradients. CPU tests compare against the existing paper transport,
verify exact affine motion/covariance and check identity-state derivatives.

This connection does not infer a volume or boundary conditions from PIDG RGB.
It does not fit the paper's mechanics to Balls Reaction, and does not make the
previous material-regularizer ablation a full-method benchmark. Saved nodal
replay also does not establish gradients through the dynamic time integrator.
Those distinctions must remain explicit in comparison tables.

Run from repository root:
PYTHONPATH=. .venv/bin/python scripts/mechanics_gaussian_bridge/test_bridge.py
Use the isolated pidg-official-v01/env/bin/python for render_rollout.py,
--physical-output a sealed learned-method Q2 forecast directory,
--output a new local directory, --gpu 0 (or 1/2).
The current rollout loader is specific to the declared rectangular Q2 scenes;
it does not silently substitute block geometry for a public captured object.
