"""Exact observed marginal Hessian from local mechanics and force Schur blocks.

Same target as NeuralMarginalModel.objective. Local AD avoids thousands of
full-image gradient calls. The logdet force-volume Hessian is retained, including
its global negative trace term. No clipping of negative eigenvalues.
"""
import time
import torch
from physgauss.physics import material_response


def block_hessians(function, base):
    base = base.detach().requires_grad_()
    def gradient(shift):
        values = function(base+shift)
        return torch.autograd.grad(values.sum(), base, create_graph=True)[0]
    # Explicit JVPs avoid a PyTorch vmap/index-scatter double-backward limitation.
    columns = []
    for index in range(base.shape[-1]):
        tangent = base.new_zeros(base.shape[-1]); tangent[index] = 1.
        with torch.autograd.forward_ad.dual_level():
            shift = torch.autograd.forward_ad.make_dual(torch.zeros_like(tangent), tangent)
            value = gradient(shift)
            columns.append(torch.autograd.forward_ad.unpack_dual(value).tangent.detach())
    return torch.stack(columns, -1)


def observed_precision(model, observations, fitted):
    began = time.perf_counter(); mean = fitted["mean"]; n = len(mean); d = model.free_dof; k = model.force_dimension
    with torch.no_grad():
        full = model.step_trajectory(mean[None]); elimination = model.elimination_at(mean[None])
        a = elimination.conditional_mean(model.residual(mean[None]))[0]
        x, p, u = full.x[0, :-1].detach(), full.v[0, :-1].detach(), full.v[0, 1:].detach()
        local = torch.cat((x[:, model.free].flatten(1), p[:, model.free].flatten(1), u[:, model.free].flatten(1),
            mean[:2].expand(model.nsteps, 2), a.expand(model.nsteps, k)), -1)
    state_map = model.local_state_map(mean)
    mapping = mean.new_zeros(model.nsteps, 3*d+2+k, n+k)
    mapping[:, :2*d, :n] = state_map
    rows = torch.arange(model.nsteps*d)
    current_map = mean.new_zeros(model.nsteps*d, n)
    current_map[rows, 8+rows] = model.marginal_config.velocity_coordinate_scale
    mapping[:, 2*d:3*d, :n] = current_map.reshape(model.nsteps, d, n)
    mapping[:, 3*d, 0] = 1.; mapping[:, 3*d+1, 1] = 1.
    mapping[:, 3*d+2:, n:] = torch.eye(k, dtype=mean.dtype)

    def decode_nodes(q):
        nodes, previous = x.clone(), p.clone()
        nodes[:, model.free] = q[:, :d].reshape(model.nsteps, -1, 3)
        previous[:, model.free] = q[:, d:2*d].reshape(model.nsteps, -1, 3)
        return nodes, previous

    def local_joint(q):
        nodes, previous = decode_nodes(q)
        young = (model.prior.mean[0]+model.prior.std[0]*q[:, 3*d]).exp()
        drag = (model.prior.mean[1]+model.prior.std[1]*q[:, 3*d+1]).exp()
        force = material_response(nodes[None], model.mesh, young[None])[0][0]
        acceleration = (force/model.mesh.mass[:, None]+mean.new_tensor([0., -.6, 0.]))[:, model.free].flatten(1)
        c = drag*model.kick_sizes/2
        residual = ((1+c[:, None])*q[:, 2*d:3*d]-(1-c[:, None])*q[:, d:2*d]
            -model.kick_sizes[:, None]*acceleration-torch.einsum("sdk,sk->sd", model.load_mapping, q[:, 3*d+2:]))
        covariance = model.covariance_at_nodes(nodes[None], previous[None])[0]
        chol = torch.linalg.cholesky(covariance)
        white = torch.linalg.solve_triangular(chol, residual[..., None], upper=False)[..., 0]
        return .5*white.square().sum(-1)+chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    local_h = block_hessians(local_joint, local)
    flat_map = mapping.flatten(0, 1)
    joint = flat_map.T@(local_h@mapping).flatten(0, 1)
    joint[:8, :8] += torch.eye(8, dtype=mean.dtype)
    joint[n:, n:] += torch.eye(k, dtype=mean.dtype)
    joint[1, 1] += model.normalizer_curvature(mean)
    torch.testing.assert_close(joint[n:, n:], elimination.precision[0], rtol=1e-10, atol=1e-8)
    precision = joint[:n, :n]-joint[:n, n:]@torch.cholesky_solve(joint[n:, :n], elimination.cholesky[0])

    # Exact Hessian of +.5 logdet A(z), A=I+sum_j G_j^T Q_j^-1 G_j.
    s = elimination.conditional_covariance[0].detach()
    def contributions(q):
        nodes, previous = decode_nodes(q)
        covariance = model.covariance_at_nodes(nodes[None], previous[None])[0]
        return model.load_mapping.transpose(-1, -2)@torch.linalg.solve(covariance, model.load_mapping)
    local_noise = local[:, :2*d].detach()
    weighted = lambda q: .5*(contributions(q)*s.T).sum((-1, -2))
    volume_local_h = block_hessians(weighted, local_noise)
    noise_map = state_map.flatten(0, 1)
    precision += noise_map.T@(volume_local_h@state_map).flatten(0, 1)
    derivatives = torch.autograd.functional.jacobian(lambda shift: contributions(local_noise+shift),
        mean.new_zeros(2*d), vectorize=True, strategy="forward-mode").detach().permute(0, 3, 1, 2)
    a_chol = elimination.cholesky[0]
    white = torch.linalg.solve_triangular(a_chol, derivatives, upper=False)
    white = torch.linalg.solve_triangular(a_chol, white.transpose(-1, -2), upper=False).transpose(-1, -2)
    global_features = torch.einsum("spab,spz->abz", white, state_map).flatten(0, 1)
    precision -= .5*global_features.T@global_features

    # True observed image curvature, including nonlinear residual curvature.
    with torch.no_grad(): frames = model.trajectory(mean[None]).x
    for slot, (frame, _) in enumerate(model.view_specs):
        nodes = frames[:, [frame]].clone()
        def image_loss(q):
            shifted = nodes.clone(); shifted[0, 0, model.free] = q.reshape(-1, 3)
            prediction = model.renderer(shifted, model.view_eye[[slot]], model.view_rotation[[slot]])
            return .5*((prediction[0, 0]-observations.rgb[slot])/observations.sigma).square().sum()
        h = torch.autograd.functional.hessian(image_loss, nodes[0, 0, model.free].flatten(),
            vectorize=False).detach()
        transform = model.position_map[frame]
        precision += transform.T@h@transform
    asymmetry = float((precision-precision.T).norm()/precision.norm())
    precision = .5*(precision+precision.T)
    chol, info = torch.linalg.cholesky_ex(precision)
    if int(info): raise RuntimeError("Observed marginal Hessian is indefinite; no clipping.")
    transform = torch.linalg.solve_triangular(fitted["precision_cholesky"].T, torch.eye(n, dtype=mean.dtype), upper=True)
    white_hessian = transform.T@precision@transform
    result = dict(fitted); result["precision_cholesky"] = chol
    return result, {"method": "exact local AD joint Hessian, force Schur complement and full logdetA Hessian; exact observed image curvature",
        "relative_antisymmetry": asymmetry, "seconds": time.perf_counter()-began,
        "white_eigenvalue_min": float(torch.linalg.eigvalsh(white_hessian).min()),
        "white_eigenvalue_max": float(torch.linalg.eigvalsh(white_hessian).max()),
        "qualification": "observed Hessian Gaussian remains local, not an exact nonlinear posterior"}, white_hessian


def directional_refinement(model, observations, fitted, white_hessian, seed=98001):
    mean = fitted["mean"]; chol = fitted["precision_cholesky"]
    transform = torch.linalg.solve_triangular(chol.T, torch.eye(len(mean), dtype=mean.dtype), upper=True)
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(3, len(mean), dtype=mean.dtype, generator=generator)
    directions /= directions.norm(dim=-1, keepdim=True)
    parameter_directions = transform[:8]/transform[:8].norm(dim=-1, keepdim=True)
    directions = torch.cat((directions, parameter_directions))
    # Independent full-target reverse AD checks distinguish a local assembly bug
    # from finite-difference truncation around the sharp smooth-magnitude scale.
    point = mean.detach()[None].requires_grad_()
    gradient, = torch.autograd.grad(model.objective(point, observations).sum(), point, create_graph=True)
    ad_errors = []
    for direction in directions:
        product = (gradient[0]*(transform@direction)).sum()
        derivative, = torch.autograd.grad(product, point, retain_graph=True)
        expected = white_hessian@direction
        ad_errors.append(float((derivative[0].detach()@transform-expected).norm()/expected.norm()))
    if max(ad_errors) > 1e-7:
        raise RuntimeError(f"Local observed curvature disagrees with full-target AD: {ad_errors}")
    del gradient, point
    records = []
    for epsilon in (.01, .005, .001, .0005, .0001, .00005):
        errors = []
        for direction in directions:
            gradients = []
            for sign in (1, -1):
                point = (mean+sign*epsilon*(transform@direction)).detach()[None].requires_grad_()
                derivative, = torch.autograd.grad(model.objective(point, observations).sum(), point)
                gradients.append(derivative[0].detach()@transform)
            finite = (gradients[0]-gradients[1])/(2*epsilon)
            expected = white_hessian@direction
            errors.append(float((finite-expected).norm()/expected.norm()))
        records.append({"epsilon_white": epsilon, "relative_Hv_errors": errors, "full_target_AD_relative_Hv_errors": ad_errors})
        if epsilon <= .005 and max(errors) <= 1e-4: break
    if max(records[-1]["relative_Hv_errors"]) > 1e-4:
        raise RuntimeError(f"Observed curvature finite differences did not refine: {records}")
    return records
