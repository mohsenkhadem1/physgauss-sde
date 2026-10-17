"""Nonlinear RGB posterior with an arbitrary fitted Gaussian proposal.

Unlike the historical common-linearization sampler, the correction uses the
complete prior and normalized reference density. Refit proposals do not change
the target. No ground-truth force is accepted by this API.
"""
import math
import time
import torch
from .inference import fit, objective
from .mcmc import MixtureReference
from .physics import InvalidDeformation
from .scale_inference import LinearComponent, ScalePosterior


def fitted_reference(model, observations, progress=print, exact_hessian=False):
    began = time.perf_counter()
    fitted = fit(model, observations, progress)
    precision_cholesky = fitted.precision_cholesky
    hessian_record = {"method": "Gauss-Newton", "eigenvalue_floor": None}
    mean = fitted.mean
    polishing = []
    if exact_hessian:
        # Finite differences of exact gradients include residual curvature.
        # This changes only the proposal, never the nonlinear target density.
        h = 1e-3
        for iteration in range(3):
            columns = []
            for start in range(0, model.dimension, 2):
                count = min(2, model.dimension-start)
                shift = torch.zeros(count, model.dimension, dtype=mean.dtype)
                shift[torch.arange(count), torch.arange(start, start+count)] = h
                points = torch.cat((mean[None]+shift, mean[None]-shift)).requires_grad_()
                gradient, = torch.autograd.grad(objective(model, observations, points), points)
                columns.append(((gradient[:count]-gradient[count:])/(2*h)).detach().T)
            hessian = torch.cat(columns, 1)
            hessian = .5*(hessian+hessian.T)
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            if bool((eigenvalues <= 0).any()):
                # Depth sorting is piecewise smooth. Do not turn a large
                # negative finite-difference eigenvalue into a huge proposal.
                from .inference import laplace_precision
                precision_cholesky = laplace_precision(model, observations, mean,
                    model.config.finite_difference_chunk, progress=progress)
                hessian_record = {"method": "Gauss-Newton fallback after indefinite nonlinear curvature",
                    "minimum_raw_eigenvalue": float(eigenvalues.min()), "eigenvalue_floor": None,
                    "rejected_nonpositive_eigenvalues": int((eigenvalues <= 0).sum())}
                break
            floor = .2
            precision = (eigenvectors*eigenvalues.clamp_min(floor)[None])@eigenvectors.T
            precision_cholesky = torch.linalg.cholesky(precision)
            variable = mean[None].requires_grad_()
            value = objective(model, observations, variable)
            gradient, = torch.autograd.grad(value, variable)
            direction = -torch.cholesky_solve(gradient.detach().T, precision_cholesky).T[0]
            decrement = float((-gradient.detach()[0]@direction).clamp_min(0).sqrt())
            polishing.append({"iteration": iteration, "objective": float(value.detach()), "decrement": decrement})
            hessian_record = {"method": "central finite differences of nonlinear objective gradients",
                "difference_step": h, "gradient_batch_chunk": 2, "minimum_raw_eigenvalue": float(eigenvalues.min()),
                "floored_eigenvalues": int((eigenvalues < floor).sum()), "eigenvalue_floor": floor}
            if iteration == 2 or decrement < .05:
                break
            accepted = False
            for backtrack in range(16):
                candidate = mean + .5**backtrack*direction
                try:
                    with torch.no_grad():
                        candidate_value = float(objective(model, observations, candidate[None]))
                except InvalidDeformation:
                    continue
                if candidate_value < float(value.detach()) + 1e-4*.5**backtrack*float(gradient.detach()[0]@direction):
                    mean = candidate.detach(); accepted = True; break
            if not accepted:
                break
    component = LinearComponent(1., mean.detach(), precision_cholesky, 0.)
    dummy = fitted.mean.new_zeros(1)
    posterior = ScalePosterior([component], dummy+1, dummy+1, mean.detach(), dummy,
                               fitted.mean.new_empty(0, model.dimension))
    reference = MixtureReference(posterior, fixed_scale=1.)
    variable = mean.detach()[None].requires_grad_()
    center_value = objective(model, observations, variable)
    center_gradient, = torch.autograd.grad(center_value, variable)
    return reference, {"map_objective": fitted.objective, "map_gradient_norm": fitted.gradient_norm,
        "map_gauss_newton_decrement": fitted.newton_decrement, "proposal_precision": hessian_record,
        "nonlinear_polishing": polishing,
        "proposal_center_objective": float(center_value.detach()),
        "proposal_center_gradient_norm": float(center_gradient.norm()),
        "seconds": time.perf_counter()-began}


class GaussianTargetCorrection:
    def __init__(self, model, observations, reference):
        self.model, self.observations = model, observations
        if len(reference.components) != 1:
            raise ValueError("This correction conditions on one fixed model, without model probabilities.")
        component = reference.components[0]
        self.mean = component.mean
        self.cholesky = component.precision_cholesky
        self.half_logdet = self.cholesky.diagonal().log().sum()

    def _values(self, coordinates):
        predictions, invalid = [], []
        for selected in coordinates.split(8):
            try:
                predictions.append(self.model.predict(selected, self.observations.indices))
                invalid += [False]*len(selected)
            except InvalidDeformation:
                for row in selected.split(1):
                    try:
                        predictions.append(self.model.predict(row, self.observations.indices))
                        invalid.append(False)
                    except InvalidDeformation:
                        predictions.append(row.new_zeros(1, *self.observations.rgb.shape))
                        invalid.append(True)
        prediction = torch.cat(predictions)
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("Nonfinite renderer output; stopping inference.")
        invalid = torch.tensor(invalid, dtype=torch.bool)
        likelihood = -.5*((prediction-self.observations.rgb[None])/self.observations.sigma).flatten(1).square().sum(1)
        white = (coordinates-self.mean)@self.cholesky
        correction = likelihood-.5*coordinates.square().sum(1)+.5*white.square().sum(1)-self.half_logdet
        return torch.where(invalid, -torch.inf, correction), likelihood, invalid

    @torch.no_grad()
    def __call__(self, coordinates, ignored_scales):
        return self._values(coordinates)

    def value_and_grad(self, coordinates, ignored_scales):
        with torch.enable_grad():
            variable = coordinates.detach().requires_grad_(True)
            values = self._values(variable)
            gradient, = torch.autograd.grad(torch.where(values[2], 0., values[0]).sum(), variable)
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Nonfinite target gradient; stopping inference.")
        return tuple(v.detach() for v in values), gradient.detach()
