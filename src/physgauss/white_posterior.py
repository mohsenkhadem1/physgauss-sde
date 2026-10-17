"""Fixed Gaussian coordinates and optional image Taylor proposal forces.

The Taylor image model is used ONLY to make Hamiltonian proposals cheaply.
Acceptance always evaluates the original nonlinear renderer and normalized
force-marginalized density. No image or physical normalizer is removed.
"""
import torch
from .physics import InvalidDeformation


class WhitePosterior:
    def __init__(self, model, observations, mean, chol, image_taylor=True):
        self.model, self.observations, self.mean, self.chol = model, observations, mean.detach(), chol.detach()
        self.transform = torch.linalg.solve_triangular(chol.T, torch.eye(len(mean), dtype=mean.dtype), upper=True)
        self.image_taylor = image_taylor
        with torch.no_grad():
            self.center = model.objective(mean[None], observations)[0]
            frames = model.trajectory(mean[None]).x
        if image_taylor:
            self.local_mean = frames[0, model.view_frames][:, model.free].flatten(1)
            self.image_values, self.image_gradients, self.image_hessians = [], [], []
            for slot, (frame, _) in enumerate(model.view_specs):
                nodes = frames[:, [frame]].clone()
                def loss(q):
                    shifted = nodes.clone(); shifted[0, 0, model.free] = q.reshape(-1, 3)
                    rgb = model.renderer(shifted, model.view_eye[[slot]], model.view_rotation[[slot]])[0, 0]
                    return .5*((rgb-observations.rgb[slot])/observations.sigma).square().sum()
                q = self.local_mean[slot].detach().requires_grad_()
                value = loss(q)
                g, = torch.autograd.grad(value, q)
                h = torch.autograd.functional.hessian(loss, q, vectorize=False)
                self.image_values.append(value.detach()); self.image_gradients.append(g.detach()); self.image_hessians.append(h.detach())
            self.image_values, self.image_gradients, self.image_hessians = [torch.stack(x) for x in
                (self.image_values, self.image_gradients, self.image_hessians)]

    def decode(self, white):
        return self.mean+white@self.transform.T

    def taylor_image(self, state):
        local = self.model.trajectory(state).x[:, self.model.view_frames][:, :, self.model.free].flatten(2)
        delta = local-self.local_mean
        return (self.image_values+(delta*self.image_gradients).sum(-1)
                +.5*torch.einsum('bvi,vij,bvj->bv', delta, self.image_hessians, delta)).sum(-1)

    def proposal_objective(self, state):
        if not self.image_taylor:
            return self.model.objective(state, self.observations)
        return self.model.dynamics_objective(state)+self.taylor_image(state)

    def __call__(self, white):
        w = white.detach().requires_grad_()
        try:
            ratios = -self.proposal_objective(self.decode(w))+self.center+.5*w.square().sum(-1)
            grad, = torch.autograd.grad(ratios.sum(), w)
            invalid = (~torch.isfinite(ratios)) | (~torch.isfinite(grad).all(-1))
            return ratios.detach(), torch.nan_to_num(grad.detach()), invalid
        except (InvalidDeformation, ValueError):
            if len(w) == 1:
                return w.new_full((1,), -torch.inf), torch.zeros_like(w), torch.ones(1, dtype=torch.bool)
            rows = [self(row[None]) for row in w]
            return tuple(torch.cat([row[i] for row in rows]) for i in range(3))

    @torch.no_grad()
    def full_log_ratio(self, white):
        try:
            ratios = -self.model.objective(self.decode(white), self.observations)+self.center+.5*white.square().sum(-1)
            return ratios, ~torch.isfinite(ratios)
        except (InvalidDeformation, ValueError):
            if len(white) == 1:
                return white.new_full((1,), -torch.inf), torch.ones(1, dtype=torch.bool)
            rows = [self.full_log_ratio(row[None]) for row in white]
            return tuple(torch.cat([row[i] for row in rows]) for i in range(2))
