"""Metropolis correction of a Gaussian-mixture reference using nonlinear RGB.

The reference only preconditions proposals. Every transition targets the full
nonlinear likelihood with the original discrete scale / standard-normal priors.
No withheld observations or truth values enter this module.
"""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass
class SamplerConfig:
    chains: int = 4
    warmup: int = 1000
    draws: int = 2000
    beta: float = .5
    defensive_mass: float = .05
    independent_probability: float = .15
    transport_probability: float = .15
    seed: int = 67001
    langevin: bool = False

    def validate(self):
        if self.chains < 2 or self.warmup < 0 or self.draws < 16:
            raise ValueError("Need >=2 chains, nonnegative warmup, and >=16 retained draws.")
        if not 0 < self.beta <= 1 or not 0 < self.defensive_mass < 1:
            raise ValueError("Invalid pCN step or defensive mixture mass.")
        probabilities = [self.independent_probability, self.transport_probability]
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or sum(probabilities) >= 1:
            raise ValueError("Move probabilities must leave positive probability for pCN.")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative.")
        return self


class MixtureReference:
    def __init__(self, posterior, fixed_scale=None):
        if fixed_scale is None:
            self.components = posterior.components
            nle = posterior.prior_weights.new_tensor([c.negative_log_evidence for c in self.components])
            # Recompute in log space, including components whose float weights underflow.
            self.log_weights = torch.log_softmax(posterior.prior_weights.log() - nle, 0)
            self.prior_weights = posterior.prior_weights
        else:
            self.components = [c for c in posterior.components if c.scale == fixed_scale]
            if len(self.components) != 1:
                raise ValueError("Fixed scale is absent from the reference.")
            self.log_weights = posterior.reference.new_zeros(1)
            self.prior_weights = posterior.reference.new_ones(1)
        self.scales = posterior.reference.new_tensor([c.scale for c in self.components])
        self.dimension = len(self.components[0].mean)

    def decode(self, white, indices):
        coordinates = torch.empty_like(white)
        for i in indices.unique().tolist():
            selected = indices == i
            component = self.components[i]
            delta = torch.linalg.solve_triangular(component.precision_cholesky.T,
                                                   white[selected].T, upper=True).T
            coordinates[selected] = component.mean + delta
        return coordinates

    def whiten_gradient(self, gradient, indices):
        result = torch.empty_like(gradient)
        for i in indices.unique().tolist():
            selected = indices == i
            result[selected] = torch.linalg.solve_triangular(
                self.components[i].precision_cholesky, gradient[selected].T, upper=False).T
        return result


def metropolis_log_acceptance(old_ratio, new_ratio, old_indices, new_indices,
                              reference_log_weights, proposal_log_weights, switch):
    """Target in whitened coordinates is q(k)*phi(u)*exp(log_ratio).

    The component-dependent affine Jacobian is absorbed by the normalized
    Gaussian reference. Omitting q(k)/g(k) would change the scale prior.
    """
    result = new_ratio - old_ratio
    factor = reference_log_weights - proposal_log_weights
    return result + torch.where(switch, factor[new_indices] - factor[old_indices], 0.)


def sample_nonlinear(reference, correction, config=SamplerConfig(), progress=None):
    """correction(z, scales) -> (log Lnonlin/Llinear, log Lnonlin, invalid).

    Random numbers are independent across chains. Adaptation uses only each
    chain's warmup acceptance; all proposal settings freeze for retained draws.
    Initial states are independent reference draws, not importance resamples.
    """
    config.validate()
    generator = torch.Generator().manual_seed(config.seed)
    dtype = reference.log_weights.dtype
    proposal_weights = ((1-config.defensive_mass) * reference.log_weights.exp()
                        + config.defensive_mass * reference.prior_weights)
    proposal_log_weights = proposal_weights.log()
    indices = torch.multinomial(proposal_weights, config.chains, True, generator=generator)
    white = torch.randn(config.chains, reference.dimension, dtype=dtype, generator=generator)
    coordinates = reference.decode(white, indices)
    def evaluate(z, ids):
        if config.langevin:
            values, grad = correction.value_and_grad(z, reference.scales[ids])
            return (*values, reference.whiten_gradient(grad, ids))
        return (*correction(z, reference.scales[ids]), torch.zeros_like(z))
    ratio, log_likelihood, invalid, gradient = evaluate(coordinates, indices)
    if bool(invalid.any()) or not bool(torch.isfinite(ratio).all()):
        raise RuntimeError("Nonfinite initial target state; choose a new seed or improve the reference.")
    beta = torch.full((config.chains,), config.beta, dtype=dtype)
    window_accept = torch.zeros_like(beta)
    window_count = torch.zeros_like(beta)
    shape = (config.chains, config.draws)
    saved = {
        "coordinates": torch.empty(*shape, reference.dimension, dtype=dtype),
        "component_indices": torch.empty(shape, dtype=torch.long),
        "scales": torch.empty(shape, dtype=dtype),
        "log_ratio": torch.empty(shape, dtype=dtype),
        "log_likelihood": torch.empty(shape, dtype=dtype),
    }
    counts = {phase: {key: torch.zeros(config.chains, dtype=torch.long) for key in
        ["pcn_attempts", "pcn_accepts", "independent_attempts", "independent_accepts",
         "transport_attempts", "transport_accepts", "scale_change_attempts",
         "scale_change_accepts", "invalid_proposals"]} for phase in ["warmup", "retained"]}
    total = config.warmup + config.draws
    for step in range(total):
        # One mixture-of-kernels transition per chain and iteration.
        move = torch.rand(config.chains, generator=generator)
        independent = move < config.independent_probability
        transport = (move >= config.independent_probability) & (
            move < config.independent_probability + config.transport_probability)
        pcn = ~(independent | transport)
        switch = ~pcn
        candidate_indices = indices.clone()
        if bool(switch.any()):
            candidate_indices[switch] = torch.multinomial(proposal_weights, int(switch.sum()),
                                                         True, generator=generator)
        innovation = torch.randn(white.shape, dtype=dtype, generator=generator)
        candidate_white = white.clone()
        candidate_white[pcn] = ((1-beta[pcn].square()).sqrt()[:, None] * white[pcn]
                                + beta[pcn, None] * innovation[pcn])
        if config.langevin:
            candidate_white[pcn] += (1-(1-beta[pcn].square()).sqrt())[:, None]*gradient[pcn]
        candidate_white[independent] = innovation[independent]
        candidate = reference.decode(candidate_white, candidate_indices)
        proposed_ratio, proposed_likelihood, invalid, proposed_gradient = evaluate(candidate, candidate_indices)
        if bool(torch.isnan(proposed_ratio).any()) or bool(torch.isposinf(proposed_ratio).any()):
            raise RuntimeError("Nonlinear likelihood correction returned NaN or positive infinity.")
        proposed_ratio = torch.where(invalid, -torch.inf, proposed_ratio)
        log_acceptance = metropolis_log_acceptance(ratio, proposed_ratio, indices, candidate_indices,
                                                   reference.log_weights, proposal_log_weights, switch)
        if config.langevin and bool(pcn.any()):
            log_acceptance[pcn] += langevin_log_correction(white[pcn], candidate_white[pcn],
                gradient[pcn], proposed_gradient[pcn], beta[pcn])
        accept = torch.rand(config.chains, dtype=dtype, generator=generator).log() < log_acceptance
        phase = "warmup" if step < config.warmup else "retained"
        counters = counts[phase]
        changed = candidate_indices != indices
        for name, mask in [("pcn", pcn), ("independent", independent), ("transport", transport),
                           ("scale_change", changed)]:
            counters[name + "_attempts"] += mask.long()
            counters[name + "_accepts"] += (mask & accept).long()
        counters["invalid_proposals"] += invalid.long()
        white[accept] = candidate_white[accept]
        coordinates[accept] = candidate[accept]
        indices[accept] = candidate_indices[accept]
        ratio[accept] = proposed_ratio[accept]
        log_likelihood[accept] = proposed_likelihood[accept]
        gradient[accept] = proposed_gradient[accept]
        if step < config.warmup:
            window_accept += (pcn & accept).to(dtype)
            window_count += pcn.to(dtype)
            if (step + 1) % 25 == 0:
                gain = 1 / math.sqrt((step+1)/25)
                rate = window_accept / window_count.clamp_min(1)
                update = torch.where(window_count > 0, gain*(rate-.35), 0.)
                beta = (beta.log() + update).exp().clamp(.03, .98)
                window_accept.zero_(); window_count.zero_()
        else:
            j = step - config.warmup
            saved["coordinates"][:, j] = coordinates
            saved["component_indices"][:, j] = indices
            saved["scales"][:, j] = reference.scales[indices]
            saved["log_ratio"][:, j] = ratio
            saved["log_likelihood"][:, j] = log_likelihood
        if progress and ((step+1) % 100 == 0 or step+1 == total):
            progress(f"  transition {step+1}/{total}, beta=" + str([round(x, 3) for x in beta.tolist()]))
    saved["config"] = asdict(config)
    saved["final_beta_per_chain"] = beta.tolist()
    saved["transition_counts"] = {p: {k: v.tolist() for k, v in c.items()} for p, c in counts.items()}
    return saved


def langevin_log_correction(old, new, old_gradient, new_gradient, beta):
    """Proposal asymmetry plus Gaussian-reference density ratio for pCNL.

    u' = a*u + (1-a)*grad(r(u)) + beta*epsilon, a=sqrt(1-beta^2).
    Combined with r'-r this is the full Metropolis-Hastings ratio.
    """
    a = (1-beta.square()).sqrt()[:, None]
    forward = new-a*old-(1-a)*old_gradient
    reverse = old-a*new-(1-a)*new_gradient
    log_q_ratio = -.5*(reverse.square().sum(1)-forward.square().sum(1))/beta.square()
    log_phi_ratio = -.5*(new.square().sum(1)-old.square().sum(1))
    return log_q_ratio+log_phi_ratio
