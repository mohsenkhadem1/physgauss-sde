"""Metropolis-corrected Hamiltonian sampling in a fixed Gaussian reference.

The Gaussian part is integrated by an exact rotation. Only the non-Gaussian
correction is kicked. This avoids the dimension-dependent energy error of a
leapfrog integrator applied to thousands of nearly Gaussian coordinates.
Warmup step-size adaptation is discarded and frozen before retained draws.
"""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass
class SplitHMCConfig:
    chains: int = 4
    warmup: int = 500
    draws: int = 2000
    step_size: float = .3
    min_time: float = 1.0
    max_time: float = 2.0
    target_accept: float = .8
    seed: int = 98001

    def validate(self):
        if self.chains < 2 or self.warmup < 0 or self.draws < 16:
            raise ValueError('Need >=2 chains, nonnegative warmup, and >=16 draws.')
        if not 0 < self.step_size <= 1 or not 0 < self.min_time <= self.max_time:
            raise ValueError('Invalid Hamiltonian time settings.')
        if not .5 < self.target_accept < 1 or self.seed < 0:
            raise ValueError('Invalid adaptation or seed.')
        return self


def split_trajectory(position, momentum, log_ratio, gradient, evaluate, step_size, steps):
    """Reversible volume-preserving kick/rotate/kick map; invalid proposals reject.

    evaluate(w) returns log(target/standard_normal), its gradient, and invalid.
    An invalid intermediate point invalidates the entire trajectory. It is
    allowed to finish solely to retain the vectorized operation; it cannot be accepted.
    """
    q, p, g = position.clone(), momentum.clone(), gradient
    invalid = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    c, s = math.cos(step_size), math.sin(step_size)
    for _ in range(steps):
        p = p+.5*step_size*g
        q, p = c*q+s*p, c*p-s*q
        value, g, bad = evaluate(q)
        invalid |= bad
        p = p+.5*step_size*g
    # Momentum reversal makes the deterministic proposal an involution.
    return q, -p, value, g, invalid


def sample_split_hmc(evaluate, dimension, config=SplitHMCConfig(), initial=None, progress=None, resume=None):
    config.validate()
    rng = torch.Generator().manual_seed(config.seed)
    if resume is not None:
        if config.warmup != 0 or config.chains != resume['white'].shape[0]:
            raise ValueError('Continuation must keep the chains and discard no further warmup.')
        for field in ['min_time', 'max_time', 'seed']:
            if getattr(config, field) != resume['config'][field]:
                raise ValueError('Continuation cannot change the frozen kernel.')
        if config.step_size != resume['step_size']:
            raise ValueError('Continuation must use the frozen step size.')
        initial = resume['white'][:, -1]
    q = (torch.randn(config.chains, dimension, dtype=torch.float64, generator=rng)
         if initial is None else initial.detach().clone())
    if q.shape != (config.chains, dimension):
        raise ValueError('Initial chains have wrong dimensions.')
    ratio, gradient, invalid = evaluate(q)
    if hasattr(evaluate, 'full_log_ratio'):
        ratio, full_invalid = evaluate.full_log_ratio(q)
        invalid |= full_invalid
    if bool(invalid.any()):
        raise ValueError('Invalid initial chain state.')
    epsilon = config.step_size
    # Shared adaptation based on all chains; independent momenta and MH draws.
    log_epsilon = math.log(epsilon)
    mu, averaged, hbar = math.log(10*epsilon), log_epsilon, 0.
    saved = torch.empty(config.chains, config.draws, dimension, dtype=q.dtype)
    ratios = torch.empty(config.chains, config.draws, dtype=q.dtype)
    energy_errors = torch.empty_like(ratios)
    counts = {phase: {k: [0]*config.chains for k in ['accepted', 'invalid', 'divergent']}
              for phase in ['warmup', 'retained']}
    if resume is not None:
        rng.set_state(resume['rng_state'])
        counts = {p: {k: list(v) for k, v in record.items()} for p, record in resume['counts'].items()}
    gradient_evaluations = 1
    for iteration in range(config.warmup+config.draws):
        # A random, state-independent length avoids harmonic resonance.
        duration = config.min_time+(config.max_time-config.min_time)*float(torch.rand((), generator=rng))
        steps = max(1, round(duration/epsilon))
        p = torch.randn(q.shape, dtype=q.dtype, generator=rng)
        candidate, final_p, new_ratio, new_gradient, bad = split_trajectory(
            q, p, ratio, gradient, evaluate, epsilon, steps)
        gradient_evaluations += steps
        if hasattr(evaluate, 'full_log_ratio'):
            new_ratio, full_invalid = evaluate.full_log_ratio(candidate)
            bad |= full_invalid
        delta = (-new_ratio+.5*(candidate.square().sum(1)+final_p.square().sum(1))
                 +ratio-.5*(q.square().sum(1)+p.square().sum(1)))
        bad |= ~torch.isfinite(delta)
        probability = torch.where(bad, 0., torch.exp(-delta.clamp_min(0)))
        accept = torch.rand(config.chains, dtype=q.dtype, generator=rng) < probability
        q[accept], ratio[accept], gradient[accept] = candidate[accept], new_ratio[accept], new_gradient[accept]
        phase = 'warmup' if iteration < config.warmup else 'retained'
        for key, mask in [('accepted', accept), ('invalid', bad), ('divergent', bad | (delta.abs() > 100))]:
            counts[phase][key] = [a+int(b) for a, b in zip(counts[phase][key], mask)]
        if iteration < config.warmup:
            t = iteration+1
            rate = float(probability.mean())
            hbar = (1-1/(t+10))*hbar+(config.target_accept-rate)/(t+10)
            log_epsilon = mu-math.sqrt(t)/.05*hbar
            # A bounded step prevents excessive warmup excursions.
            log_epsilon = min(math.log(.8), max(math.log(.015), log_epsilon))
            averaged = t**(-.75)*log_epsilon+(1-t**(-.75))*averaged
            epsilon = math.exp(averaged if t == config.warmup else log_epsilon)
        else:
            i = iteration-config.warmup
            saved[:, i], ratios[:, i], energy_errors[:, i] = q, ratio, torch.where(bad, torch.inf, delta)
        if progress and ((iteration+1) % 25 == 0 or iteration+1 == config.warmup+config.draws):
            progress({'transition': iteration+1, 'total': config.warmup+config.draws,
                      'step_size': epsilon, 'steps': steps, 'counts': counts[phase]})
    if resume is not None:
        saved = torch.cat((resume['white'], saved), 1)
        ratios = torch.cat((resume['log_ratio'], ratios), 1)
        energy_errors = torch.cat((resume['energy_errors'], energy_errors), 1)
        gradient_evaluations += resume['gradient_evaluations']
    return {'white': saved, 'log_ratio': ratios, 'energy_errors': energy_errors,
            'config': asdict(config), 'step_size': epsilon, 'counts': counts,
            'gradient_evaluations': gradient_evaluations, 'rng_state': rng.get_state()}
