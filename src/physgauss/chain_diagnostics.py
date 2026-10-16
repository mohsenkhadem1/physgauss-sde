"""Rank-normalized/folded split R-hat and bulk/tail multichain ESS.

Vehtari et al., 2021, arxiv.org/abs/1903.08008. FFT autocovariance uses the
biased denominator N. ESS uses Geyer's initial positive/monotone pair sequence.
Constant or frozen chains are flagged, never given an automatic convergence pass.
"""
import math
import torch


def rank_normalize(x):
    flat = x.flatten()
    _, inverse, counts = torch.unique(flat, sorted=True, return_inverse=True, return_counts=True)
    # Ties receive their mean one-based rank.
    ranks = counts.cumsum(0).to(x.dtype) - (counts.to(x.dtype)-1)/2
    p = (ranks[inverse] - .375) / (len(flat) + .25)
    return torch.special.ndtri(p).reshape(x.shape)


def split_chains(x):
    n = x.shape[1] // 2
    return torch.cat((x[:, :n], x[:, -n:]), 0)


def basic_rhat(x):
    n = x.shape[1]
    within = x.var(1, unbiased=True).mean()
    if within <= 0:
        return None
    between_over_n = x.mean(1).var(unbiased=True)
    return float((((n-1)/n*within + between_over_n)/within).sqrt())


def effective_sample_size(x):
    """Multichain autocorrelation estimate with between-chain variance penalty."""
    m, n = x.shape
    centered = x - x.mean(1, keepdim=True)
    size = 1 << (2*n-1).bit_length()
    fft = torch.fft.rfft(centered, n=size, dim=1)
    acov = torch.fft.irfft(fft.conj()*fft, n=size, dim=1)[:, :n] / n
    within = acov[:, 0].mean()*n/(n-1)
    variance = (n-1)/n*within + x.mean(1).var(unbiased=True)
    if variance <= 0 or within <= 0:
        return None
    rho = 1 - (within - acov.mean(0))/variance
    rho[0] = 1.
    pairs = rho[:(n//2)*2].reshape(-1, 2).sum(1)
    nonpositive = torch.nonzero(pairs <= 0).flatten()
    if len(nonpositive):
        pairs = pairs[:int(nonpositive[0])]
    if len(pairs) == 0:
        return float(m*n)
    pairs = torch.cummin(pairs, 0).values
    tau = float(-1 + 2*pairs.sum())
    # Cap antithetic ESS at S log10(S), following the reference estimator.
    tau = max(tau, 1/math.log10(m*n))
    return float(m*n/tau)


def scalar_diagnostics(x):
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 16 or not bool(torch.isfinite(x).all()):
        raise ValueError("Need finite [>=2 chains, >=16 draws].")
    x = x.to(torch.float64)
    split = split_chains(x)
    ranked = rank_normalize(split)
    folded = rank_normalize((split - torch.quantile(split.flatten(), .5)).abs())
    values = [basic_rhat(ranked), basic_rhat(folded)]
    rhat = max(v for v in values if v is not None) if all(v is not None for v in values) else None
    bulk = effective_sample_size(ranked)
    bounds = torch.quantile(x.flatten(), x.new_tensor([.05, .95]))
    tails = [effective_sample_size((split <= bound).to(x.dtype)) for bound in bounds]
    tail = min(tails) if all(v is not None for v in tails) else None
    frozen = int((x.var(1, unbiased=False) == 0).sum())
    passed = rhat is not None and bulk is not None and tail is not None and frozen == 0
    passed = passed and rhat < 1.01 and bulk >= 400 and tail >= 400
    return {"rank_folded_split_rhat": rhat, "bulk_ess": bulk, "tail_ess": tail,
            "frozen_chains": frozen, "diagnostic_pass": bool(passed)}


def diagnose_quantities(values, names):
    if values.ndim != 3 or values.shape[2] != len(names):
        raise ValueError("Expected [chains, draws, named quantities].")
    records = {name: scalar_diagnostics(values[:, :, i]) for i, name in enumerate(names)}
    return {"criteria": "rank/folded split Rhat <1.01, bulk and tail ESS >=400, no frozen chains",
            "all_monitored_quantities_pass": all(v["diagnostic_pass"] for v in records.values()),
            "failed_quantities": [name for name, v in records.items() if not v["diagnostic_pass"]],
            "quantities": records,
            "limitation": "Passing finite-chain diagnostics cannot exclude unvisited modes or establish model calibration."}
