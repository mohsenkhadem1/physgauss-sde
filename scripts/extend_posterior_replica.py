"""One predeclared fixed-kernel extension after a borderline MCSE replica check.

Preserves both initial 2,000-draw results. Adds 2,000 draws per chain without
adaptation; supplementary results never replace the frozen 36-case benchmark.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from physgauss.split_hmc import SplitHMCConfig, sample_split_hmc
from physgauss.white_posterior import WhitePosterior
from physgauss.study import write_json
from physgauss.cli import source_fingerprint
from converged_sparse_comparison import setup, summarize_chains, compact_diagnostics, score


def run(source, output, kind):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old_chain, = source.rglob('chains.pt')
    old_reference, = source.rglob('reference.pt')
    shutil.copyfile(old_reference, output/'reference.pt')
    model, obs, fit, row, data, manifest = setup('pulse_and_stochastic__4101__neural', output)
    previous = torch.load(old_chain, weights_only=True)
    if previous['white'].shape != (4, 2000, 1448):
        raise ValueError('Extension requires exactly the original four 2000-draw chains.')
    target = WhitePosterior(model, obs, fit['mean'], fit['precision_cholesky'])
    cfg = SplitHMCConfig(chains=4, warmup=0, draws=2000, step_size=previous['step_size'],
        min_time=previous['config']['min_time'], max_time=previous['config']['max_time'],
        target_accept=.9, seed=previous['config']['seed'])
    began = time.perf_counter()
    chains = sample_split_hmc(target, model.state_dimension, cfg, resume=previous, progress=print)
    torch.save(chains, output/'chains.pt')
    result = summarize_chains(model, target, chains)
    write_json(output/'diagnostics.json', compact_diagnostics(result['diagnostics']))
    scores = score(model, target, chains, result, data, row, output)
    q = result['diagnostics']['quantities']
    write_json(output/'summary.json', {'kind': kind, 'case': row['case'], 'source_fingerprint_sha256': source_fingerprint(),
        'frozen_step_size': cfg.step_size, 'initial_retained_per_chain': 2000, 'retained_per_chain': 4000,
        'additional_adaptation': False, 'seed': cfg.seed, 'all_quantities_pass': result['diagnostics']['all_monitored_quantities_pass'],
        'worst_rhat': max(v['rank_folded_split_rhat'] or float('inf') for v in q.values()),
        'minimum_bulk_ess': min(v['bulk_ess'] or 0 for v in q.values()),
        'minimum_tail_ess': min(v['tail_ess'] or 0 for v in q.values()), 'scores': scores,
        'counts': chains['counts'], 'additional_seconds': time.perf_counter()-began,
        'qualification': 'supplementary fixed-budget extension of both independent kernels after borderline MCSE comparison; original results preserved, no model or benchmark selection'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['primary', 'small_step'], required=True)
    a = p.parse_args(); run(a.source, a.output, a.kind)
