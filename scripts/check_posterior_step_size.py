"""Independent fixed-small-step replication of two development posteriors.

Follow-up numerical sensitivity check, not a new model or blind experiment.
500 fixed-kernel burn-in transitions, then 2,000 retained transitions per chain.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from physgauss.split_hmc import SplitHMCConfig, sample_split_hmc
from physgauss.white_posterior import WhitePosterior
from physgauss.study import write_json
from physgauss.cli import source_fingerprint
from converged_sparse_comparison import setup, summarize_chains, compact_diagnostics, score


def run(index, output):
    plan = json.loads(Path('configs/converged_sparse_comparison.json').read_text())
    case = plan['cases'][index]
    output = Path(output)/case; output.mkdir(parents=True, exist_ok=True)
    model, obs, fit, row, data, manifest = setup(case, output)
    step = {27: .2, 29: .15}[index]
    cfg = SplitHMCConfig(chains=4, warmup=0, draws=500, step_size=step,
        min_time=1., max_time=4., target_accept=.9, seed=108001+index)
    target = WhitePosterior(model, obs, fit['mean'], fit['precision_cholesky'])
    began = time.perf_counter()
    burn = sample_split_hmc(target, model.state_dimension, cfg, progress=print)
    cfg.draws = 2000
    chains = sample_split_hmc(target, model.state_dimension, cfg, resume=burn, progress=print)
    for key in ['white', 'log_ratio', 'energy_errors']:
        chains[key] = chains[key][:, 500:]
    chains['counts'] = {'warmup': burn['counts']['retained'], 'retained': {
        key: [a-b for a, b in zip(chains['counts']['retained'][key], burn['counts']['retained'][key])]
        for key in chains['counts']['retained']}}
    torch.save(chains, output/'chains.pt')
    result = summarize_chains(model, target, chains)
    write_json(output/'diagnostics.json', compact_diagnostics(result['diagnostics']))
    scores = score(model, target, chains, result, data, row, output)
    q = result['diagnostics']['quantities']
    write_json(output/'summary.json', {'case': case, 'source_fingerprint_sha256': source_fingerprint(),
        'fixed_step_size': step, 'burnin': 500, 'retained_per_chain': 2000, 'seed': cfg.seed,
        'all_quantities_pass': result['diagnostics']['all_monitored_quantities_pass'], 'scores': scores,
        'worst_rhat': max(v['rank_folded_split_rhat'] or float('inf') for v in q.values()),
        'minimum_bulk_ess': min(v['bulk_ess'] or 0 for v in q.values()),
        'minimum_tail_ess': min(v['tail_ess'] or 0 for v in q.values()),
        'counts': chains['counts'], 'seconds': time.perf_counter()-began,
        'qualification': 'independent fixed-small-step follow-up after preliminary proposal-error inspection; no diffusion/model retuning'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-index', type=int, choices=[27, 29], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.case_index, a.output)
