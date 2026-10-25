"""Saved, independently audited results only; no simulation/inference/RNG.

All methods retain the same eight objects and four noise repeats. This is a
fixed-grid diagnostic, not a population interval or learned-motion benchmark.
Figures remain local; compact scalar summaries can accompany the source.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/operator-dev/noisy-poke-diagnostics-media-v01'
POLICIES = ('before', 'adaptive', 'fixed', 'uniform_random')


def aggregate(rows):
    """Equal object means, with square roots taken after averaging loss/variance."""
    if len(rows) != 32 or sorted((r['index'], r['object_index'], r['replicate'])
                                for r in rows) != [(i, i % 8, i // 8) for i in range(32)]:
        raise ValueError('Every prescribed object/noise trial, with no exclusions')
    objects = []
    for obj in range(8):
        group = [r for r in rows if r['object_index'] == obj]
        values = {}
        for policy in POLICIES:
            pair = np.array([[r['metrics'][policy][k]
                              for k in ('MSE_m2', 'posterior_variance_m2')]
                             for r in group], dtype=float)
            if not np.isfinite(pair).all() or np.any(pair < 0):
                raise ValueError('Finite nonnegative errors and posterior variances')
            values[policy] = dict(zip(('MSE_m2', 'posterior_variance_m2'), pair.mean(axis=0).tolist()))
        objects.append(dict(object_index=obj, metrics=values,
                            adaptive_minus_fixed_MSE_m2=values['adaptive']['MSE_m2'] - values['fixed']['MSE_m2'],
                            adaptive_minus_random_MSE_m2=values['adaptive']['MSE_m2'] - values['uniform_random']['MSE_m2']))
    means = {}
    for policy in POLICIES:
        mse = float(np.mean([o['metrics'][policy]['MSE_m2'] for o in objects]))
        var = float(np.mean([o['metrics'][policy]['posterior_variance_m2'] for o in objects]))
        means[policy] = dict(MSE_m2=mse, posterior_variance_m2=var,
                             RMSE_m=float(np.sqrt(mse)), posterior_RMS_SD_m=float(np.sqrt(var)))
    return dict(policies=means, objects=objects, objects_retained=8, noise_repeats_per_object=4,
                all32_retained=True, population_or_calibration_claim=False)


def read(path):
    return json.loads(path.read_text())


def load():
    # Import guards and source validation execute only for actual postprocessing.
    from experiments import run_noisy_poke_benchmark_v01 as continuous
    from experiments import run_matched_finite_poke_control_v01 as finite
    from experiments import supervise_noisy_poke_pipeline_v01 as operator
    complete = read(operator.LOCAL / 'completed.json')
    if complete.get('completed') is not True:
        raise ValueError('Completed external operator required')
    continuous.verify(complete)
    ca = read(operator.exact(complete['noisy_audit']))
    fa = read(operator.exact(complete['finite_audit']))
    if ca.get('audit_passed') is not True or fa.get('audit_passed') is not True:
        raise ValueError('Both independent terminal audits required')
    if ca['summary']['comparison_blocked'] or fa['summary']['comparison_blocked']:
        raise ValueError('Unresolved comparisons stay blocked; no subset figure')
    continuous.verify(ca)
    continuous.verify(fa)
    sources = dict(continuous_audit=continuous.record(continuous.LOCAL / 'independent-terminal-review.json'),
                   finite_audit=finite.record(finite.LOCAL / 'independent-terminal-review.json'),
                   operator_complete=operator.record(operator.LOCAL / 'completed.json'))
    continuous_rows = []
    for rec in ca['all32_trial_records']:
        r = read(operator.exact(rec))
        if r.get('qualified') is not True:
            raise ValueError('No unresolved trial may enter the comparison')
        continuous_rows.append(dict(index=r['index'], object_index=r['object_index'], replicate=r['replicate'],
                                    metrics=dict(before=r['before']['field'],
                                                 **{p: r['policies'][p] for p in POLICIES[1:]})))
    evaluation = read(operator.exact(fa['source_evaluation']))
    models = {'Continuous linear': aggregate(continuous_rows)}
    for name, title in (('linear', 'Finite linear'), ('NH', 'Finite nonlinear')):
        rows = []
        for r in evaluation['rows']:
            if r['model'] != name:
                continue
            if r.get('choice_qualified') is not True:
                raise ValueError('No unresolved finite choice may enter the comparison')
            rows.append(dict(index=r['index'], object_index=r['object_index'], replicate=r['replicate'],
                             metrics=r['policies']))
        models[title] = aggregate(rows)
    return dict(models=models, source_commitments=sources,
                scope='Static held-out forces on eight fixed development objects, four noise repeats each. '
                      'Finite models share a uniform eight-state prior and known fixture stiffness; '
                      'the continuous prior differs. Error/uncertainty are descriptive, with no population claim.')


def draw(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(4)
    for col, (name, model) in enumerate(summary['models'].items()):
        top, lower = axes[:, col]
        top.bar(x - .18, [model['policies'][p]['RMSE_m'] * 1e6 for p in POLICIES],
                width=.36, label='Actual error (RMSE)', color='#2563a6')
        top.bar(x + .18, [model['policies'][p]['posterior_RMS_SD_m'] * 1e6 for p in POLICIES],
                width=.36, label='Reported uncertainty (RMS SD)', color='#d89028')
        top.set(title=name, ylabel='Displacement (µm)', xticks=x,
                xticklabels=['Before poke', 'Adaptive', 'Fixed (5)', 'Random'])
        top.tick_params(axis='x', rotation=20)
        top.legend(fontsize=8)
        for key, label, color in (('adaptive_minus_fixed_MSE_m2', 'Adaptive − fixed', '#2563a6'),
                                  ('adaptive_minus_random_MSE_m2', 'Adaptive − random', '#8063a6')):
            lower.plot(range(8), [o[key] * 1e12 for o in model['objects']],
                       marker='o', label=label, color=color)
        lower.axhline(0, color='#555555', linewidth=.8)
        lower.set(xlabel='Object (four fixture locations × two materials)',
                  ylabel='Paired MSE difference (µm²)\nNegative favours adaptive', xticks=range(8))
        lower.legend(fontsize=8)
    fig.suptitle('Static held-out prediction: eight fixed objects × four noise repeats\n'
                 'Finite pair shares its prior; comparison to continuous also changes the prior', fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if set(os.sched_getaffinity(0)) != {19} or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Saved-result plotting requires CPU19/noGPU')
    OUTPUT.mkdir(exist_ok=True)
    summary_path, image_path = OUTPUT / 'scalar-summary.json', OUTPUT / 'comparison.png'
    if summary_path.exists() or image_path.exists():
        raise FileExistsError('No result or presentation overwrite')
    summary = load()
    draw(summary, image_path)
    with summary_path.open('x') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(summary=str(summary_path.relative_to(ROOT)), image=str(image_path.relative_to(ROOT)))))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
