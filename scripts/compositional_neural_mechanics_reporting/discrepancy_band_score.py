"""Prospective prediction-only discrepancy supplement; never refits any model.

The validation-derived scalar MSE is not a calibrated stochastic error model.
Report coverage AND interval score, so width increases are not free successes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'reports/compositional-neural-mechanics-v16'
B = ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-reserved'
PLAN = R/'discrepancy-band-supplement-plan.json'
RESULT = R/'unknown-action-reserved-results.json'
AUDIT = R/'unknown-action-final-independent-audit.json'
VARIANCE = R/'model-discrepancy-variance-freeze.json'
Z90 = 1.6448536269514722


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def passed(record):
    return record.get('passed') is True or record.get('status') == 'passed'


def band_metrics(error, sd, variance):
    error, sd = np.asarray(error, float), np.asarray(sd, float)
    if error.shape != sd.shape or error.ndim != 1 or not error.size:
        raise ValueError('Matched nonempty coordinate arrays required')
    if not np.isfinite(error).all() or not np.isfinite(sd).all() or (sd < 0).any():
        raise ValueError('Finite errors/nonnegative standard deviations required')
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('Finite nonnegative fixed variance required')
    halfwidth = Z90*np.sqrt(sd*sd+variance)
    # Central 90% interval score: width + (2/alpha) * distance outside interval.
    interval_score = 2*halfwidth+20*np.maximum(np.abs(error)-halfwidth, 0)
    return dict(coverage=float(np.mean(np.abs(error) <= halfwidth)),
                mean_halfwidth_m=float(np.mean(halfwidth)),
                mean_interval_score_m=float(np.mean(interval_score)))


def prepare():
    if PLAN.exists() or RESULT.exists():
        raise ValueError('Prospective plan must precede reserved outcome report')
    if not passed(read(R/'model-discrepancy-final-independent-audit.json')):
        raise ValueError('Independent discrepancy audit required')
    variance = read(VARIANCE)
    if not variance['available']:
        raise ValueError('Completed validation variance freeze required')
    inputs = [VARIANCE, R/'model-discrepancy-final-independent-audit.json',
              R/'model-discrepancy-development.json', R/'unknown-action-method-freeze.json',
              Path(__file__).resolve(), ROOT/'tests/test_discrepancy_band_score_v16.py']
    document = dict(created_unix=time.time(), input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
                    variance_method='Use each method fixed scalar validation MSE, unchanged for every object/action/coordinate',
                    interval='Central nominal90%, zero mean discrepancy, sqrt(local_variance + fixed_MSE)',
                    metrics=['coverage', 'mean_halfwidth_m', 'mean_interval_score_m'],
                    rules=['Require primary independent audit before private truth access',
                           'All144 primary rows retained; no fit, mean, policy or primary band changes',
                           'No calibrated-posterior or parameter-bias-correction claim',
                           'Report interval score as well as coverage; increased width has a cost'],
                    unknown_result_absent_at_preparation=True)
    with PLAN.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')


def score():
    plan = read(PLAN)
    for name, digest in plan['input_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Prospectively fixed input changed: '+name)
    # All checks before reading any private arrays.
    if not AUDIT.exists() or not passed(read(AUDIT)):
        raise ValueError('Independent primary audit required before supplemental truth access')
    primary = read(RESULT)
    primary_audit = read(AUDIT)
    audit_evidence = primary_audit.get('evidence_sha256', {})
    if audit_evidence.get(str(RESULT.relative_to(ROOT))) != sha(RESULT):
        raise ValueError('Independent primary audit must bind exact outcome report')
    frozen = read(R/'unknown-action-method-freeze.json')
    if primary['plan_sha256'] != sha(R/'unknown-action-method-freeze.json'):
        raise ValueError('Primary plan mismatch')
    for name, digest in primary['evidence_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Primary scored evidence changed')
    expected = {(i, m, p, j) for i in range(6) for m in frozen['models']
                for p in ('initial', *frozen['policies']) for j in (0, 1)}
    keys = [(r['world'], r['model'], r['policy'], r['command']) for r in primary['rows']]
    if len(keys) != 144 or set(keys) != expected:
        raise ValueError('Exactly144 declared primary rows required')
    variances = read(VARIANCE)['method_variance_m2']
    rows = []
    for original in primary['rows']:
        row = {k:original[k] for k in ('world', 'model', 'policy', 'action', 'command', 'state')}
        if row['state'] != 'complete':
            row['error'] = original.get('error', 'Primary failed row')
            rows.append(row)
            continue
        world, model, action, command = row['world'], row['model'], row['action'], row['command']
        with np.load(B/f'world{world}'/model/'forecasts.npz', allow_pickle=False) as data:
            sl = slice(command*78, (command+1)*78)
            mean = data[action+'_mean'][sl].copy()
            sd = data[action+'_sd'][sl].copy()
        with np.load(B/f'world{world}/private/held-truth.npz', allow_pickle=False) as data:
            target = data[f'held{command}'].reshape(-1)
        if mean.shape != (78,) or sd.shape != (78,) or target.shape != (78,):
            raise ValueError('Exact78 coordinate forecast/reference required')
        error = mean-target
        primary_band = band_metrics(error, sd, 0.)
        extra_band = band_metrics(error, sd, variances[model])
        rmse = float(np.sqrt(np.mean(error*error)))
        if not np.isclose(rmse, original['position_coordinate_rmse_m'], rtol=1e-10, atol=1e-12):
            raise ValueError('Primary mean prediction changed')
        if primary_band['coverage'] != original['nominal90_coordinate_coverage']:
            raise ValueError('Primary coverage mismatch')
        if extra_band['coverage'] < primary_band['coverage']:
            raise ValueError('Variance addition cannot reduce central coverage')
        row.update(position_coordinate_rmse_m=rmse, primary=primary_band,
                   supplementary=extra_band, fixed_discrepancy_variance_m2=variances[model])
        rows.append(row)
    summaries = []
    for model in frozen['models']:
        for policy in ('initial', *frozen['policies']):
            selected = [r for r in rows if r['model'] == model and r['policy'] == policy]
            valid = [r for r in selected if r['state'] == 'complete']
            summary = dict(model=model, policy=policy, objects=6, expected_rows=12, complete_rows=len(valid))
            if len(valid) == 12:
                for kind in ('primary', 'supplementary'):
                    summary[kind] = {key:float(np.mean([r[kind][key] for r in valid]))
                                     for key in valid[0][kind]}
            summaries.append(summary)
    output = dict(plan_sha256=sha(PLAN), primary_sha256=sha(RESULT), primary_audit_sha256=sha(AUDIT),
                  variance_sha256=sha(VARIANCE), rows=rows, summaries=summaries,
                  scope=__doc__, primary_outputs_unchanged=True)
    with (R/'unknown-action-discrepancy-band-results.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--score', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else score()
