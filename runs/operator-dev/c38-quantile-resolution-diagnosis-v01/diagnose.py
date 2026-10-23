"""Posthoc numerical diagnostic on sealed predictions; no labels/targets/Ys."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_prediction_evaluation as e

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
SEALS = [dict(path='runs/gaussian-campaign-v01/c38-initial-public-inference-v03/prediction-seal.json', sha256='4b8c8dbaed0a566d4cfd72a51a60921f2384be8b78c29e2b20d416b32fdc91e7'),
         dict(path='runs/gaussian-campaign-v01/c38-postprobe-inference-v01/prediction-seal.json', sha256='a8f78a344c4499c2261fe6a76f42e6ba6c401dc5d9a88fbe7419f14cb49bc8f8')]
if __name__ == '__main__':
    todo = []
    for phase, record in zip(('initial', 'down'), SEALS):
        seal = e.io.read_json(ROOT, record)
        for row in seal['objects']:
            if phase == 'down' and row['force_Fz_N'] != -2.:
                continue
            state = e.io.read_json(ROOT, row['result'])
            for model, saved in state['models'].items():
                for f in saved['files'].values():
                    e.io.checked(ROOT, f)
                todo.append((phase, model, saved['files']))
    assert len(todo) == 256
    stats = {}
    for phase, model, files in todo:
        pred = e.restore_prediction(ROOT, files)
        for method in ('adaptive', 'GL128'):
            row = pred['methods'][method]
            key = phase + '/' + model + '/' + method
            s = stats.setdefault(key, dict(predictions=0, continuous_quantiles=0,
                midpoint_CDF_enclosure_excludes_nominal_probability=0,
                midpoint_CDF_absolute_error_above_0p01=0, maximum_midpoint_CDF_absolute_error=0.,
                zero_width_nonconstant_point_intervals=0, maximum_root_bracket_width_m=0.))
            s['predictions'] += 1
            for entry in row['canonical_quantile_details']:
                quantiles = entry['quantiles']
                if quantiles[0]['value_m'] == quantiles[2]['value_m'] and any(q['kind'] == 'continuous_numerical_CDF_bracket' for q in quantiles):
                    s['zero_width_nonconstant_point_intervals'] += 1
                for p, q in zip((.025, .5, .975), quantiles):
                    if q['kind'] != 'continuous_numerical_CDF_bracket':
                        continue
                    c = q['CDF_at']
                    error = abs(c['cdf'] - p)
                    s['continuous_quantiles'] += 1
                    s['midpoint_CDF_enclosure_excludes_nominal_probability'] += int(not c['lower'] <= p <= c['upper'])
                    s['midpoint_CDF_absolute_error_above_0p01'] += int(error > .01)
                    s['maximum_midpoint_CDF_absolute_error'] = max(s['maximum_midpoint_CDF_absolute_error'], error)
                    s['maximum_root_bracket_width_m'] = max(s['maximum_root_bracket_width_m'], q['upper_m'] - q['lower_m'])
    result = dict(schema='C38-posthoc-quantile-resolution-diagnostic-v01', stats=stats,
                  prediction_seals=SEALS, private_labels_or_canonical_targets_or_observations_read=False,
                  actual_C38_or_C39_predictions_or_policies_modified=False,
                  scope='Numerical resolution diagnostic only; finite CDF midpoint error is expected at a positional stopping tolerance. It is not a new coverage estimate or a posthoc interval correction.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(stats))
