"""Complete-cohort analysis for the independent force-law OOD extension."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.force_law_study import BASE, ROOT, OUT, SPLIT, verify, seal_data, DATA_SEAL
from scripts.position_ood.seal import verify as verify_main, digest
from scripts.position_ood.score import metrics, paired_interval, deformation
from scripts.position_ood.study import within_deadline
from scripts.phystwin.sequential_reference import write

LABELS = {'extrapolated':'Spatial inverse (ours)', 'extrapolated_material_frame':'Material-indexed load',
          'neural_id':'Neural (ID actions)', 'neural_all':'Neural (all actions)',
          'phystwin':'PhysTwin core adaptation', 'mpm':'PhysGaussian MPM adaptation'}


def paths(i, settings):
    result = {m:BASE/f'inverse/{SPLIT}-{i}/{m}-n3-p3.npz'
              for m in ['extrapolated', 'extrapolated_material_frame']}
    result.update({f'neural_{mode}':BASE/f'neural-predictions/{SPLIT}-{i}-selected-{mode}.npz'
                   for mode in ['id', 'all']})
    result['phystwin'] = BASE/f'phystwin/{SPLIT}-{i}/{settings["phystwin"]["tag"]}/predictions.npz'
    result['mpm'] = BASE/f'{settings["mpm"]["tag"]}/{SPLIT}-{i}/predictions.npz'
    return result


def main(wait=False):
    plan = verify()
    settings = verify_main()['settings']
    while True:
        missing = [str(p.relative_to(BASE)) for i in range(6) for p in paths(i, settings).values() if not p.exists()]
        if DATA_SEAL.exists() and not missing:
            break
        if not wait:
            raise RuntimeError(f'Complete six-object prediction cohort required; {len(missing)} prediction files missing')
        within_deadline()
        time.sleep(60)
    seal_data()
    rows = []
    sources = {}
    for i, case in enumerate(plan[SPLIT]):
        with np.load(BASE/f'fields/{SPLIT}-{i}.npz') as data:
            truth = data['field'].copy()
        row = dict(object=i, force_law=case['force_law']['kind'], methods={}, deformation=deformation(truth))
        for method, path in paths(i, settings).items():
            with np.load(path) as data:
                prediction = data['predicted'].copy()
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f'Invalid prediction: {path}')
            row['methods'][method] = metrics(prediction, truth)
            sources[str(path.relative_to(BASE))] = digest(path)
        rows.append(row)
    cohorts = {}
    for name, indices in [('anisotropic_harmonic', [0, 1, 2]), ('two_gaussian', [3, 4, 5]), ('pooled', list(range(6)))]:
        records = {}
        for method in plan['methods']:
            records[method] = {}
            for group, actions in plan['groups'].items():
                values = {metric:[float(np.mean(np.asarray(rows[i]['methods'][method][metric])[actions])) for i in indices]
                          for metric in ['coordinate_rmse_mm', 'incremental_coordinate_rmse_mm', 'hidden_surface_rmse_mm']}
                records[method][group] = {metric:dict(per_object=v, summary=paired_interval(v)) for metric, v in values.items()}
        comparisons = {}
        for method in plan['methods'][1:]:
            comparisons[method] = {}
            for group in plan['groups']:
                a = np.asarray(records['extrapolated'][group]['coordinate_rmse_mm']['per_object'])
                b = np.asarray(records[method][group]['coordinate_rmse_mm']['per_object'])
                comparisons[method][group] = paired_interval(b-a)
        cohorts[name] = dict(objects=indices, methods=records, paired_advantage_mm=comparisons)
    report = dict(complete=True, declared_objects=6, rows=rows, cohorts=cohorts,
                  groups=plan['groups'], local_prediction_sha256=sources,
                  scope=plan['scope'], aggregation='Equal-weight mean of action coordinate RMSE within each object, then objects. Positive paired advantage favors the spatial inverse. Three objects per force law: intervals are descriptive and do not establish broad family-level generalization.')
    write(OUT/'force-law-results.json', report)
    lines = ['\\begin{table*}[t]', '\\centering\\small', '\\begin{tabular}{lrrrr}', '\\toprule',
             '& \\multicolumn{2}{c}{Anisotropic harmonic field} & \\multicolumn{2}{c}{Two-Gaussian field} \\\\',
             'Method & New motions & Spatial transfer & New motions & Spatial transfer \\\\', '\\midrule']
    for method in plan['methods']:
        values = [cohorts[law]['methods'][method][group]['coordinate_rmse_mm']['summary']['mean']
                  for law in ['anisotropic_harmonic', 'two_gaussian'] for group in ['new_interaction', 'spatial_transfer']]
        lines.append(LABELS[method]+' & '+' & '.join(f'{v:.3f}' for v in values)+' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}',
              '\\caption{Unseen environmental force laws: coordinate RMSE (mm) on a separately predeclared six-object extension, three objects per law. All methods use their frozen settings and the same original three observations. Material coefficients remain unknown but are drawn within the familiar range. No model is retrained on these force laws.}',
              '\\label{tab:force-law}', '\\end{table*}']
    (ROOT/'papers'/BASE.name/'generated-force-law-table.tex').write_text('\n'.join(lines)+'\n')
    visual_plan = json.loads((OUT/'force-law-visual-plan.json').read_text())
    for i in visual_plan['objects']:
        # Rendering stored predictions is reproducible analysis, not a new fit.
        # Only waiting for unfinished experiments above is deadline bounded.
        subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'), str(ROOT/'scripts/position_ood/render.py'),
                        '--split', SPLIT, '--object', str(i), '--phystwin-tag', settings['phystwin']['tag'],
                        '--mpm-tag', settings['mpm']['tag']], check=True, cwd=ROOT)
    print('Complete force-law report generated', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
