"""Report the separately frozen Gaussian family without selecting a winner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark_improvement_v14.common import REPORT, read, write, sha
from scripts.benchmark_improvement_v14.summarize import compare


def main():
    gaussian_path = REPORT / 'phystwin-gaussian-scores.json'
    primary_path = REPORT / 'phystwin-scores.json'
    gaussian, primary = read(gaussian_path), read(primary_path)
    frozen = read(REPORT / 'phystwin-gaussian-freeze.json')
    if gaussian['freeze_sha256'] != sha(REPORT / 'phystwin-gaussian-freeze.json'):
        raise ValueError('Gaussian score freeze differs from current frozen family')
    if primary['freeze_sha256'] != sha(REPORT / 'phystwin-freeze.json'):
        raise ValueError('Primary score freeze differs from frozen width family')
    if frozen['parent_width_freeze_sha256'] != primary['freeze_sha256']:
        raise ValueError('The two result families do not share the declared parent')
    expected = {case['name']: case for case in frozen['cases']}
    for record in (gaussian, primary):
        rows = record['cases']
        if len(rows) != len(expected) or {r['case']['name']: r['case'] for r in rows} != expected:
            raise ValueError('Exact eight-case census required for both families')
    old = {row['case']['name']: row for row in primary['cases']}
    rows = []
    for row in gaussian['cases']:
        parent = old[row['case']['name']]
        if row['metrics']['object_frame'] != parent['metrics']['object_frame']:
            raise ValueError('Original-model metrics differ between evaluators')
        methods = dict(parent['metrics'])
        methods['attachment_width_primary'] = methods['benchmark_v14']
        methods.update(row['metrics'])
        methods['benchmark_v14'] = row['metrics']['gaussian_v14']
        rows.append(dict(case=row['case'], metrics=methods,
                         archived_controls=parent.get('archived_controls', {})))
    methods = ['object_frame', 'attachment_width_primary', 'model_selection_selected',
               'official_phystwin', 'position_ridge']
    result = dict(
        scope=frozen['parameters']['scope'],
        gaussian_scores_sha256=sha(gaussian_path), primary_scores_sha256=sha(primary_path),
        freeze_sha256=gaussian['freeze_sha256'],
        comparisons={method: compare(rows, method, 'manual_track_mean_mm') for method in methods},
        coordinate_vs_original=compare(rows, 'object_frame', 'coordinate_rmse_mm'),
        shape_vs_original=compare(rows, 'object_frame', 'aligned_shape_rmse_mm'),
        fallback_cases=[r['case']['name'] for r in gaussian['cases'] if r['fallback_original']],
        decision='No post-score family or per-case selection; both frozen families remain separate.')
    write(REPORT / 'phystwin-gaussian-summary.json', result)
    for name, item in result['comparisons'].items():
        print(name, item.get('baseline_mean_mm'), '->', item.get('proposal_mean_mm'),
              'wins/losses', item.get('wins'), item.get('losses'),
              'complete', item.get('all_cases_and_frames_complete'))


if __name__ == '__main__':
    main()
