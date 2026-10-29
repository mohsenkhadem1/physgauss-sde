"""Read-only artifact checks; manuscript and video review remain explicit steps.

This never starts experiments, changes a frozen input, or declares the campaign
complete. It records missing/invalid deliverables rather than silently reducing
the declared study to the outputs currently available.
"""
from pathlib import Path
import json
import sys
import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE, OUT, ROOT
from scripts.position_ood.seal import verify, digest
from scripts.position_ood.campaign_jobs import jobs
from scripts.position_ood.force_law_study import verify as verify_force
from scripts.position_ood.force_law_report import paths as force_paths
from scripts.phystwin.sequential_policy import verify as verify_prior
from scripts.phystwin.sequential_reference import write


def finite_archive(path, expected_actions=None):
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            array = data[key]
            if array.dtype.kind in 'fci' and not np.isfinite(array).all():
                raise ValueError(f'Nonfinite array {path}:{key}')
        if 'predicted' in data:
            shape = data['predicted'].shape
            assert len(shape) == 3 and shape[1:] == (81, 3), (path, shape)
            if expected_actions is not None:
                assert shape[0] == expected_actions, (path, shape)


def hash_map(root, mapping):
    for name, expected in mapping.items():
        assert digest(root / name) == expected, f'Changed source: {name}'


def main():
    checks = []

    def check(name, function):
        try:
            detail = function()
            checks.append(dict(name=name, passed=True, detail=detail))
        except Exception as error:
            checks.append(dict(name=name, passed=False, detail=f'{type(error).__name__}: {error}'))

    check('primary_method_seal', lambda: verify()['created_utc'])
    check('force_law_method_seal', lambda: verify_force()['created_utc'])
    check('force_law_data_seal', lambda: hash_map(ROOT, json.loads((BASE/'force-law-data-seal.json').read_text())['hashes']))
    for name in ['runs/phystwin-informative-v08/seal.json',
                 'runs/phystwin-cost-aware-v07/seal.json',
                 'runs/phystwin-sequential-support-v06/analysis/seal.json',
                 'runs/phystwin-sequential-support-v06/analysis-rgbd/seal.json']:
        check(name, lambda name=name: verify_prior(json.loads((ROOT/name).read_text())))

    settings = json.loads((BASE/'method-seal.json').read_text())['settings']
    declared = jobs(settings)
    absent = [job['name'] for job in declared if not Path(job['result']).exists()]
    checks.append(dict(name='all_primary_job_outputs', passed=len(declared) == 394 and not absent,
                       detail=dict(declared=len(declared), ready=len(declared)-len(absent), missing=absent)))

    def inspect_outputs():
        count = 0
        for job in declared:
            path = Path(job['result'])
            if not path.exists():
                continue
            if path.suffix == '.npz':
                finite_archive(path)
            else:
                record = json.loads(path.read_text())
                if job['script'] == 'run_inverse.py':
                    finite_archive(path.with_suffix('.npz'), 10)
                elif job['script'] in ['phystwin.py', 'fit_mpm.py']:
                    finite_archive(path.parent/'predictions.npz', 10)
                if job['script'] in ['run_inverse.py', 'fit_mpm.py']:
                    assert np.isfinite(record['best']['loss']), path
            count += 1
        return dict(inspected=count, scope='Available outputs only; completeness is checked separately.')
    check('available_primary_outputs_parse_and_finite', inspect_outputs)

    def force_predictions():
        for i in range(6):
            for path in force_paths(i, settings).values():
                finite_archive(path, 16)
        report = json.loads((OUT/'force-law-results.json').read_text())
        assert report['complete'] and report['declared_objects'] == 6 and len(report['rows']) == 6
        assert len(report['local_prediction_sha256']) == 36
        hash_map(BASE, report['local_prediction_sha256'])
        return dict(objects=6, predictions=36)
    check('complete_force_law_comparison', force_predictions)

    for name in ['extrapolated-n3.json', 'transfer.json']:
        def calibration(name=name):
            data = json.loads((BASE/'calibration'/name).read_text())
            assert data['objects'] == 12
            hash_map(BASE, data['sources'])
            return dict(objects=data['objects'])
        check('frozen_calibration_'+name, calibration)

    def acquisition():
        data = json.loads((OUT/'transfer_test-active-results.json').read_text())
        assert data['complete'] and data['declared_objects'] == 9 and len(data['rows']) == 9
        assert data['matched_frame_ablation']['complete'] and len(data['matched_frame_ablation']['rows']) == 9
        assert [row['object'] for row in data['rows']] == list(range(9))
        summary = json.loads((OUT/'uncertainty-publication-summary.json').read_text())
        assert all(summary['active'][policy]['objects'] == 9 for policy in ['before', 'fixed', 'random', 'adaptive'])
        return dict(objects=9, policies=3, matched_frame_ablation=True)
    check('complete_acquisition_comparison', acquisition)

    for split, count in [('test_id', 18), ('test_material', 12), ('test_environment', 12)]:
        def cohort(split=split, count=count):
            for kind, expected in [('main', count), ('matched', 6)]:
                data = json.loads((OUT/f'{split}-{kind}-aggregate.json').read_text())
                assert data['declared_indices'] == list(range(expected))
                methods = {'extrapolated-n3-p3', 'neural-selected-id', 'neural-selected-all'}
                if kind == 'matched':
                    methods.update({'extrapolated_material_frame-n3-p3', 'extrapolated_nominal-n3-p3',
                                    'phystwin-'+settings['phystwin']['tag'],
                                    'physgaussian-'+settings['mpm']['tag']})
                assert {row['method'] for row in data['rows']} == methods
                assert all(row['complete'] and row['declared_objects'] == expected for row in data['rows'])
                assert all(group['objects'] == list(range(expected))
                           for row in data['rows'] for group in row['groups'].values())
            coverage = json.loads((OUT/f'{split}-extrapolated-coverage.json').read_text())
            assert coverage['objects'] == count and len(coverage['rows']) == count
            return dict(objects=count, matched_objects=6)
        check('complete_'+split, cohort)

    for filename, count in [('reference-summary.json', 87), ('force-law-reference-summary.json', 6)]:
        def references(filename=filename, count=count):
            report = json.loads((OUT/filename).read_text())
            assert len(report['rows']) == count
            assert len({(row.get('split', 'force_law_test'), row['object'])
                        for row in report['rows']}) == count
            for row in report['rows']:
                split = row.get('split', 'force_law_test')
                stem = f"{split}-{row['object']}"
                qualification = json.loads((BASE/f'qualification/{stem}.json').read_text())
                assert row['qualified'] and qualification['qualified']
                spatial = 1e6*max(qualification['spatial'][-1]['rms_m'])
                quadrature = 1e6*max(qualification['quadrature']['rms_m'])
                assert spatial < 20 and quadrature < 20
                assert np.isclose(spatial, row['spatial_rms_um'])
                assert np.isclose(quadrature, row['quadrature_rms_um'])
                with np.load(BASE/f'fields/{stem}.npz') as data:
                    actions = 16 if split == 'force_law_test' else (9 if split.startswith('transfer_') else 10)
                    assert data['field'].shape == (actions, 81, 3) and np.isfinite(data['field']).all()
                    assert data['minimum_J'].min() > .1
                    assert data['residual_N'].max() < 1e-9
                    assert data['reaction_balance_N'].max() < 1e-7
            return dict(qualified_fields=count)
        check('qualified_'+filename, references)

    paper = ROOT/'papers'/BASE.name
    names = ['main', 'matched', 'incremental', 'transfer', 'coverage', 'active', 'force-law']
    for name in names:
        check('table_'+name, lambda name=name: (paper/f'generated-{name}-table.tex').read_text() and 'present')
    for name in ['method', 'scenarios', 'qualitative', 'acquisition']:
        check('figure_'+name, lambda name=name: digest(paper/f'figures/{name}.pdf'))

    def videos():
        selection = json.loads((OUT/'qualitative-selection.json').read_text())['selected']
        cases = [(row['split'], row['object']) for row in selection] + [('force_law_test', 0), ('force_law_test', 3)]
        assert len(cases) == 4 and len(set(cases)) == 4
        for split, i in cases:
            path = OUT/f'{split}-{i}-actual-vs-predicted.mp4'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == meta['video_sha256']
            hash_map(BASE, meta['source_sha256'])
            assert meta['deformation_scale'] == 1 and meta['frames'] > 0
        return cases
    check('four_final_comparison_videos_and_provenance', videos)
    for name in ['paper.pdf', 'supplement.pdf']:
        check('final_'+name, lambda name=name: digest(paper/name))

    def placeholders():
        text = (paper/'main.tex').read_text() + (paper/'results.tex').read_text()
        markers = ['Draft status.', 'Quantitative conclusions in the final manuscript',
                   'are in progress', 'still being evaluated']
        present = [marker for marker in markers if marker in text]
        assert not present, present
        return 'Known interim placeholders removed; scientific prose still requires review.'
    check('interim_manuscript_placeholders_removed', placeholders)
    report = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  automatic_checks_pass=all(row['passed'] for row in checks), checks=checks,
                  remaining_manual_audit=['Check every manuscript claim against complete reports.',
                    'Render and inspect every final PDF page; verify main-paper page limit.',
                    'Decode and visually inspect all final videos, including scale and action labels.',
                    'Confirm final artifacts are committed and pushed; raw datasets remain local.',
                    'Stop campaign services at completion or the absolute deadline; preserve unrelated jobs.'],
                  scope='Automated artifact checks are necessary evidence, not proof of full scientific or visual completion.')
    write(OUT/'release-readiness.json', report)
    print(json.dumps(dict(automatic_checks_pass=report['automatic_checks_pass'],
                         passed=sum(row['passed'] for row in checks), total=len(checks),
                         incomplete=[row['name'] for row in checks if not row['passed']])))


if __name__ == '__main__':
    main()
