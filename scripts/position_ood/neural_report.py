"""Complete-cohort neural baseline analysis, independent of unfinished fits."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.aggregate import summarize
from scripts.position_ood.seal import verify, digest
from scripts.position_ood.study import BASE, OUT
from scripts.phystwin.sequential_reference import write


def main():
    verify()
    counts = dict(test_id=18, test_material=12, test_environment=12, transfer_test=9)
    sources = {}
    for split, count in counts.items():
        for i in range(count):
            reference = BASE / f'fields/{split}-{i}.npz'
            with np.load(reference) as data:
                shape = data['field'].shape
            sources[str(reference.relative_to(BASE))] = digest(reference)
            for mode in ['id', 'all']:
                path = BASE / f'neural-predictions/{split}-{i}-selected-{mode}.npz'
                with np.load(path) as data:
                    prediction = data['predicted']
                    if prediction.shape != shape or not np.isfinite(prediction).all():
                        raise RuntimeError(f'Invalid prediction: {path}')
                sources[str(path.relative_to(BASE))] = digest(path)
    cohorts = {split: summarize(split, ['neural-selected-all', 'neural-selected-id'])
               for split in counts}
    assert all(row['complete'] for report in cohorts.values() for row in report['rows'])
    report = dict(cohorts=cohorts, local_source_sha256=sources,
                  scope='Frozen neural baselines only. All-action and ID-action training remain separately reported; these results do not select a model or compare against unfinished physical fits. Transfer reuses nine familiar-parameter objects and is not an additional independent population.')
    write(OUT / 'neural-complete-results.json', report)
    for split, result in cohorts.items():
        group = 'spatial_transfer' if split == 'transfer_test' else 'new_interaction'
        print(split, group, {row['method']: row['groups'][group]['summary']['mean']
                             for row in result['rows']}, flush=True)


if __name__ == '__main__':
    main()
