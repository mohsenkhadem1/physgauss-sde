"""Resume common-appearance evaluation only after the whole forecast census closes.

Run with the rendering environment on one explicitly selected GPU. Interrupted
render directories require inspection; completed images and scores are reused
only after their inputs and content hashes have been verified.
"""
import argparse
import json
from pathlib import Path

from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.run_development import write
from scripts.complex_scenes_v18.score_held import census, METHODS
from scripts.complex_scenes_v18.world_queries import Query
from scripts.complex_scenes_v18.world_aggregation import paired


def closed_census(campaign):
    rows = census(campaign)
    if any(row['state'] == 'pending' for row in rows):
        raise ValueError('Pending predictions: no held reference targets were opened')
    return rows


def images(folder, query, output):
    from scripts.complex_scenes_v18.shared_appearance import render, CHECKPOINT
    from scripts.complex_scenes_v18.score_shared_appearance import verified
    trajectory = folder / 'trajectory.npz'
    physical = json.loads((folder / 'complete.json').read_text())
    if not physical['completed'] or sha(trajectory) != physical['trajectory_sha256']:
        raise ValueError('Unsealed or changed physical trajectory')
    seal_path = output / 'images-seal.json'
    if output.exists():
        if not seal_path.exists():
            raise ValueError(f'Interrupted image directory requires review: {output}')
        seal = verified(output)
        if (seal['trajectory_sha256'] != sha(trajectory)
                or seal['public_query'] != query
                or seal['appearance_sha256'] != sha(CHECKPOINT)):
            raise ValueError('Existing render does not match sealed inputs')
    else:
        render(trajectory, Query(**query), output, frame_count=7)
    return seal_path


def evaluate(campaign):
    campaign = Path(campaign)
    rows = closed_census(campaign)
    from scripts.complex_scenes_v18.score_shared_appearance import score
    for row in rows:
        if row['state'] != 'sealed':
            continue
        world, index = row['world_id'], row['query_index']
        refs = list((campaign / 'references' / world / str(index)).glob('*/complete.json'))
        if len(refs) != 1:
            raise ValueError('Missing or ambiguous declared reference')
        reference = json.loads(refs[0].read_text())
        if (reference['private_world_sha256'] != row['world_material_commitment']
                or reference['public_query'] != row['public_query']):
            raise ValueError('Reference world/query mismatch')
        if not reference['completed']:
            row['state'] = 'reference_invalid'
            continue
        root = campaign / 'appearance' / world / f'query{index}'
        ref_folder = root / 'reference'
        pred_folder = root / row['method']
        ref_seal = images(refs[0].parent, row['public_query'], ref_folder)
        pred_seal = images(Path(row['physical_output']), row['public_query'], pred_folder)
        report_path = root / f"{row['method']}-score.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if (report['prediction_images_seal_sha256'] != sha(pred_seal)
                    or report['reference_images_seal_sha256'] != sha(ref_seal)):
                raise ValueError('Existing appearance score has changed inputs')
        else:
            report = score(pred_folder, ref_folder)
            write(report_path, report)
        row.update(state='complete', metrics=report['means'], appearance_report_sha256=sha(report_path))
        print(json.dumps(dict(world=world, query=index, method=row['method'], metrics=row['metrics'])), flush=True)
    aggregates = []
    for family in [None] + sorted({r['family'] for r in rows}):
        selected = rows if family is None else [r for r in rows if r['family'] == family]
        for baseline in METHODS[1:]:
            for metric in ('lpips_alex_v01', 'psnr_foreground_union_db', 'ssim_rgb_foreground_union'):
                result = paired(selected, 'ours', baseline, metric=metric)
                result.update(family=family or 'all', lower_is_better=metric == 'lpips_alex_v01')
                result['scope'] = ('Conditional on both methods completing both queries; differences are ours minus baseline; '
                                   'negative favors ours for LPIPS, positive for PSNR/SSIM; bootstrap resamples worlds')
                aggregates.append(result)
    result = dict(scope='Shared object-only appearance and motion transport; noninitial frames; not native end-to-end reconstruction',
                  rows=rows, paired_world_aggregates=aggregates,
                  public_manifest_sha256=sha(campaign / 'public-manifest.json'))
    write(campaign / 'appearance-results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    evaluate(args.campaign)
