"""Report prospectively selected n4/n6 reference differences, not error bounds."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.held_appearance import closed_census
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.run_development import write


def compare(a, b):
    if not np.array_equal(a['material_queries'], b['material_queries']):
        raise ValueError('Common material samples required')
    if a['times_s'].shape != b['times_s'].shape or not np.allclose(a['times_s'], b['times_s'], atol=1e-10, rtol=0):
        raise ValueError('Identical complete physical clocks required')
    if a['surface_positions_m'].shape != b['surface_positions_m'].shape:
        raise ValueError('Identical sample shapes required')
    delta = a['surface_positions_m'] - b['surface_positions_m']
    if np.iscomplexobj(delta):
        if np.any(delta.imag != 0):
            raise ValueError('Nonreal reference displacement')
        delta = delta.real
    if not np.isfinite(delta).all() or len(delta) < 2:
        raise ValueError('Finite noninitial reference frames required')
    frame = 1000 * np.sqrt(np.mean(np.sum(delta ** 2, axis=-1), axis=(1, 2)))
    return dict(noninitial_vector_rms_mm=float(np.sqrt(np.mean(frame[1:] ** 2))),
                peak_frame_vector_rms_mm=float(frame.max()))


def audit(campaign):
    campaign = Path(campaign)
    closed_census(campaign)  # No reference targets before all predictions close.
    public = json.loads((campaign / 'public-manifest.json').read_text())
    rows = []
    for world in public['worlds']:
        if not world['refinement']:
            continue
        for index, query in enumerate(world['queries']):
            row = dict(world_id=world['world_id'], family=world['family'], query_index=index)
            rows.append(row)
            paths = [list((campaign / 'references' / world['world_id'] / slot).glob('*/complete.json'))
                     for slot in (str(index), f'{index}-refined')]
            if any(len(p) != 1 for p in paths):
                row['state'] = 'missing_or_pending'
                continue
            seals = [json.loads(p[0].read_text()) for p in paths]
            for n, seal in zip((4, 6), seals):
                if (seal['private_world_sha256'] != world['world_material_commitment']
                        or seal['public_query'] != query or seal['n'] != n):
                    raise ValueError('Reference world/query/resolution mismatch')
            row['complete_sha256'] = [sha(p[0]) for p in paths]
            if not all(s['completed'] for s in seals):
                row.update(state='reference_failed', errors=[s.get('error') for s in seals])
                continue
            trajectories = [p[0].parent / 'trajectory.npz' for p in paths]
            if any(sha(p) != s['trajectory_sha256'] for p, s in zip(trajectories, seals)):
                raise ValueError('Changed reference trajectory')
            with np.load(trajectories[0]) as a, np.load(trajectories[1]) as b:
                row.update(state='complete', **compare(a, b))
            row['minimum_dense_J'] = [s.get('minimum_dense_J') for s in seals]
            row['trajectory_sha256'] = [s['trajectory_sha256'] for s in seals]
    result = dict(scope='Prospective n4/n6 spatial and contact-sampling sensitivity on one world per family and both queries; these differences are not rigorous error bounds or population convergence evidence',
                  rows=rows, completed=all(r['state'] == 'complete' for r in rows) and bool(rows),
                  public_manifest_sha256=sha(campaign / 'public-manifest.json'))
    write(campaign / 'reference-sensitivity.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    audit(args.campaign)
