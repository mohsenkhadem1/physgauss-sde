"""Describe actual deformation separately from rigid motion; development only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.sequence_metrics import proper_alignment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def census(path):
    with np.load(path) as data:
        rest = data['rest_nodes']; positions = rest[None] + data['u']
        weights = data['mass_row_sums']; weights = weights / weights.sum()
        times = data['times_s']; base_extent = np.ptp(rest, axis=0)
        nonrigid = []; aligned_extents = []
        for frame in positions:
            aligned, _, _ = proper_alignment(frame, rest, weights)
            nonrigid.append(float(np.sqrt(np.sum(weights[:, None]*(aligned-rest)**2))))
            aligned_extents.append(np.ptp(aligned, axis=0)/base_extent)
        extent = np.asarray(aligned_extents); nonrigid = np.asarray(nonrigid)
        com = (positions*weights[None, :, None]).sum(1)
        return dict(case=path.parent.name, trajectory_sha256=sha(path),
                    completed=json.loads(path.with_name('complete.json').read_text())['completed'],
                    frames=len(times), duration_s=float(times[-1]-times[0]),
                    max_nonrigid_point_rms_mm=float(nonrigid.max()*1000),
                    max_nonrigid_time_s=float(times[nonrigid.argmax()]),
                    min_rigid_aligned_bbox_ratio=extent.min(axis=0).tolist(),
                    max_rigid_aligned_bbox_ratio=extent.max(axis=0).tolist(),
                    com_displacement_from_rest_mm=float(np.linalg.norm(com-weights@rest, axis=1).max()*1000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='runs/compositional-dynamics-v17/refinement')
    parser.add_argument('--output', default='reports/compositional-dynamics-v17/deformation-census.json')
    args = parser.parse_args()
    paths = sorted(Path(args.input).glob('*/trajectory.npz'))
    if not paths:
        raise ValueError('No recorded trajectories')
    out = dict(scope='Development census only; no qualified truth/prediction claims. All available trajectory files retained, including failed prefixes. Aligned bounding boxes describe geometry, not local strain. Positive nodal weights are not full consistent-mass norm.',
               source_sha256={str(p): sha(p) for p in [Path(__file__), Path('scripts/compositional_dynamics_v17/sequence_metrics.py')]},
               rows=[census(path) for path in paths])
    Path(args.output).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(output=args.output, trajectories=len(paths))))


if __name__ == '__main__':
    main()
