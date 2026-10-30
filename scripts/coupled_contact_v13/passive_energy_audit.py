"""Saved-frame passive-drop energy sanity check, not a conservation proof."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.coupled_contact_v13.engineered_scenarios import build
from scripts.coupled_contact_v13.consistent_mass import assemble_mass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass', choices=['lumped', 'consistent'], required=True)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records, sources = [], {}
    for name in args.input:
        path = ROOT / name
        sources[name] = digest(path)
        records.extend(json.loads(path.read_text())['runs'])
    seen, rows = set(), []
    for record in records:
        if not record['success'] or record['spec'][0] != 'edge_drop' or record['path'] in seen:
            continue
        seen.add(record['path'])
        name, young, friction, cells = record['spec'][:4]
        path = ROOT / record['path'] / 'trajectory.npz'
        system = build(name, young, friction, (cells,) * 3)
        matrix = assemble_mass(system.rest, system.tets, system.material.density) if args.mass == 'consistent' else None
        with np.load(path) as data:
            if not np.array_equal(data['rest'], system.rest) or not np.array_equal(data['tets'], system.tets):
                raise ValueError('Saved geometry differs from declared engineered scene')
            energy = []
            for position, velocity in zip(data['frames'], data['velocities']):
                momentum = matrix @ velocity if matrix is not None else system.mass[:, None] * velocity
                energy.append([system.elastic_energy(position), .5 * float(np.sum(velocity * momentum)),
                               float(np.sum(system.mass * position[:, 2]) * 9.81)])
        energy = np.asarray(energy)
        if not np.isfinite(energy).all():
            raise ValueError('Nonfinite saved energy')
        total = energy.sum(axis=1)
        rows.append(dict(spec=record['spec'], trajectory_sha256=digest(path),
                         initial_mechanical_J=float(total[0]),
                         maximum_mechanical_excluding_contact_J=float(total.max()),
                         final_elastic_kinetic_gravitational_J=energy[-1].tolist(),
                         maximum_excess_over_initial_J=float(max(0, total.max() - total[0]))))
    if not rows:
        raise ValueError('No completed passive edge-drop records')
    result = dict(scope='Post-hoc saved-frame sanity only: elastic plus kinetic plus gravitational energy. '
                  'Positive contact storage omitted; no full energy balance, per-step monotonicity or exact conservation claim. '
                  'The engineered edge-drop has static colliders and no moving tool work.',
                  mass=args.mass, source_report_sha256=sources, auditor_sha256=digest(Path(__file__)),
                  runs=rows)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(runs=len(rows), maximum_excess_J=max(r['maximum_excess_over_initial_J'] for r in rows))))


if __name__ == '__main__':
    main()
