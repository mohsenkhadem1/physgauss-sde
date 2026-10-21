"""Build a local crossed-hypothesis prediction dataset without target selection.

All declared cells must be present. Public commands/hardware and supplied
physical hypotheses form features; simulated motion is a label only. This
engineering cohort is not a calibrated-inference or unseen-layout benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from campaign_hypothesis_inputs import known_hardware_from_public, validate_realized_hypothesis, support_translation
    from campaign_exciting_commands import make_exciting_commands
    from prepare_campaign_hybrid_pilot import bin_known_commands
except ImportError:
    from experiments.campaign_hypothesis_inputs import known_hardware_from_public, validate_realized_hypothesis, support_translation
    from experiments.campaign_exciting_commands import make_exciting_commands
    from experiments.prepare_campaign_hybrid_pilot import bin_known_commands

ROOT = Path(__file__).resolve().parents[1]
TIMES = np.arange(61, dtype=np.float64) / 20
ROLES = ('fit', 'development_force_transfer', 'unseen_middle_material')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or str(relative) != path.as_posix():
        raise ValueError('Canonical repository-relative path required')
    full = (ROOT / path).resolve()
    if not full.is_relative_to(ROOT):
        raise ValueError('Path escapes repository')
    return full


def read(path):
    return json.loads(Path(path).read_text())


def input_features(public, hypothesis, program_id):
    """No trajectory/observation/role/ID argument can enter these features."""
    known = known_hardware_from_public(public)
    spec = validate_realized_hypothesis(known, hypothesis)
    clock = make_exciting_commands(program_id)
    intervals = np.stack([clock.endpoint_times_s[:-1], clock.endpoint_times_s[1:]], -1)
    controls, _ = bin_known_commands(intervals, clock.handle_forces_world_N, TIMES)
    material = spec['material']
    mask = np.zeros(len(known.points), dtype=bool)
    mask[spec['fixture_nodes']] = True
    return {
        'rest': known.points.copy(),
        'controls': controls,
        'material': np.array([material['youngs_modulus_Pa'], material['density_kg_m3'],
                              material['poissons_ratio'], material['body_linear_drag_s_inverse']]),
        'support_centers': np.asarray(spec['support_origins_world_m']),
        'support_active': np.asarray(spec['support_active'], dtype=bool),
        'fixture_mask': mask,
        'support_offsets': np.stack([support_translation(spec, t) for t in TIMES]),
        'handle_patches': np.stack([h.ordered_patch_nodes for h in known.handles]),
        'handle_centers': np.stack([h.origin_world_m for h in known.handles]),
        'handle_masses': np.asarray([h.mass_kg for h in known.handles]),
        'handle_inertias': np.stack([h.com_inertia_body_kg_m2 for h in known.handles]),
        'handle_com_offsets': np.stack([h.com_offset_body_m for h in known.handles]),
        'handle_initial_rotations': np.stack([h.initial_rotation_body_to_world for h in known.handles]),
    }


def prepare(manifest_path, index_path, output):
    source = resolve(manifest_path)
    manifest = read(source)
    index = read(resolve(index_path))
    destination = resolve(output)
    if destination.exists():
        raise FileExistsError('Fresh local dataset directory required')
    if index['manifest_sha256'] != sha(source) or not index.get('completed'):
        raise ValueError('A complete batch with this exact input manifest is required')
    cells = manifest['cells']
    if len(cells) != 96 or len({c['id'] for c in cells}) != 96:
        raise ValueError('Exactly the declared 96-cell engineering cohort required')
    records = {r['id']: r for r in index['records']}
    if len(records) != len(index['records']) or set(records) != {c['id'] for c in cells}:
        raise ValueError('Missing, extra or duplicate batch cells')
    hardware = {h['id']: h for h in manifest['public_hardware']}
    hypotheses = {h['id']: h for h in manifest['hypotheses']}
    arrays = {}
    provenance = []
    tets = faces = None
    for cell in cells:
        record = records[cell['id']]
        if not record.get('completed') or cell['evaluation_role'] not in ROLES:
            raise ValueError('Failed cells cannot be dropped or relabeled')
        directory = resolve(record['output'])
        execution, summary = read(directory/'execution.json'), read(directory/'summary.json')
        if not execution['terminal'] or not execution['completed'] or not summary['completed'] or (directory/'failure.json').exists():
            raise ValueError('Incomplete or failed capture: '+cell['id'])
        protocol = read(directory/'protocol.json')
        if protocol['integration_hz'] != index['integration_hz']:
            raise ValueError('Common private integration rate required')
        public = hardware[cell['public_hardware_id']]
        spec = hypotheses[cell['hypothesis_id']]['spec']
        if read(directory/'public-hardware.json') != public or read(directory/'private-provenance/realized-hypothesis.json') != spec:
            raise ValueError('Executed public hardware or hypothesis differs from manifest')
        clock = make_exciting_commands(cell['program_id'])
        if read(directory/'public-command-manifest.json') != clock.public_manifest():
            raise ValueError('Executed public command program differs from manifest')
        trajectory = directory/'trajectory.npz'
        if sha(trajectory) != summary['trajectory_sha256']:
            raise ValueError('Trajectory bytes changed')
        with np.load(trajectory, allow_pickle=False) as raw:
            q = raw['positions_m'].copy()
            if (q.shape != (61, len(public['rest_points_m']), 3) or not np.isfinite(q).all()
                    or not np.array_equal(raw['times_s'], TIMES)
                    or np.max(np.abs(raw['actual_state_times_s']-TIMES)) > 1e-12):
                raise ValueError('Complete aligned finite 61-frame labels required')
        features = input_features(public, spec, cell['program_id'])
        if not np.array_equal(q[0], features['rest']):
            raise ValueError('Known-rest initial condition differs from target')
        known = known_hardware_from_public(public)
        if tets is None:
            tets, faces = known.tets.copy(), known.faces.copy()
        elif not np.array_equal(tets, known.tets) or not np.array_equal(faces, known.faces):
            raise ValueError('Shared-connectivity assumption failed')
        for key, value in {**features, 'q': q, 'ids': cell['id'],
                           'train_mask': cell['evaluation_role']=='fit',
                           'evaluation_role': cell['evaluation_role']}.items():
            arrays.setdefault(key, []).append(value)
        provenance.append({'id': cell['id'], 'output': record['output'],
                           'trajectory_sha256': summary['trajectory_sha256'],
                           'evaluation_role': cell['evaluation_role']})
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays.update(tets=tets, faces=faces, times=TIMES)
    counts = {role: int(np.sum(arrays['evaluation_role']==role)) for role in ROLES}
    if counts != {role: 32 for role in ROLES}:
        raise ValueError('Declared 32/32/32 roles changed')
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/'dataset.npz', **arrays)
    report = dict(schema='crossed-hypothesis-forward-dataset-v0.1',
        manifest_sha256=sha(source), index_sha256=sha(resolve(index_path)),
        source_builder_sha256=sha(__file__), dataset_sha256=sha(destination/'dataset.npz'),
        evaluation_role_counts=counts, records=provenance,
        arrays={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in arrays.items()},
        feature_allowlist=list(features)+['times','tets','faces'],
        label_only_fields=['q'], audit_only_fields=['ids','train_mask','evaluation_role'],
        material_order=['E_Pa','rho_kg_m3','nu','body_drag_s_inverse'],
        scope='Engineering supplied-hypothesis prediction; same four hardware contexts; three dependent joint material tuples.',
        limitations=['No unseen-layout claim', 'Three tuples do not identify four separate material effects',
                     'No sparse-image conditioning, posterior calibration or active-policy claim'],
        integration_hz_audit_only=index['integration_hz'], raw_dataset_LOCAL=True)
    (destination/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = prepare(args.manifest, args.index, args.output)
    print(json.dumps({k:report[k] for k in ['dataset_sha256','evaluation_role_counts']}))
