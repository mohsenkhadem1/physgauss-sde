"""Post-score deformation diagnostic for every frozen cohort object.

The existing independent finite-model oracle is evaluated only after all
decisions and scores are committed. Nothing feeds back into inference, action
selection, numerical gates, or cohort membership. No physical arrays published.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from experiments import campaign_diverse_force_cohort_v01 as cohort
from experiments.diagnose_c38_supported_motion import motion_components
from experiments.campaign_p2_keast_quadrature import positive_keast24

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'runs/operator-dev/diverse-force-cohort-v01/protocol.json'
DIGEST = '026aa14f924f02eed441c3d6e27d76ee9d88d02d5c5910170dd60767576c552c'
RUN = ROOT/'runs/gaussian-campaign-v01/diverse-force-cohort-v01'
OUT = ROOT/'runs/operator-dev/diverse-force-deformation-v01'


def run():
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != DIGEST:
        raise ValueError('Exact frozen cohort required')
    protocol = json.loads(PROTOCOL.read_text())
    dependencies = ['experiments/diagnose_c38_supported_motion.py',
                    'experiments/campaign_p2_keast_quadrature.py',
                    'experiments/campaign_compliant_force_challenge_v01.py']
    hashes = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in dependencies}
    summary = json.loads((RUN/'summary.json').read_text())
    if not summary['completed'] or summary['objects'] != 12 or not summary['all_objects_retained']:
        raise ValueError('All frozen outcomes must be committed first')
    for name, digest in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed frozen cohort source')
    for i in range(12):
        decision = json.loads((RUN/f'object-{i:02d}/decisions-before-target-truth.json').read_text())
        if decision['protocol_sha256'] != DIGEST:
            raise ValueError('Exact pre-score decisions required')
    OUT.mkdir(parents=True, exist_ok=False)
    X, tets, _ = cohort.old.original.tiny_geometry()
    bary, weights = positive_keast24()
    rest_dm = (X[tets[:, 1:]]-X[tets[:, :1]]).transpose(0, 2, 1)
    volumes = np.linalg.det(rest_dm)/6
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError('Positive finite rest volumes required')
    samples = np.einsum('qa,eai->eqi', bary, X[tets]).reshape(-1, 3)
    volume_weights = (volumes[:, None]*weights[None]).ravel()
    angle = .3
    Q = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    rigid = motion_components(samples, samples@Q+np.array([.01, .02, -.01]), volume_weights)
    if rigid['nonrigid_coordinate_RMS_m'] > 1e-13:
        raise ValueError('Proper rigid-registration control failed')
    rows = []
    for i, unit in enumerate(protocol['cohort_unit_coordinates_LOCAL']):
        theta = cohort.physical(np.asarray(unit)[None])[0]
        measurements = []
        minimum_j = float('inf')
        for force in cohort.old.force_programs().values():
            displacement = cohort.old.independent_numpy_forward(theta, force)
            for frame in cohort.TARGET_FRAMES:
                y = X+displacement[frame]
                deformed = np.einsum('qa,eai->eqi', bary, y[tets]).reshape(-1, 3)
                measurements.append(motion_components(samples, deformed, volume_weights))
                dm = (y[tets[:, 1:]]-y[tets[:, :1]]).transpose(0, 2, 1)
                minimum_j = min(minimum_j, float(np.linalg.det(dm@np.linalg.inv(rest_dm)).min()))
        total = float(np.sqrt(np.mean([m['total_coordinate_motion_RMS_m']**2 for m in measurements])))
        nonrigid = float(np.sqrt(np.mean([m['nonrigid_coordinate_RMS_m']**2 for m in measurements])))
        rows.append(dict(index=i, total_volume_coordinate_motion_RMS_m=total,
                         nonrigid_volume_coordinate_RMS_m=nonrigid,
                         nonrigid_to_total_RMS_ratio=nonrigid/total if total > 0 else None,
                         minimum_selected_state_J=minimum_j))
    report = dict(completed=True, objects=12, all_objects_retained=True,
                  cohort_protocol_SHA256=DIGEST,
                  cohort_summary_SHA256=hashlib.sha256((RUN/'summary.json').read_bytes()).hexdigest(),
                  source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependency_SHA256=hashes,
                  method='Proper volume-weighted rigid registration, positive exact polynomial quadrature of linear tet fields',
                  frames=list(cohort.TARGET_FRAMES), force_programs=list(cohort.old.force_programs()),
                  ratio_is_RMS_not_energy_fraction=True, rows=rows,
                  rigid_control_nonrigid_RMS_m=rigid['nonrigid_coordinate_RMS_m'],
                  no_new_inference_or_action_selection=True, no_objects_filtered=True,
                  independent_oracle_recomputed_post_score=True,
                  raw_parameters_and_states_local=True,
                  scope='Finite small-strain manufactured model; no nonlinear or real-material validation')
    path = OUT/'scalar-diagnostic.json'
    if hashes != {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in dependencies}:
        raise ValueError('Diagnostic dependencies changed during execution')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(objects=12, min_nonrigid_RMS_ratio=min(r['nonrigid_to_total_RMS_ratio'] for r in rows),
                         max_nonrigid_RMS_ratio=max(r['nonrigid_to_total_RMS_ratio'] for r in rows))))


if __name__ == '__main__':
    run()
