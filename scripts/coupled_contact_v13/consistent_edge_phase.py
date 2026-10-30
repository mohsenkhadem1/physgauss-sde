"""Saved-frame development diagnosis; never a qualification substitute."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / 'reports/coupled-contact-v13/consistent-higher-refinement.json'
    data = json.loads(source.read_text())
    rows = []
    for run in data['runs']:
        if run['scenario'] != 'edge_drop' or not run['success']:
            continue
        path = ROOT / run['path'] / 'trajectory.npz'
        with np.load(path) as d:
            x, t = d['sampled'], d['times']
            rest = x[0] - x[0].mean(0)
            rotations = []
            for frame in x:
                u, _, vh = np.linalg.svd(rest.T @ (frame - frame.mean(0)))
                correction = np.eye(3)
                correction[-1, -1] = np.linalg.det(u @ vh)
                rotation = u @ correction @ vh
                rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))))
            contacts = []
            for index in [0, 1]:
                active = np.linalg.norm(d['reactions'][:, index], axis=1) > 1e-4
                contacts.append(dict(first_s=float(t[active][0]) if active.any() else None,
                                     last_s=float(t[active][-1]) if active.any() else None))
            rows.append(dict(key=run['key'], trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             times_s=t.tolist(), rotation_deg=rotations, contacts_floor_edge=contacts,
                             sample_mean_m=x.mean(1).tolist()))
    result = dict(scope='Development diagnosis at saved100Hz frames, excluding no cases. '
                  'Contact times quantized to saved frames; sample mean is not mass-weighted COM. '
                  'Proper Kabsch rotation separates bulk orientation from deformation. '
                  'These diagnostics do not replace spatial or temporal acceptance gates.',
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    output = ROOT / 'reports/coupled-surface-v13/consistent-edge-phase.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for r in rows:
        print(r['key'], r['contacts_floor_edge'], 'final_rotation_deg', round(r['rotation_deg'][-1], 3))


if __name__ == '__main__':
    main()
