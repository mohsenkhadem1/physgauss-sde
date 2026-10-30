"""Locate saved-frame strain energy around the sharp support, without fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / 'reports/coupled-contact-v13/consistent-higher-refinement.json'
    payload = path.read_bytes()
    report = json.loads(payload)
    rows = []
    for run in report['runs']:
        if run['scenario'] != 'edge_drop' or run['spec'][-1] not in (8, 12, 16) or not run['success']:
            continue
        file = ROOT / run['path'] / 'trajectory.npz'
        with np.load(file) as d:
            rest, tets = d['rest'], d['tets']
            dm = (rest[tets[:, 1:]]-rest[tets[:, 0, None]]).transpose(0, 2, 1)
            volume = np.linalg.det(dm)/6
            inverse = np.linalg.inv(dm)
            young, nu = run['young'], .3
            mu, lam = young/(2*(1+nu)), young*nu/((1+nu)*(1-2*nu))
            frames = []
            for target in (.06, .08, .10, .12, .15, .20):
                index = int(np.argmin(abs(d['times']-target)))
                x = d['frames'][index]
                f = (x[tets[:, 1:]]-x[tets[:, 0, None]]).transpose(0, 2, 1) @ inverse
                j = np.linalg.det(f)
                logj = np.log(j)
                density = .5*mu*((f*f).sum((1, 2))-3)-mu*logj+.5*lam*logj**2
                energy = density*volume
                centroid = x[tets].mean(1)
                # Distance to either finite top-edge segment of the fixed rail.
                dx = np.minimum(abs(centroid[:, 0]-.002), abs(centroid[:, 0]-.018))
                dy = np.maximum(abs(centroid[:, 1])-.07, 0.)
                distance = np.sqrt(dx*dx+dy*dy+(centroid[:, 2]-.05)**2)
                near = distance <= .003
                total = float(energy.sum())
                frames.append(dict(time_s=float(d['times'][index]), elastic_energy_j=total,
                    min_saved_J=float(j.min()), max_energy_density_J_per_m3=float(density.max()),
                    p95_energy_density_J_per_m3=float(np.percentile(density, 95)),
                    near_corner_rest_volume_fraction=float(volume[near].sum()/volume.sum()),
                    near_corner_energy_fraction=float(energy[near].sum()/total) if total>1e-15 else None))
            rows.append(dict(key=run['key'], trajectory_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), frames=frames))
    out = dict(scope='Post-hoc saved-frame development diagnostic, not a convergence or singularity proof. '
        'Compressible Neo-Hookean elastic energy only; no contact potential or viscous work. '
        'Near-corner means current tetrahedron centroid within3mm of either finite top-edge segment. '
        'Centroid classification is resolution dependent; energy and rest-volume fractions are both shown.',
        input_sha256=hashlib.sha256(payload).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    (ROOT/'reports/coupled-surface-v13/edge-strain-localization.json').write_text(json.dumps(out, indent=2)+'\n')
    for r in rows:
        print(r['key'], [(f['time_s'], round(f['elastic_energy_j'], 5), round(f['max_energy_density_J_per_m3'])) for f in r['frames'][:3]])


if __name__ == '__main__':
    main()
