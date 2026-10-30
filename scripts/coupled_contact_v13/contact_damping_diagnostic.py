"""Idealized contact-dashpot timestep estimate, not a nonlinear acceptance gate."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import json
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.coupled_contact_v13.engineered_scenarios import build
from scripts.coupled_contact_v13.consistent_mass import assemble_mass


def main():
    rows = []
    for cells in [4, 8, 12, 16]:
        system = build('edge_drop', 6000., .35, (cells,) * 3)
        mass = assemble_mass(system.rest, system.tets, system.material.density)
        triangles = system.surface_triangles
        points = system.rest[triangles]
        area = .5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
        contact = system.colliders[0]
        density = 2 * contact.damping_ratio * np.sqrt((contact.stiffness / .0004) * system.mass.sum() / area.sum())
        active = np.all(np.isclose(points[:, :, 2], system.rest[:, 2].min()), axis=1)
        local = (density * area[active] / 12)[:, None, None] * (np.eye(3) + np.ones((3, 3)))
        ids = triangles[active]
        damping = coo_matrix((local.ravel(), (np.broadcast_to(ids[:, :, None], local.shape).ravel(),
                                            np.broadcast_to(ids[:, None, :], local.shape).ravel())), shape=mass.shape).tocsc()
        value = float(eigsh(damping, k=1, M=mass, which='LM', return_eigenvectors=False,
                            v0=np.linspace(1., 2., len(system.rest)), tol=1e-7)[0])
        row = dict(cells=cells, full_flat_bottom_damping_lambda_per_s=value,
                   pure_damping_Euler_limit_us=2e6 / value)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(scope='Idealized pure-damping linear stability estimate for a fully active flat bottom face, '
                  'undeformed mesh and fixed normal. Exact P1 triangle surface damping matrix C and consistent '
                  'volume mass M; forward velocity Euler requires dt*lambda_max(M^-1 C)<2 in this isolated problem. '
                  'Not an exact stability bound for unilateral evolving contact, elastic stiffness or deformation, '
                  'and not an acceptance gate.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs=rows)
    (ROOT / 'reports/coupled-surface-v13/contact-damping-step-diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
