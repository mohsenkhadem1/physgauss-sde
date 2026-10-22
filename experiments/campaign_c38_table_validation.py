"""Pure finite-point validation of the predeclared 40-state response table.

No files, simulations, noise or observations are opened/generated here. The
calling executable must verify source, case metadata, field hashes and role
identities before constructing rows. No validation material is refitted.
"""
import numpy as np
from experiments.campaign_c38_nonlinear_table_catalog import catalog
from experiments.campaign_c38_inverse_modulus_table import InverseModulusTable

POINTS = np.stack(np.meshgrid(np.linspace(0, .24, 7), np.linspace(-.04, .04, 3),
                             np.linspace(-.04, .04, 3), indexing='ij'), axis=-1).reshape(-1, 3)


def validate_finite_table(rows, *, tolerance_m):
    expected = {c['id']: c for c in catalog()}
    if len(rows) != 80:
        raise ValueError('All 80 declared mesh fields required; no filtering')
    fields = {}
    qualifications = []
    for row in rows:
        c = row['case']
        if c['id'] not in expected or any(c[k] != expected[c['id']][k] for k in ('young_Pa', 'support', 'Fz_N')):
            raise ValueError('Exact case/material/support/force binding required')
        n = row['cells']
        if list(n) not in ([32, 16, 16], [40, 20, 20]):
            raise ValueError('Exact predeclared mesh pair required')
        key = (c['support'], c['Fz_N'], c['young_Pa'], n[0])
        if key in fields:
            raise ValueError('Duplicate physical case/resolution field')
        points = np.asarray(row['material_probes_m'])
        q = np.asarray(row['probe_positions_m'])
        if not np.array_equal(points, POINTS) or q.shape != POINTS.shape or not np.isfinite(q).all():
            raise ValueError('Exact qualified 63-point binding and finite fields required')
        fields[key] = q - POINTS
        if type(row['mechanics_passed']) is not bool or type(row['finite_spatial_passed']) is not bool:
            raise ValueError('Explicit original physical/spatial outcomes required')
        qualifications.append(row['mechanics_passed'] and row['finite_spatial_passed'])
    all_cases = list(expected.values())
    nodes = sorted({c['young_Pa'] for c in all_cases if c['material_use'] == 'TABLE_NODE'})
    withheld = sorted({c['young_Pa'] for c in all_cases if c['material_use'] == 'WITHHELD_INTERPOLATION_VALIDATION'})
    checks = []
    for support in sorted({c['support'] for c in all_cases}):
        for force in sorted({c['Fz_N'] for c in all_cases}):
            for nx in (32, 40):
                table = InverseModulusTable(nodes, np.array([fields[(support, force, E, nx)] for E in nodes]))
                result = table.validate_withheld(withheld, np.array([fields[(support, force, E, nx)] for E in withheld]), tolerance_m=tolerance_m)
                perpoint = result.pop('per_point_error_m')
                checks.append(dict(support=support, Fz_N=force, cells=[nx, nx//2, nx//2],
                                   worst_reference_point_ids=perpoint.argmax(axis=1).tolist(), **result))
    return dict(checks=checks, direct_mesh_fields=80, validation_materials=2,
                all_direct_finite_physical_and_spatial_checks_passed=all(qualifications),
                all_withheld_interpolation_checks_passed=all(c['passed'] for c in checks),
                finite_checks_passed=all(qualifications) and all(c['passed'] for c in checks),
                tolerance_m=float(tolerance_m),
                maximum_withheld_point_error_m=max(c['maximum_error_m'] for c in checks),
                continuous_material_error_bound_established=False,
                observation_noise_drawn=False, posterior_computed=False, dataset_admitted=False)
