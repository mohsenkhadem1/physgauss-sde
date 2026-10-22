import copy
import numpy as np
import pytest
from experiments.campaign_c38_table_validation import validate_finite_table, POINTS
from experiments.campaign_c38_nonlinear_table_catalog import catalog


def fields():
    rows = []
    for c in catalog():
        x = 75000./c['young_Pa']
        scale = 1. if c['support'] == 'both_ends' else 1.3
        u = POINTS * (.01*x + .002*x*x) * scale * (c['Fz_N'] + 4.)
        for nx in (32, 40):
            rows.append(dict(case=c, cells=[nx, nx//2, nx//2], material_probes_m=POINTS,
                             probe_positions_m=POINTS+u, mechanics_passed=True,
                             finite_spatial_passed=True))
    return rows


def test_independent_quadratic_all_cases_both_meshes():
    result = validate_finite_table(fields(), tolerance_m=1e-6)
    assert len(result['checks']) == 16 and result['finite_checks_passed']
    assert result['maximum_withheld_point_error_m'] < 1e-15
    assert not result['continuous_material_error_bound_established']
    assert not result['dataset_admitted']


def test_interpolation_failure_retained_and_not_refitted():
    rows = fields()
    index = next(i for i,r in enumerate(rows) if r['case']['material_use'].startswith('WITHHELD'))
    rows[index]['probe_positions_m'] = rows[index]['probe_positions_m'] + np.array([2e-6, 0, 0])
    before = rows[index]['probe_positions_m'].copy()
    result = validate_finite_table(rows, tolerance_m=1e-6)
    assert not result['finite_checks_passed'] and not result['all_withheld_interpolation_checks_passed']
    np.testing.assert_array_equal(rows[index]['probe_positions_m'], before)


def test_physical_failure_is_not_hidden_by_exact_interpolation():
    rows = fields(); rows[0]['finite_spatial_passed'] = False
    result = validate_finite_table(rows, tolerance_m=1e-6)
    assert result['all_withheld_interpolation_checks_passed']
    assert not result['finite_checks_passed']


def test_missing_duplicate_or_wrong_point_binding_rejected():
    rows = fields()
    with pytest.raises(ValueError):
        validate_finite_table(rows[:-1], tolerance_m=1e-6)
    duplicate = list(rows); duplicate[-1] = rows[0]
    with pytest.raises(ValueError):
        validate_finite_table(duplicate, tolerance_m=1e-6)
    wrong = copy.deepcopy(rows); wrong[0]['material_probes_m'][0,0] += 1e-7
    with pytest.raises(ValueError):
        validate_finite_table(wrong, tolerance_m=1e-6)
