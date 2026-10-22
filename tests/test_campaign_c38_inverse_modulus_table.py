import numpy as np
import pytest
from experiments.campaign_c38_inverse_modulus_table import InverseModulusTable


def test_independent_quadratic_and_input_ownership():
    E = np.array([75000., np.sqrt(75000.*150000.), 150000.])
    a = np.arange(15.).reshape(5, 3) * 1e-4
    b = np.flip(a, axis=0)
    c = np.ones_like(a) * 2e-4
    values = np.array([a + b*(75000./e) + c*(75000./e)**2 for e in E])
    table = InverseModulusTable(E, values)
    query = np.array([75000.*2**.25, 75000.*2**.75])
    expected = np.array([a+b*(75000./e)+c*(75000./e)**2 for e in query])
    np.testing.assert_allclose(table.predict(query), expected, atol=2e-18, rtol=0)
    np.testing.assert_array_equal(table.predict(E), values)
    values[:] = 99
    E[:] = 1
    np.testing.assert_allclose(table.predict(query), expected, atol=2e-18, rtol=0)
    assert not table.young_Pa.flags.writeable and not table.displacement_m.flags.writeable


def test_failed_withheld_error_retained_without_refitting():
    E = np.array([75000., 100000., 150000.])
    direct = lambda e: np.full((2, 3), (75000./e)**3 * .1)
    table = InverseModulusTable(E, np.array([direct(e) for e in E]))
    before = table.displacement_m.copy()
    query = np.array([85000., 125000.])
    result = table.validate_withheld(query, np.array([direct(e) for e in query]), tolerance_m=1e-6)
    assert not result['passed'] and result['maximum_error_m'] > 1e-6
    np.testing.assert_array_equal(before, table.displacement_m)
    assert result['finite_validation_only'] and not result['validation_promoted_to_table']


def test_no_extrapolation_or_validation_node_reuse():
    table = InverseModulusTable([1., 2., 3.], np.zeros((3, 4, 3)))
    for q in [0., .99, 3.01, np.nan]:
        with pytest.raises(ValueError):
            table.predict(q)
    for query in [[1.], [2.], [3.], [1.5, 1.5]]:
        with pytest.raises(ValueError):
            table.validate_withheld(query, np.zeros((len(query), 4, 3)), tolerance_m=1e-6)


def test_euclidean_error_gate_not_coordinate_gate():
    table = InverseModulusTable([1., 2., 3.], np.zeros((3, 2, 3)))
    result = table.validate_withheld([1.5], np.full((1, 2, 3), .8e-6), tolerance_m=1e-6)
    assert not result['passed']
    assert result['maximum_error_m'] == pytest.approx(np.sqrt(3)*.8e-6)
