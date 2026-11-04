import pytest
import torch
from scripts.gaussian_boundary_flow_v02.feature_support import feature_support_diagnostic


def fixture():
    clean = torch.zeros(100, 6)
    clean[:, 3] = torch.linspace(-.01, .01, 100)
    basis = torch.zeros(6, 1); basis[3] = 1
    return dict(clean_features=clean, mean=clean.mean(0), basis=basis,
                scale=torch.tensor([.006]), frames=2, points=1, noise_sigma=.001)


def test_orthogonal_mismatch_with_small_retained_norm():
    args = fixture(); real = torch.zeros(6); real[4] = .05
    result = feature_support_diagnostic(observed_features=real, **args)
    assert result['out_of_bank_warning']
    assert not result['scores']['projected_standardized_norm']['flag']
    assert result['scores']['unrepresented_rms_m']['flag']
    assert 'not a calibrated OOD probability' in result['scope']


def test_projected_mismatch_and_central_screen():
    args = fixture(); real = torch.zeros(6); real[3] = .1
    result = feature_support_diagnostic(observed_features=real, **args)
    assert result['scores']['projected_standardized_norm']['flag']
    result = feature_support_diagnostic(observed_features=torch.zeros(6), **args)
    assert not result['out_of_bank_warning']
    assert result == feature_support_diagnostic(observed_features=torch.zeros(6), **args)


def test_noise_reference_has_zero_frame_zero_error_and_nonzero_tail():
    args = fixture(); args['noise_sigma'] = 0
    result = feature_support_diagnostic(observed_features=torch.zeros(6), **args)
    assert result['scores']['unrepresented_rms_m']['reference_max'] == 0
    args['noise_sigma'] = .001
    result = feature_support_diagnostic(observed_features=torch.zeros(6), **args)
    assert result['scores']['unrepresented_rms_m']['reference_quantile'] > .001


def test_mask_and_invalid_basis():
    args = fixture(); mask = torch.tensor([False, False, False, True, True, True])
    args['clean_features'] = args['clean_features'][:, mask]
    args['mean'] = args['mean'][mask]; args['basis'] = args['basis'][mask]
    result = feature_support_diagnostic(observed_features=torch.zeros(3), coordinate_mask=mask, **args)
    assert not result['out_of_bank_warning']
    args['basis'] *= 2
    with pytest.raises(ValueError, match='orthonormal'):
        feature_support_diagnostic(observed_features=torch.zeros(3), coordinate_mask=mask, **args)
