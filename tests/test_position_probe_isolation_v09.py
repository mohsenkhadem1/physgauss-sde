import pytest
from scripts.position_ood.run_inverse import validate_prefix

def test_local_and_spatial_targets_cannot_be_acquired_by_accident():
    validate_prefix('test_id',(0,1,2,7))
    validate_prefix('transfer_test',(0,1,2,8))
    with pytest.raises(ValueError):validate_prefix('test_id',(0,1,2,9))
    with pytest.raises(ValueError):validate_prefix('transfer_test',(0,1,2,7))
    with pytest.raises(ValueError):validate_prefix('transfer_test',(0,1,2,4,5))
