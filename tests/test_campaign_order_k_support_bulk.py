from fractions import Fraction as Q
import pytest
from experiments.campaign_order_k_support_bulk import cell_region_weights,opposite_patch_joint_mass


def test_exact_conditional_cell_weights_keep_positive_sliver_without_normalization():
    tiny=Q(1,10**100);weights=cell_region_weights([tiny,1-tiny],1,7)
    assert weights[0]>0 and sum(weights)==Q(1,7)
    with pytest.raises(ValueError):cell_region_weights([Q(1,2)],1,7)
    with pytest.raises(ValueError):cell_region_weights([0,1],1,7)


def test_deterministic_opposite_factorization_has_only_one_face_probability():
    # Integrating independent two conditional-face partitions leaves1/F.
    a=[Q(1,3),Q(2,3)];b=[Q(1,4),Q(3,4)]
    total=sum(opposite_patch_joint_mass(6,x,y) for x in a for y in b)
    assert total==Q(1,6) and total!=Q(1,36)
    assert sum(total for _ in range(6))==1
