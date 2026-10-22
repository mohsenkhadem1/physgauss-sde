import numpy as np
import pytest
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_p2_support_trace import support_trace_certificate


@pytest.mark.parametrize('kind',['left_face','right_face','both_ends','bottom_face','separated_bottom_patches'])
def test_uniform_declared_faces_and_patch_areas_conform(kind):
    m=P2Geometry((4,2,2),support=kind);before=m.mask.copy()
    c=support_trace_certificate(m,kind)
    assert c['conforming'] and c['triangles_crossing_declared_patch_edges']==0
    np.testing.assert_array_equal(m.mask,before)


@pytest.mark.parametrize('mapping,cells',[('uniform',(6,2,2)),('clamp_graded',(8,4,4))])
def test_nonempty_node_mask_does_not_certify_shifted_continuous_patch(mapping,cells):
    m=P2Geometry(cells,support='separated_bottom_patches',mapping=mapping)
    assert m.support_masks[0].any() and m.support_masks[1].any()
    c=support_trace_certificate(m,'separated_bottom_patches')
    assert not c['conforming'] and c['triangles_crossing_declared_patch_edges']>0


def test_curved_reference_rejected_and_fullface_grading_allowed():
    m=P2Geometry((4,2,2),support='both_ends',mapping='clamp_graded')
    assert support_trace_certificate(m,'both_ends')['conforming']
    m.rest[m.elements[0,4],1]+=1e-6
    with pytest.raises(ValueError,match='Straight'):
        support_trace_certificate(m,'both_ends')
