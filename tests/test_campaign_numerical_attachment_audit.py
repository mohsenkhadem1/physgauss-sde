import unittest
import numpy as np
from experiments.campaign_numerical_attachment_audit import attachment_descriptor


class AttachmentTests(unittest.TestCase):
    def test_descriptor_node_offset_pairing_is_permutation_invariant(self):
        p=np.array([[0.,0,0],[.01,0,0],[0,.02,0]])
        a=attachment_descriptor(p,0,[2,0,1],.001)
        b=attachment_descriptor(p,0,[0,1,2],.001)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(a['reference_local_offsets_m'],p)

    def test_same_center_different_reference_patch_changes_authored_inertia(self):
        p=np.array([[0.,0,0],[.01,0,0],[-.01,0,0],[0,.01,0],[0,-.01,0],[.02,0,0],[-.02,0,0]])
        a=attachment_descriptor(p,0,[0,1,2,3,4],.001)
        b=attachment_descriptor(p,0,[0,5,6,3,4],.001)
        self.assertEqual(a['center_m'],b['center_m'])
        self.assertNotEqual(a['authored_diagonal_inertia_kg_m2'],b['authored_diagonal_inertia_kg_m2'])
        self.assertEqual(a['handle_mass_kg'],b['handle_mass_kg'])


if __name__=='__main__':unittest.main()
