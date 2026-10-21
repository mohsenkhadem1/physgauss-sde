import copy
import unittest
from experiments.campaign_numerical_hardware_pair import replace_patch


class HardwarePairTests(unittest.TestCase):
    def test_replacement_preserves_every_other_spec_field_and_source(self):
        s={'load_nodes':[0,10],'load_patches':[list(range(9)),list(range(10,19))],
            'fixture_nodes':[99],'material':'unchanged','support_centers_m':[[1,2,3]]}
        c={'original_anchor_nodes':[0,10],'handle_index':1,'original_patch_sorted':list(range(10,19)),
            'replacement_patch_sorted':[10,11,12,13,14,15,16,17,20]}
        old=copy.deepcopy(s);new=replace_patch(s,c);self.assertEqual(s,old)
        new['load_patches'][1]=s['load_patches'][1];self.assertEqual(new,s)
        c['replacement_patch_sorted'][-1]=99
        with self.assertRaises(ValueError):replace_patch(s,c)


if __name__=='__main__':unittest.main()
