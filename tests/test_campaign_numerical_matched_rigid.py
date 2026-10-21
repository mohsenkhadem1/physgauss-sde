import unittest
import numpy as np
from experiments.campaign_numerical_matched_rigid import verify_native_material,verify_native_rest

class NativeRuntimeVerificationTests(unittest.TestCase):
    def test_native_material_mismatch_is_not_silently_replaced_with_usd(self):
        material={'youngs_modulus_Pa':4249.367739516093,'poissons_ratio':.3407184360170016}
        e=np.array([[material['youngs_modulus_Pa']]],np.float32);nu=np.array([[material['poissons_ratio']]],np.float32)
        self.assertEqual(verify_native_material(e,nu,material)['youngs_modulus_Pa']['native'],float(e[0,0]))
        for bad_e,bad_nu in [(e*100,nu),(e,nu*.5),(np.array([[np.nan]],np.float32),nu),(np.zeros((0,1),np.float32),nu)]:
            with self.assertRaises(ValueError):verify_native_material(bad_e,bad_nu,material)

    def test_rest_buffers_allow_cooking_element_reorder_but_reject_changed_reference(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float)
        t=np.array([[0,1,2,3],[1,2,3,4]],int)
        self.assertTrue(verify_native_rest(p[None].astype(np.float32),t[::-1][None],p,t)['oriented_tet_multiset_matches'])
        bad=p[None].astype(np.float32);bad[0,1,0]+=1e-4
        with self.assertRaises(AssertionError):verify_native_rest(bad,t[None],p,t)
        reflected=t.copy();reflected[0,[0,1]]=reflected[0,[1,0]]
        with self.assertRaises(AssertionError):verify_native_rest(p[None].astype(np.float32),reflected[None],p,t)
