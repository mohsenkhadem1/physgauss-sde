from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_recovery import restore_settings,restore_correlations,CORRELATION_ROUNDOFF_BOUND


class Model:
    def __init__(self):self.rho_x=.6442199856083427;self.rho_y=.564549927694081;self.version='fixed'
    def specification(self):return {'version':self.version,'rgb_spatial_correlation':[self.rho_x,self.rho_y]}


class RecoveryTests(unittest.TestCase):
    def settings(self,model):return {'model':model.specification(),'input_sha256':'input','source':'source','seed':7,'kernel':[.6,1.8]}

    def test_original_constants_restored_bit_exactly_without_mutating_saved_settings(self):
        model=Model();current=self.settings(model);old=copy.deepcopy(current)
        old['model']['rgb_spatial_correlation']=[.6442199856083397,.5645499276940822]
        original=copy.deepcopy(old)
        restored,proof=restore_settings(model,old,current)
        self.assertEqual(restored,original);self.assertEqual(old,original)
        self.assertEqual([model.rho_x,model.rho_y],old['model']['rgb_spatial_correlation'])
        self.assertGreater(proof['maximum_recomputation_difference'],0)
        self.assertLess(proof['maximum_recomputation_difference'],CORRELATION_ROUNDOFF_BOUND)

    def test_source_input_kernel_and_model_changes_remain_fatal(self):
        for field,value in [('source','new'),('input_sha256','different'),('seed',8),('kernel',[.6,2.])]:
            model=Model();current=self.settings(model);old=copy.deepcopy(current);old[field]=value
            with self.assertRaisesRegex(ValueError,'protocol changed'):restore_settings(model,old,current)
        model=Model();old=self.settings(model);model.version='other'
        with self.assertRaisesRegex(ValueError,'model changed'):restore_settings(model,old,self.settings(model))

    def test_material_correlation_change_and_nonfinite_values_rejected(self):
        for value in [.64422,float('nan'),float('inf'),-.1,.96,True]:
            model=Model();old=model.specification();old['rgb_spatial_correlation'][0]=value
            with self.assertRaises(ValueError):restore_correlations(model,old)
