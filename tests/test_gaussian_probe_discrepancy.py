import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from calibrate_gaussian_probe_discrepancy import validate_roles,fit_discrepancy,probe_discrepancy_covariance,diagnostics

class ProbeCalibrationTests(unittest.TestCase):
    def test_role_guard_rejects_test_and_overlap(self):
        plan={'episodes':[{'episode_id':'a','split':'validation'},{'episode_id':'b','split':'validation'},{'episode_id':'t','split':'test'}]}
        validate_roles(plan,['a'],['b'])
        for left,right in [(['a'],['a']),(['a'],['t']),(['t'],['b']),([],['b'])]:
            with self.assertRaises(ValueError):validate_roles(plan,left,right)

    def test_covariance_retains_bias_and_temporal_dependence(self):
        t=np.arange(15)*.2
        r=np.ones((8,15,12))*.0005
        p=fit_discrepancy(r,t)
        self.assertGreaterEqual(p['sequence_common_std_m'],.0004999)
        c=probe_discrepancy_covariance(t,12,p)
        self.assertGreater(c[0,-1],.0004999**2)
        self.assertTrue((np.linalg.eigvalsh(c)>0).all())
        np.testing.assert_allclose(np.diag(c),np.diag(c)[0],atol=1e-20)
        self.assertAlmostEqual(diagnostics(r,c)['range_rmse_m'],.0005)

    def test_independent_noise_calibrates_without_false_sequence_signal(self):
        rng=np.random.default_rng(102)
        r=rng.normal(0,.0002,(128,20,12));t=np.arange(20)*.1
        p=fit_discrepancy(r,t)
        self.assertAlmostEqual(p['independent_std_m'],.0002,delta=.000004)
        c=probe_discrepancy_covariance(t,12,p)
        d=diagnostics(r,c)
        self.assertAlmostEqual(d['marginal_90_coverage'],.9,delta=.015)
        self.assertAlmostEqual(d['mean_whitened_squared_residual'],1.,delta=.06)

    def test_no_assessment_data_argument_in_fit_and_time_covariance(self):
        r=np.zeros((2,3,2));p=fit_discrepancy(r,[.1,.2,.3]);p.update(frame_common_std_m=.001,frame_ar1=.5,sequence_common_std_m=0)
        c=probe_discrepancy_covariance([.1,.2,.3],2,p)
        self.assertAlmostEqual(c[0,2],.5e-6)
        self.assertAlmostEqual(c[0,4],.25e-6)
        with self.assertRaises(ValueError):fit_discrepancy(np.full((2,3,2),np.nan),[.1,.2,.3])

if __name__=='__main__':unittest.main()
