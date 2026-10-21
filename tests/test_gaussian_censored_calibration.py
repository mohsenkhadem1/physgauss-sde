from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from censored_range_observation import CensoredRanges
from calibrate_gaussian_censored_probes import fit_conservative_parameters,read_sequence,verify_generation_source,load_generation_amendment
from calibrate_gaussian_probe_discrepancy import probe_discrepancy_covariance
from gaussian_probe_roles import load_role_manifest

class CensoredCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.times=np.arange(15)*.2+.1;self.ids=['a','b','c','d'];self.shape=(4,180)

    def test_model_missing_ranges_never_enter_covariance_fit(self):
        pv=np.ones(self.shape,bool);ov=pv.copy();pv[:,0]=False
        predicted=CensoredRanges(pv,np.zeros(self.shape))
        actual=np.full(self.shape,.0001);actual[:,0]=.05
        p1,e1,d1=fit_conservative_parameters(predicted,CensoredRanges(ov,actual),self.ids,self.times,12)
        actual[:,0]=1000.
        p2,e2,d2=fit_conservative_parameters(predicted,CensoredRanges(ov,actual),self.ids,self.times,12)
        self.assertEqual(p1,p2);self.assertEqual(e1,e2)
        self.assertLess(p1['independent_std_m'],.0002)
        self.assertEqual(d1['joint_valid_counts_per_sequence'],[179]*4)

    def test_correlated_bias_is_retained_and_covariance_positive(self):
        bias=np.array([.0001,.0002,-.0004,.0003])[:,None]
        predicted=CensoredRanges(np.ones(self.shape,bool),np.zeros(self.shape))
        observed=CensoredRanges(np.ones(self.shape,bool),np.broadcast_to(bias,self.shape))
        p,e,d=fit_conservative_parameters(predicted,observed,self.ids,self.times,12)
        self.assertGreaterEqual(p['sequence_common_std_m'],.0003999)
        self.assertGreaterEqual(p['independent_std_m'],.0004/1.6448536269514722-1e-12)
        covariance=probe_discrepancy_covariance(self.times,12,p)
        self.assertTrue((np.linalg.eigvalsh(covariance)>0).all());self.assertTrue(d['qualified'])
        self.assertAlmostEqual(e,.1)

    def test_no_joint_returns_has_explicit_broad_unqualified_fallback(self):
        p,e,d=fit_conservative_parameters(CensoredRanges(np.zeros(self.shape,bool),np.zeros(self.shape)),CensoredRanges(np.ones(self.shape,bool),np.zeros(self.shape)),self.ids,self.times,12)
        self.assertFalse(d['qualified']);self.assertFalse(d['range_support_sufficient'])
        self.assertEqual(p['independent_std_m'],.05);self.assertEqual(e,.49)
        self.assertIsNone(d['joint_range_rmse_m'])

    def test_visibility_smoothing_counts_case_blocks_not_pixels(self):
        pv=np.ones(self.shape,bool);ov=pv.copy();ov[0,:90]=False
        p=CensoredRanges(pv,np.zeros(self.shape));o=CensoredRanges(ov,np.zeros(self.shape))
        _,epsilon,d=fit_conservative_parameters(p,o,self.ids,self.times,12)
        bigp=CensoredRanges(np.repeat(p.visible,3,axis=-1),np.repeat(p.residual_m,3,axis=-1))
        bigo=CensoredRanges(np.repeat(o.visible,3,axis=-1),np.repeat(o.residual_m,3,axis=-1))
        _,large,_=fit_conservative_parameters(bigp,bigo,self.ids,self.times,36)
        self.assertEqual(epsilon,large);self.assertEqual(d['effective_calibration_case_count'],4)

    def test_sensor_exception_only_accepts_declared_case_and_required_proof(self):
        root=Path(__file__).resolve().parents[1]
        roles=load_role_manifest(root/'configs/gaussian_probe_roles_v01.json')
        amendment=load_generation_amendment(root/'configs/gaussian_probe_sensor_v03.json')
        original={'source_sha256':{'isaac_calibrated_probes_v02.py':roles['generation_source_sha256']}}
        verify_generation_source(original,roles,'t0012',Path('/absent'))
        replacement={'source_sha256':{'isaac_calibrated_probes_v03.py':amendment['worker_sha256'],'isaac_probe_sensor_guard.py':amendment['sensor_guard_sha256']}}
        with self.assertRaises(ValueError):verify_generation_source(replacement,roles,'t0013',Path('/absent'))
        with self.assertRaises(ValueError):verify_generation_source(replacement,roles,'t0014',Path('/absent'),amendment)
        with self.assertRaises(FileNotFoundError):verify_generation_source(replacement,roles,'t0013',Path('/absent'),amendment)
        bad={'source_sha256':{**replacement['source_sha256'],'isaac_probe_sensor_guard.py':'wrong'}}
        with self.assertRaises(ValueError):verify_generation_source(bad,roles,'t0013',Path('/absent'),amendment)

    def test_modified_amendment_rejected_before_response_access(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'amendment.json';path.write_text('{}')
            with self.assertRaises(ValueError):load_generation_amendment(path)

    def test_nonreserved_or_test_rejected_before_any_private_read(self):
        roles=load_role_manifest(Path(__file__).resolve().parents[1]/'configs/gaussian_probe_roles_v01.json')
        for case in [{'episode_id':'t0000','split':'train'},{'episode_id':'t0012','split':'test'}]:
            with self.assertRaises(ValueError):read_sequence(None,Path('/absent'),Path('/absent'),roles,case,'poke_0',[],12,1,'cpu')

if __name__=='__main__':unittest.main()
