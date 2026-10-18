"""Proper prediction score agrees with independent closed-form Gaussian CRPS."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_pacnerf_discrepancy import uncertainty_scores
from pacnerf_prediction_metrics import prediction_scores


class PredictionScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary_root=Path(__file__).resolve().parents[1]/'runs';cls.temporary_root.mkdir(exist_ok=True)

    def test_normal_crps_matches_closed_form_in_center_and_far_tail(self):
        class Model:
            frames=[0,3,10,13]
            data={'scene':'normal-score-test','gaussians':torch.zeros(1,3,dtype=torch.float64)}
            renderer=type('Renderer',(),{'gaussian_basis':torch.ones(1,3,1,dtype=torch.float64)})()
            def decode(self,z):return z[:,None,:].expand(-1,14,-1),None
        n=2000
        z=torch.special.ndtri((torch.arange(n,dtype=torch.float64)+.5)/n)[:,None]
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as folder:
            for truth_value in [0.,5.]:
                truth=torch.full((14,1,3),truth_value,dtype=torch.float64)
                torch.save({'particles':truth},Path(folder)/'normal-score-test-evaluation.pt')
                scores=uncertainty_scores(Model(),z,folder)
                phi=math.exp(-truth_value**2/2)/math.sqrt(2*math.pi)
                cdf=.5*(1+math.erf(truth_value/math.sqrt(2)))
                expected=truth_value*(2*cdf-1)+2*phi-1/math.sqrt(math.pi)
                self.assertAlmostEqual(scores['mean_crps_m'],expected,delta=.001)
                numpy_scores=prediction_scores(z.numpy()[:,None,None,:].repeat(6,axis=1).repeat(3,axis=-1),truth.numpy()[4:10])
                self.assertAlmostEqual(numpy_scores['mean_crps_m'],scores['mean_crps_m'],delta=1e-10)
                ninety=next(r for r in scores['coverage_curve'] if r['nominal']==.9)
                self.assertAlmostEqual(ninety['mean_width_m'],2*1.6448536269514722,delta=.006)
                if truth_value==0:
                    self.assertEqual(ninety['coverage'],1.)
                    self.assertAlmostEqual(ninety['mean_interval_score_m'],ninety['mean_width_m'])
                else:
                    self.assertEqual(ninety['coverage'],0.)
                    self.assertGreater(ninety['mean_interval_score_m'],ninety['mean_width_m'])

    def test_streamed_scaled_force_moments_keep_amplitude_correlations(self):
        from test_pacnerf import fixture
        from pacnerf_discrepancy import DiscrepancyModel
        from compare_pacnerf_discrepancy import force_moments_stream
        model=DiscrepancyModel(fixture())
        z=torch.zeros(40,model.dimension,dtype=torch.float64)
        z[:,-2]=torch.linspace(-.7,.8,len(z))
        with torch.no_grad():
            mean,cov,norm=model.force_conditionals(z)
            factor=torch.ones_like(mean);factor[:,model.original_force_dimension:]=torch.exp(z[:,-2,None])
            expected_scaled_mean=mean*factor
            expected_scaled_within=(cov*factor[:,:,None]*factor[:,None,:]).mean(0)
            actual,within,scaled,scaled_within,normalizers=force_moments_stream(model,z)
            torch.testing.assert_close(actual,mean,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(within,cov.mean(0),atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(scaled,expected_scaled_mean,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(scaled_within,expected_scaled_within,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(normalizers,norm,atol=1e-9,rtol=1e-9)
