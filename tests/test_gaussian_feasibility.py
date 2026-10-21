import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from gaussian_feasibility import volume_feasibility,feasible_posterior_samples

class FeasibilityTests(unittest.TestCase):
    def test_volume_filter_rejects_inversion_collapse_and_nonfinite(self):
        rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        reference={'rest_points_m':rest,'tetrahedra':np.array([[0,1,2,3]])}
        q=torch.tensor(np.broadcast_to(rest,(4,3,4,3)).copy())
        q[1,:,3,2]=-.5;q[2,:,3,2]=.05;q[3,1,2,0]=float('nan')
        result=volume_feasibility(q,reference,.1)
        self.assertEqual(result['valid'].tolist(),[True,False,False,False])
        self.assertEqual(result['finite'].tolist(),[True,True,True,False])
        self.assertAlmostEqual(float(result['minimum_volume_ratio'][0]),1.)

    def test_fixed_budget_reports_insufficient_support_without_refilling(self):
        from unittest.mock import patch
        rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        reference={'rest_points_m':rest,'tetrahedra':np.array([[0,1,2,3]])}
        physical=torch.zeros(4,25);physical[:,0]=torch.arange(4.);physical[:,16]=.04
        candidates={'physical':physical,'effective_physical':physical,
                    'boundary_type':torch.zeros(4,dtype=torch.long),'force_type':torch.zeros(4,dtype=torch.long)}
        def prediction(bundle,block,reference,times,chunk):
            q=torch.tensor(np.broadcast_to(rest,(len(block['physical']),1,4,3)).copy())
            q[:,0,3,2]=torch.where(block['physical'][:,0]==0,1.,.05)
            return q
        with patch('evaluate_gaussian_inverse.posterior_samples',return_value=candidates) as sample,patch('evaluate_gaussian_inverse.predict_particles',side_effect=prediction):
            result=feasible_posterior_samples({},None,reference,[0.],samples=2,candidate_budget=4)
        sample.assert_called_once()
        self.assertTrue(result['stats']['insufficient_support'])
        self.assertEqual(result['stats']['retained_samples'],1)
        self.assertEqual(result['stats']['rejected_candidates'],3)
        self.assertEqual(result['stats']['estimated_raw_retained_probability'],.25)
        self.assertEqual(len(result['raw_predictions']),2)

if __name__=='__main__':unittest.main()
