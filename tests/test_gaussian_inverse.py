import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import sys,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from train_gaussian_inverse import collect_cases,add_label_tolerance,GaussianInverseModel,LABEL_TOLERANCE,predictive_energy_score
from gaussian_operator import PhysicalParameterTransform


class InverseTrainingTests(unittest.TestCase):
    def test_catalog_never_reads_test_private_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'episodes/train').mkdir(parents=True)
            (root/'episodes/train/summary.json').write_text('{"completed":true}')
            (root/'episodes/train/validation.json').write_text('{"passed":true}')
            (root/'plan.json').write_text(json.dumps({'dataset':'unit','episodes':[
                {'episode_id':'train','split':'train'},{'episode_id':'test','split':'test'}]}))
            rows=collect_cases([root]);self.assertEqual([r['recipe']['episode_id'] for r in rows],['train'])
            self.assertFalse((root/'episodes/test').exists())

    def test_jitter_is_physical_and_strictly_bounded(self):
        transform=PhysicalParameterTransform();p=(transform.lower+transform.upper)[None]/2
        noise=torch.ones_like(p)
        noisy=add_label_tolerance(p,noise,transform)
        torch.testing.assert_close(noisy-p,torch.tensor(LABEL_TOLERANCE)[None],rtol=.001,atol=1e-7)
        clipped=add_label_tolerance(p,noise*1e8,transform)
        self.assertTrue(transform.contains(clipped).all())
        self.assertTrue(torch.isfinite(transform.physical_to_unconstrained(clipped)).all())

    def test_inference_needs_only_public_arrays(self):
        torch.set_num_threads(2);torch.manual_seed(12);model=GaussianInverseModel(hidden=16,layers=2).eval()
        batch={'images':torch.rand(1,2,4,32,32),'poses':torch.eye(4)[None,None].expand(1,2,4,4),
               'K':torch.tensor([[620.,0,320],[0,620,240],[0,0,1.]]),
               'times':torch.tensor([[0.,1.]]),'rest':torch.rand(8,3)*.2,'calibration_size':(640,480)}
        with torch.no_grad():draw=model.infer(batch,samples=3)
        self.assertEqual(draw['physical'].shape,(1,3,25))
        self.assertTrue(model.posterior.transform.contains(draw['physical']).all())

class PredictiveScoreTests(unittest.TestCase):
    def test_energy_score_units_and_gradients(self):
        target=torch.zeros(2,3,4,3)
        prediction=torch.full((2,4,3,4,3),.01,requires_grad=True)
        score=predictive_energy_score(prediction,target)
        self.assertAlmostEqual(float(score.detach()),1.,places=5)
        score.backward();self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.norm()),0.)
        with self.assertRaises(ValueError):predictive_energy_score(prediction[:,:1],target)

if __name__=='__main__':unittest.main()
