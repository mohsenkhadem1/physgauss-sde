"""Independent exact Gaussian moments and checkpoint/RNG continuity."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_elliptical import sample


class EllipticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary_root=Path(__file__).resolve().parents[1]/'runs'
        cls.temporary_root.mkdir(exist_ok=True)

    def target(self):
        mean=torch.tensor([.6,-.4,.2,-.3],dtype=torch.float64)
        precision=torch.tensor([[2.,.3,0,0],[.3,1.5,.2,0],[0,.2,2.,.4],[0,0,.4,1.8]],dtype=mean.dtype)
        def evaluate(w):
            delta=w-mean
            value=-.5*torch.einsum('bi,ij,bj->b',delta,precision,delta)+.5*w.square().sum(-1)
            return value,~torch.isfinite(value)
        return mean,precision,evaluate

    def test_reproduces_independent_correlated_gaussian_mean_and_covariance(self):
        mean,precision,evaluate=self.target()
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as folder:
            chains=sample(evaluate,4,folder,4000,warmup=200,seed=63,chunk=1000)
        draws=chains['white'].flatten(0,1)
        torch.testing.assert_close(draws.mean(0),mean,atol=.04,rtol=0)
        torch.testing.assert_close(torch.cov(draws.T),torch.linalg.inv(precision),atol=.045,rtol=0)

    def test_resumed_prefix_and_rng_are_identical_to_uninterrupted_sampling(self):
        _,_,evaluate=self.target()
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as folder:
            direct=sample(evaluate,4,Path(folder)/'direct',100,warmup=32,seed=74,chunk=25)
            first=sample(evaluate,4,Path(folder)/'split',50,warmup=32,seed=74,chunk=25)
            resumed=sample(evaluate,4,Path(folder)/'split',100,warmup=32,seed=74,chunk=25)
            torch.testing.assert_close(resumed['white'][:,:50],first['white'],atol=0,rtol=0)
            torch.testing.assert_close(resumed['white'],direct['white'],atol=0,rtol=0)
            self.assertTrue(torch.equal(resumed['rng_state'],direct['rng_state']))
            self.assertEqual(resumed['counts'],direct['counts'])
            self.assertEqual(resumed['target_evaluations'],direct['target_evaluations']+1)

    def test_noncontiguous_parameter_block_has_correct_gaussian_moments(self):
        mean,precision,evaluate=self.target()
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as folder:
            chains=sample(evaluate,4,folder,2500,warmup=100,seed=88,chunk=1000,
                blocks=[(1,2),{'indices':[0,2,3]}])
        draws=chains['white'].flatten(0,1)
        torch.testing.assert_close(draws.mean(0),mean,atol=.05,rtol=0)
        torch.testing.assert_close(torch.cov(draws.T),torch.linalg.inv(precision),atol=.06,rtol=0)
