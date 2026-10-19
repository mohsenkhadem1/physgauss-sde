from pathlib import Path
import sys,tempfile,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_hybrid import sample,conditional_reference_transform,ConditionalStateEvaluator
from pacnerf_comparison import Target
from pacnerf_conditioned_transport import ConditionedTransport,make_target
from test_pacnerf_floor import model as floor_model


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.root=Path(__file__).resolve().parents[1]/'runs'

    def target(self):
        d=7;precision=torch.diag(torch.linspace(1.1,2.,d,dtype=torch.float64))
        precision+=torch.diag(torch.full((d-1,),.25,dtype=torch.float64),1)+torch.diag(torch.full((d-1,),.25,dtype=torch.float64),-1)
        mean=torch.linspace(-.4,.5,d,dtype=torch.float64)
        class Gaussian:
            def objective(self,z):
                delta=z-mean
                return .5*torch.einsum('bi,ij,bj->b',delta,precision,delta)
        return Target(Gaussian(),torch.zeros(d,dtype=mean.dtype),torch.eye(d,dtype=mean.dtype)),mean,precision

    def test_joint_moments_match_independent_correlated_gaussian(self):
        target,mean,precision=self.target()
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            chains=sample(target,7,4,folder,4000,warmup=200,seed=109,chunk=1000)
        values=chains['white'].flatten(0,1)
        torch.testing.assert_close(values.mean(0),mean,atol=.05,rtol=0)
        torch.testing.assert_close(torch.cov(values.T),torch.linalg.inv(precision),atol=.05,rtol=0)

    def test_retained_prefix_rng_and_warmup_continuations_are_exact(self):
        target,_,_=self.target()
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            root=Path(folder)
            direct=sample(target,7,4,root/'direct',100,warmup=32,seed=119,chunk=25)
            first=sample(target,7,4,root/'split',50,warmup=32,seed=119,chunk=25)
            resumed=sample(target,7,4,root/'split',100,warmup=32,seed=119,chunk=25)
            def stop(iteration):
                if iteration==25:raise RuntimeError('test interruption')
            with self.assertRaisesRegex(RuntimeError,'test interruption'):
                sample(target,7,4,root/'warm',100,warmup=32,seed=119,chunk=25,checkpoint_hook=stop)
            warm_resumed=sample(target,7,4,root/'warm',100,warmup=32,seed=119,chunk=25)
        torch.testing.assert_close(first['white'],resumed['white'][:,:50],atol=0,rtol=0)
        for result in [resumed,warm_resumed]:
            for key in ['white','log_ratio','energy_errors','rng_state']:
                torch.testing.assert_close(result[key],direct[key],atol=0,rtol=0)
            self.assertEqual(result['counts'],direct['counts'])
            self.assertEqual(result['parameter_slice_attempts'],direct['parameter_slice_attempts'])

    def test_conditional_transform_whitens_and_preserves_parameters(self):
        rng=torch.Generator().manual_seed(141);a=torch.randn(9,9,dtype=torch.float64,generator=rng)
        precision=a.T@a+.2*torch.eye(9,dtype=a.dtype)
        transform=conditional_reference_transform(precision,6)
        torch.testing.assert_close(transform.T@precision@transform,torch.eye(9,dtype=a.dtype),atol=1e-12,rtol=1e-12)
        self.assertTrue(bool((transform[[0,1,6,7,8],2:6]==0).all()))

    def test_supplied_initial_states_resume_exactly_and_cannot_change(self):
        target,_,_=self.target()
        initial=torch.linspace(-1,1,28,dtype=torch.float64).reshape(4,7)
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            root=Path(folder)
            direct=sample(target,7,4,root/'direct',64,warmup=12,seed=121,initial=initial)
            sample(target,7,4,root/'split',32,warmup=12,seed=121,initial=initial)
            resumed=sample(target,7,4,root/'split',64,warmup=12,seed=121,initial=initial)
            with self.assertRaisesRegex(ValueError,'protocol changed'):
                sample(target,7,4,root/'split',80,warmup=12,seed=121,initial=initial+.1)
        for key in ['white','log_ratio','energy_errors','rng_state']:
            torch.testing.assert_close(direct[key],resumed[key],atol=0,rtol=0)

    def test_frozen_parameter_partial_gradient_matches_full_floor_target(self):
        model=floor_model();guide=ConditionedTransport(model,torch.zeros(model.dimension,dtype=torch.float64))
        reference={'mean':torch.zeros(model.dimension,dtype=torch.float64),'precision':torch.eye(model.dimension,dtype=torch.float64),
            'anchor':guide.anchor,'guide':guide.guide,'parameter_blocked_reference':True}
        target=make_target(model,reference)
        q=.05*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(157))
        full,gradient,bad=target(q)
        evaluator=ConditionalStateEvaluator(target,q,model.core_dimension)
        partial,state_gradient,state_bad=evaluator(q[:,2:model.core_dimension])
        torch.testing.assert_close(partial,full,atol=1e-8,rtol=1e-10)
        torch.testing.assert_close(state_gradient,gradient[:,2:model.core_dimension],atol=2e-7,rtol=2e-9)
        self.assertTrue(torch.equal(bad,state_bad));self.assertIsNone(model._conditional_context)
        # Parameters can still move afterwards: conditional caches never leak
        # into the following parameter-slice or an ordinary full gradient.
        moved=q.clone();moved[:,0]+=.2
        actual=target.log_ratio_no_grad(moved)[0]
        expected=-target.model.objective(target.coordinates(moved))+target.center+.5*moved.square().sum(-1)
        torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-10)
