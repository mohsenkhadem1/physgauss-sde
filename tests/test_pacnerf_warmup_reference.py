from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_comparison import Target
from pacnerf_warmup_reference import adapt_reference
from verify_pacnerf_sampling_reference import verify_reference
from pacnerf_hybrid import conditional_reference_transform


class WarmupReferenceTests(unittest.TestCase):
    def test_known_rank_one_covariance_and_preserved_initial_states(self):
        dtype=torch.float64;d=7;n=256
        precision=torch.diag(torch.linspace(.7,2.,d,dtype=dtype));mean=torch.linspace(-.3,.4,d,dtype=dtype)
        class Objective:
            def objective(self,z):return .5*z.square().sum(-1)
        target=Target(Objective(),mean,precision)
        coefficient=torch.linspace(-1,1,n,dtype=dtype)
        direction=torch.linspace(.2,1.4,d,dtype=dtype);offset=torch.linspace(.1,.7,d,dtype=dtype)
        white=(offset+coefficient[:,None]*direction).reshape(4,64,d)
        states={'white':white,'position':white[:,-1],'completed_warmup':150}
        reference={'mean':mean.clone(),'precision':precision.clone(),'fit':{}}
        learned=adapt_reference(reference,target,states,4,.5)
        torch.testing.assert_close(learned['sampling_mean'],mean+target.transform@offset,atol=1e-14,rtol=1e-14)
        variance=coefficient.var()
        # Independent Sherman-Morrison inverse of .5(I+var(c)*v*v^T).
        inverse_white=2*(torch.eye(d,dtype=dtype)-variance*(direction[:,None]*direction[None])/(1+variance*direction.square().sum()))
        inverse_transform=torch.diag(precision.diagonal().sqrt())
        expected=inverse_transform@inverse_white@inverse_transform
        torch.testing.assert_close(learned['sampling_precision'],expected,atol=2e-14,rtol=2e-14)
        self.assertLess(learned['reference_learning']['maximum_native_state_preservation_error'],1e-12)
        self.assertNotIn('sampling_mean',reference)
        torch.testing.assert_close(reference['mean'],mean,atol=0,rtol=0)

    def test_independent_replay_accepts_correlated_reference_and_rejects_tampering(self):
        import copy
        dtype=torch.float64;d=9;core=6
        rng=torch.Generator().manual_seed(719)
        a=torch.randn(d,d,dtype=dtype,generator=rng)
        precision=a.T@a+torch.eye(d,dtype=dtype)
        mean=torch.randn(d,dtype=dtype,generator=rng)
        class Objective:
            def objective(self,z):return .5*z.square().sum(-1)
        target=Target(Objective(),mean,precision)
        target.transform=conditional_reference_transform(precision,core)
        white=torch.randn(4,150,d,dtype=dtype,generator=rng)+.4
        states={'white':white,'position':white[:,-1],'completed_warmup':150,'retained':150,
            'protocol':{'seed':1240009,'warmup':150,'sampler':'conditional-state-hmc-parameter-ess-v1'}}
        anchor={'mean':mean,'precision':precision,'anchor':mean.clone(),'fit':{}}
        reference=adapt_reference(anchor,target,states,core)
        settings={'case_index':8,'reference_learning':{'warmup':150,'draws_per_chain':150,'shrinkage':.5,'seed_offset':1000000}}
        self.assertTrue(verify_reference(reference,anchor,states,core,settings)['all_pass'])
        broken=copy.deepcopy(reference);broken['sampling_mean'][0]+=.01
        with self.assertRaises(AssertionError):verify_reference(broken,anchor,states,core,settings)
        wrong=copy.deepcopy(states);wrong['protocol']['seed']+=1
        with self.assertRaisesRegex(ValueError,'protocol differs'):verify_reference(reference,anchor,wrong,core,settings)
