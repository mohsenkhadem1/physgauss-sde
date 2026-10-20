from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_reflected_hybrid import ReflectedStateContext,sample
from pacnerf_comparison import Target
from pacnerf_conditioned_transport import ConditionedTransport,make_target
from test_pacnerf_floor import model as floor_model


class ReflectedHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.root=Path(__file__).resolve().parents[1]/'runs'

    def toy(self):
        d=7;core=4;dtype=torch.float64;scale=.2
        precision=torch.diag(torch.linspace(1.1,1.8,d,dtype=dtype))
        precision+=torch.diag(torch.full((d-1,),.2,dtype=dtype),1)+torch.diag(torch.full((d-1,),.2,dtype=dtype),-1)
        mean=torch.linspace(-.4,.4,d,dtype=dtype);a=torch.linspace(-.6,.8,d,dtype=dtype);b=.15
        def gaussian(z):return .5*torch.einsum('bi,ij,bj->b',z-mean,precision,z-mean)
        class Objective:
            def objective(self,z):return gaussian(z)+torch.nn.functional.softplus(-(z@a+b)/scale)
        target=Target(Objective(),torch.zeros(d,dtype=dtype),torch.eye(d,dtype=dtype))
        class Context:
            def __init__(self,fixed):self.fixed=fixed.clone()
            def geometry(self):
                offset=self.fixed@a+b-(self.fixed[:,2:core]*a[2:core]).sum(-1)
                return a[None,None,2:core].expand(len(self.fixed),1,-1),offset[:,None]
            def __call__(self,state):
                full=self.fixed.clone();full[:,2:core]=state
                value=-gaussian(full)+target.center+.5*full.square().sum(-1)
                gradient=(-(full-mean)@precision)[:,2:core]+state
                return value,gradient,torch.zeros(len(self.fixed),dtype=torch.bool)
        target.reflection_context=Context
        return target,mean,torch.linalg.inv(precision),a,b,scale

    def test_correlated_soft_floor_joint_moments_match_independent_one_dimensional_quadrature(self):
        target,mean,cov,a,b,scale=self.toy()
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            result=sample(target,7,4,folder,4000,warmup=200,seed=845,floor_scale=scale)
        hmean=float(a@mean+b);hvar=float(a@cov@a)
        h=np.linspace(hmean-12*hvar**.5,hmean+12*hvar**.5,60001)
        density=np.exp(-.5*(h-hmean)**2/hvar)/(1+np.exp(-h/scale));density/=np.trapezoid(density,h)
        mh=np.trapezoid(h*density,h);vh=np.trapezoid((h-mh)**2*density,h)
        regression=cov@a/hvar
        expected_mean=mean+regression*(mh-hmean)
        expected_cov=cov-(cov@a)[:,None]*(cov@a)[None]/hvar+regression[:,None]*regression[None]*vh
        draws=result['white'].flatten(0,1)
        torch.testing.assert_close(draws.mean(0),expected_mean,atol=.04,rtol=0)
        torch.testing.assert_close(torch.cov(draws.T),expected_cov,atol=.05,rtol=0)

    def test_retained_and_warmup_recovery_are_bit_identical(self):
        target,*rest=self.toy();scale=rest[-1]
        initial=torch.linspace(-1,1,28,dtype=torch.float64).reshape(4,7)
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            p=Path(folder)
            direct=sample(target,7,4,p/'direct',80,warmup=32,seed=849,initial=initial,floor_scale=scale)
            sample(target,7,4,p/'split',40,warmup=32,seed=849,initial=initial,floor_scale=scale)
            resumed=sample(target,7,4,p/'split',80,warmup=32,seed=849,initial=initial,floor_scale=scale)
            def stop(iteration):
                if iteration==25:raise RuntimeError('test interruption')
            with self.assertRaisesRegex(RuntimeError,'test interruption'):
                sample(target,7,4,p/'warm',80,warmup=32,seed=849,initial=initial,floor_scale=scale,checkpoint_hook=stop)
            warm=sample(target,7,4,p/'warm',80,warmup=32,seed=849,initial=initial,floor_scale=scale)
        for actual in [resumed,warm]:
            for key in ['white','log_ratio','energy_errors','rng_state']:
                torch.testing.assert_close(actual[key],direct[key],atol=0,rtol=0)
            self.assertEqual(actual['counts'],direct['counts'])

    def test_actual_neural_joint_and_marginal_height_geometry_and_residual_gradient(self):
        for force in ['marginal','joint']:
            model=floor_model(force);guide=ConditionedTransport(model,torch.zeros(model.dimension,dtype=torch.float64))
            reference={'mean':torch.zeros(model.dimension,dtype=torch.float64),'precision':torch.eye(model.dimension,dtype=torch.float64),
                'anchor':guide.anchor,'guide':guide.guide,'parameter_blocked_reference':True}
            target=make_target(model,reference)
            w=.03*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(851))
            context=ReflectedStateContext(target,w,model.core_dimension);normal,offset=context.geometry()
            for delta in [0.,.1]:
                shifted=w.clone();shifted[:,2:model.core_dimension]+=delta
                physical=target.model.to_physical(target.coordinates(shifted))
                expected=model.floor_heights(physical).flatten(1)
                actual=torch.einsum('bkd,bd->bk',normal,shifted[:,2:model.core_dimension])+offset
                torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
            value,grad,bad=context(w[:,2:model.core_dimension])
            full,full_grad,full_bad=target(w)
            variable=w.detach().requires_grad_()
            penalty=model.floor_penalty(target.model.to_physical(target.coordinates(variable)))
            floor_grad,=torch.autograd.grad(penalty.sum(),variable)
            torch.testing.assert_close(value,full+penalty.detach(),atol=2e-7,rtol=1e-9)
            torch.testing.assert_close(grad,(full_grad+floor_grad)[:,2:model.core_dimension],atol=2e-6,rtol=1e-8)
            self.assertTrue(torch.equal(bad,full_bad));self.assertIsNone(model._conditional_context)

    def test_single_chain_worker_continues_exactly_for_later_multichain_aggregation(self):
        target,*rest=self.toy();scale=rest[-1]
        initial=torch.zeros(1,7,dtype=torch.float64)
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            p=Path(folder)
            direct=sample(target,7,4,p/'direct',64,warmup=16,seed=857,initial=initial,floor_scale=scale,chains_count=1)
            sample(target,7,4,p/'split',32,warmup=16,seed=857,initial=initial,floor_scale=scale,chains_count=1)
            resumed=sample(target,7,4,p/'split',64,warmup=16,seed=857,initial=initial,floor_scale=scale,chains_count=1)
        self.assertEqual(direct['white'].shape,(1,64,7))
        for key in ['white','log_ratio','energy_errors','rng_state']:
            torch.testing.assert_close(resumed[key],direct[key],atol=0,rtol=0)
