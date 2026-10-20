"""Independent funnel quadrature detects an incorrect guide Jacobian."""
import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_interweave import centered_parameter_step
from pacnerf_comparison import Target
from pacnerf_conditioned_transport import ConditionedTransport,make_target
from test_pacnerf_floor import model as floor_model


class InterweaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def funnel(self):
        class Physical:
            def objective(self,z):
                return .5*(z[:,[0,1,3,4,5,6]]**2).sum(-1)+z[:,0]+.5*z[:,2]**2*torch.exp(-2*z[:,0])
        class Guide:
            model=Physical()
            def transform(self,z,inverse=False):
                out=z.clone();out[:,2]=z[:,2]*torch.exp((-1 if inverse else 1)*z[:,0])
                return out,z[:,0]
            def objective(self,z):
                physical,jac=self.transform(z);return self.model.objective(physical)-jac
        target=Target(Guide(),torch.zeros(7,dtype=torch.float64),torch.eye(7,dtype=torch.float64))
        target.decode=lambda w:target.model.transform(target.coordinates(w))[0]
        return target

    def test_funnel_conditional_matches_independent_quadrature_and_preserves_path(self):
        target=self.funnel()
        w=torch.zeros(4,7,dtype=torch.float64);w[:,2]=.8
        rng=torch.Generator().manual_seed(981);draws=[]
        for i in range(4500):
            w,*_=centered_parameter_step(target,w,4,rng)
            if i>=500:draws.append(w[:,0].clone())
        torch.testing.assert_close(target.decode(w)[:,2],torch.full((4,),.8,dtype=w.dtype),atol=2e-13,rtol=0)
        x=np.linspace(-8,8,100001);density=np.exp(-.5*x*x-x-.5*.8**2*np.exp(-2*x));density/=np.trapezoid(density,x)
        expected=np.trapezoid(x*density,x);variance=np.trapezoid((x-expected)**2*density,x)
        values=torch.stack(draws).flatten()
        self.assertAlmostEqual(float(values.mean()),expected,delta=.035)
        self.assertAlmostEqual(float(values.var()),variance,delta=.035)

    def test_real_neural_joint_and_marginal_keep_physical_path(self):
        for force in ['joint','marginal']:
            model=floor_model(force);guide=ConditionedTransport(model,torch.zeros(model.dimension,dtype=torch.float64))
            reference={'mean':guide.anchor,'precision':torch.eye(model.dimension,dtype=torch.float64),
                'anchor':guide.anchor,'guide':guide.guide,'parameter_blocked_reference':True}
            target=make_target(model,reference);w=torch.zeros(2,model.dimension,dtype=torch.float64)
            before=target.decode(w);new,value,*_=centered_parameter_step(target,w,model.core_dimension,torch.Generator().manual_seed(987))
            after=target.decode(new)
            torch.testing.assert_close(after[:,2:model.core_dimension],before[:,2:model.core_dimension],atol=1e-9,rtol=1e-9)
            if force=='joint':torch.testing.assert_close(after[:,model.core_dimension:-3],before[:,model.core_dimension:-3],atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(value,target.log_ratio_no_grad(new)[0],atol=1e-9,rtol=1e-9)

    def test_interwoven_single_chain_recovers_exactly_through_warmup_and_draws(self):
        from pacnerf_reflected_hybrid import sample
        target=self.funnel()
        class Context:
            def __init__(self,fixed):self.fixed=fixed
            def geometry(self):return torch.zeros(1,1,2,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64)*10
            def __call__(self,state):return torch.zeros(1,dtype=state.dtype),torch.zeros_like(state),torch.zeros(1,dtype=torch.bool)
        target.reflection_context=Context
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as folder:
            def run(name,draws,hook=None):
                return sample(target,7,4,Path(folder)/name,draws,warmup=32,seed=991,chains_count=1,
                    interweave=True,initial=torch.zeros(1,7,dtype=torch.float64),checkpoint_hook=hook)
            direct=run('direct',64)
            run('split',32);resumed=run('split',64)
            def stop(i):raise RuntimeError('interrupted warmup')
            with self.assertRaisesRegex(RuntimeError,'interrupted warmup'):run('warm',64,stop)
            warm=run('warm',64)
        for result in [resumed,warm]:
            for key in ['white','log_ratio','energy_errors','rng_state']:
                torch.testing.assert_close(result[key],direct[key],atol=0,rtol=0)
            self.assertEqual(result['counts'],direct['counts'])
