"""Composition checks for dense state preconditioning and CPU/CUDA rendering."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from bird_local import DeviceRenderer, label
from test_pacnerf_floor import model as floor_model
from pacnerf_conditioned_transport import ConditionedTransport, make_target
from pacnerf_reflected_hybrid import ReflectedStateContext
from pacnerf_interweave import centered_parameter_step
from bird_dense_sampler import sample as dense_sample, centered_parameter_step as dense_centered_step


class LocalBirdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dense_state_parameter_reference_preserves_physical_density_and_walls(self):
        for force in ['marginal', 'joint']:
            model=floor_model(force);d=model.dimension;core=model.core_dimension
            rng=torch.Generator().manual_seed(10092)
            guide=ConditionedTransport(model,torch.zeros(d,dtype=torch.float64))
            a=.015*torch.randn(d,9,dtype=torch.float64,generator=rng)
            covariance=a@a.T+.01*torch.eye(d,dtype=torch.float64)
            reference={'mean':.01*torch.randn(d,dtype=torch.float64,generator=rng),
                'precision':torch.linalg.inv(covariance),'anchor':guide.anchor,
                'guide':guide.guide,'parameter_blocked_reference':True}
            target=make_target(model,reference)
            w=.02*torch.randn(4,d,dtype=torch.float64,generator=rng)
            native=target.coordinates(w)
            physical,jac=target.model.transform(native)
            value,bad=target.log_ratio_no_grad(w)
            self.assertFalse(bool(bad.any()))
            expected=-model.objective(physical)+jac+target.center+.5*w.square().sum(-1)
            torch.testing.assert_close(value,expected,atol=1e-9,rtol=1e-10)
            torch.testing.assert_close(target.transform@target.transform.T,covariance,atol=1e-10,rtol=1e-9)
            context=ReflectedStateContext(target,w,core)
            normals,offset=context.geometry()
            height=offset+torch.einsum('bkd,bd->bk',normals,w[:,2:core])
            torch.testing.assert_close(height,model.floor_heights(physical).flatten(1),atol=1e-10,rtol=1e-10)
            # The full correlated affine reference must also survive the
            # centered/noncentered interweaving change of variables.
            moved,_,_,_,_=centered_parameter_step(target,w,core,rng)
            decoded=target.decode(moved)
            torch.testing.assert_close(decoded[:,2:core],physical[:,2:core],atol=1e-8,rtol=1e-8)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_floor_renderer_matches_cpu_values_gradients_and_selected_pixels(self):
        model=floor_model();cpu=model.renderer;gpu=DeviceRenderer(cpu,'cuda:0')
        rng=torch.Generator().manual_seed(10094)
        q=.03*torch.randn(2,3,model.modes,dtype=torch.float64,generator=rng)
        specs=[(1,0),(2,2),(3,4)]
        for grad in [False,True]:
            with torch.set_grad_enabled(grad):
                a=q.clone().requires_grad_(grad);b=q.clone().requires_grad_(grad)
                actual=gpu(a,specs);expected=cpu(b,specs)
                torch.testing.assert_close(actual,expected,atol=5e-13,rtol=1e-12)
                if grad:
                    ga,=torch.autograd.grad(actual.square().sum(),a)
                    gb,=torch.autograd.grad(expected.square().sum(),b)
                    torch.testing.assert_close(ga,gb,atol=1e-9,rtol=1e-9)
        indices=torch.tensor([[0,12,31,55]]*3)
        torch.testing.assert_close(gpu(q,specs,pixel_indices=indices),cpu(q,specs,pixel_indices=indices),atol=5e-13,rtol=1e-12)

    def test_physical_indices_are_mapped_to_mechanical_variables(self):
        self.assertEqual(label('physical_67'),'physical: frame 3 displacement mode 5')
        self.assertEqual(label('physical_236'),'physical: frame 10 displacement mode 6')
        self.assertEqual(label('physical_326'),'physical: mechanics discrepancy log scale')

    def test_joint_and_scalar_parameter_composition_recovers_through_warmup(self):
        from test_pacnerf_interweave import InterweaveTests
        target=InterweaveTests.funnel(None)
        class Context:
            def __init__(self,fixed): self.fixed=fixed
            def geometry(self):
                return torch.zeros(1,1,2,dtype=torch.float64),torch.full((1,1),10.,dtype=torch.float64)
            def __call__(self,state):
                return torch.zeros(1,dtype=state.dtype),torch.zeros_like(state),torch.zeros(1,dtype=torch.bool)
        target.reflection_context=Context
        with tempfile.TemporaryDirectory() as folder:
            def run(name,draws,hook=None):
                return dense_sample(target,7,4,Path(folder)/name,draws,warmup=32,
                    seed=19019,chains_count=1,interweave=True,
                    initial=torch.zeros(1,7,dtype=torch.float64),checkpoint_hook=hook)
            direct=run('direct',64)
            run('split',32);resumed=run('split',64)
            def stop(i): raise RuntimeError('interrupted warmup')
            with self.assertRaisesRegex(RuntimeError,'interrupted warmup'):
                run('warm',64,stop)
            warm=run('warm',64)
        for result in [resumed,warm]:
            for key in ['white','log_ratio','energy_errors','rng_state']:
                self.assertTrue(torch.equal(direct[key],result[key]),key)
            self.assertEqual(direct['counts'],result['counts'])

    def test_scalar_centered_funnel_conditional_matches_independent_quadrature(self):
        import numpy as np
        from test_pacnerf_interweave import InterweaveTests
        target=InterweaveTests.funnel(None)
        w=torch.zeros(4,7,dtype=torch.float64);w[:,2]=.8
        rng=torch.Generator().manual_seed(19021);draws=[]
        for i in range(1800):
            w,*_=dense_centered_step(target,w,4,rng)
            if i>=300:draws.append(w[:,0].clone())
        torch.testing.assert_close(target.decode(w)[:,2],torch.full((4,),.8,dtype=w.dtype),atol=2e-12,rtol=0)
        x=np.linspace(-8,8,100001)
        density=np.exp(-.5*x*x-x-.5*.8**2*np.exp(-2*x));density/=np.trapezoid(density,x)
        expected=np.trapezoid(x*density,x);variance=np.trapezoid((x-expected)**2*density,x)
        values=torch.stack(draws).flatten()
        self.assertAlmostEqual(float(values.mean()),expected,delta=.05)
        self.assertAlmostEqual(float(values.var()),variance,delta=.05)


if __name__=='__main__':
    unittest.main()
