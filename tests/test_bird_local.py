"""Composition checks for dense state preconditioning and CPU/CUDA rendering."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from bird_local import DeviceRenderer, label
from test_pacnerf_floor import model as floor_model
from pacnerf_conditioned_transport import ConditionedTransport, make_target
from pacnerf_reflected_hybrid import ReflectedStateContext
from pacnerf_interweave import centered_parameter_step


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


if __name__=='__main__':
    unittest.main()
