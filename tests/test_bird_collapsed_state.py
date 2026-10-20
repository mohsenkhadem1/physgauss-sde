from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from bird_collapsed_state import GaussianVelocities,CollapsedStateContext,indices
from test_pacnerf_floor import model as floor_model
from pacnerf_conditioned_transport import ConditionedTransport,make_target


class CollapsedVelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,force):
        model=floor_model(force);model.noise='constant'
        return model

    def test_exact_full_density_factorization_and_velocity_linear_map_both_force_arms(self):
        for force in ['marginal','joint']:
            model=self.model(force)
            z=.03*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(19041))
            bundle=GaussianVelocities(model,z)
            actual=model.dynamics(z);expected=bundle.marginal_nll+bundle.conditional_nll(z)
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-9)
            original=model.terms(z)[0];zero=model.terms(bundle.zero)[0]
            linear=torch.einsum('btdk,bk->btd',bundle.velocity_map,z[:,bundle.v_index])
            torch.testing.assert_close(original-zero,linear,atol=1e-11,rtol=1e-10)
            mean=bundle.zero.clone();mean[:,bundle.v_index]=bundle.mean[:,bundle.force_dimension:]
            if force=='joint':mean[:,model.base_dimension:model.core_dimension]=bundle.mean[:,:bundle.force_dimension]
            mean=mean.detach().requires_grad_();gradient,=torch.autograd.grad(model.dynamics(mean).sum(),mean)
            selected=bundle.v_index if force=='marginal' else torch.cat((bundle.v_index,torch.arange(model.base_dimension,model.core_dimension)))
            self.assertLess(float(gradient[:,selected].abs().max()),1e-6)

    def test_collapsed_position_and_parameter_gradients_match_finite_differences(self):
        model=self.model('marginal')
        z=.01*torch.randn(1,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(19043))
        z.requires_grad_();value=GaussianVelocities(model,z).marginal_nll
        gradient,=torch.autograd.grad(value.sum(),z)
        q,_=indices(model)
        for index in [0,model.dimension-3,int(q[0]),int(q[-1])]:
            direction=torch.zeros_like(z);direction[:,index]=1;eps=2e-5
            finite=(GaussianVelocities(model,z+eps*direction).marginal_nll-
                GaussianVelocities(model,z-eps*direction).marginal_nll)/(2*eps)
            torch.testing.assert_close(finite[0],gradient[0,index],atol=.003,rtol=3e-4)

    def test_marginal_position_reference_and_reflection_walls_preserve_geometry(self):
        for force in ['marginal','joint']:
            model=self.model(force);guide=ConditionedTransport(model,torch.zeros(model.dimension,dtype=torch.float64))
            reference={'mean':guide.anchor,'precision':torch.eye(model.dimension,dtype=torch.float64),
                'anchor':guide.anchor,'guide':guide.guide,'parameter_blocked_reference':True}
            target=make_target(model,reference)
            white=.02*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(19047))
            context=CollapsedStateContext(target,white,model.core_dimension)
            torch.testing.assert_close(context.physical(context.initial),target.decode(white),atol=1e-9,rtol=1e-9)
            normal,offset=context.geometry()
            for delta in [0.,.1]:
                position=context.initial+delta
                actual=offset+torch.einsum('bkd,bd->bk',normal,position)
                torch.testing.assert_close(actual,model.floor_heights(context.physical(position)).flatten(1),atol=1e-9,rtol=1e-9)
            value,gradient,bad=context(context.initial)
            self.assertFalse(bool(bad.any()));self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertIsNone(model._conditional_context)


if __name__=='__main__':unittest.main()
