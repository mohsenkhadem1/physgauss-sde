from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from bird_conditional_bridge import ConditionalBridge,conditional_target,learn_reference
from test_pacnerf_floor import model as floor_model
from pacnerf_conditioned_transport import ConditionedTransport,make_target
from bird_dense_sampler import centered_parameter_step


class ConditionalBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self,force='marginal'):
        model=floor_model(force);d=model.dimension
        guide=ConditionedTransport(model,torch.zeros(d,dtype=torch.float64))
        parent={'mean':guide.anchor,'precision':torch.eye(d,dtype=torch.float64),
            'anchor':guide.anchor,'guide':guide.guide,'parameter_blocked_reference':True}
        white=.03*torch.randn(4,64,d,dtype=torch.float64,generator=torch.Generator().manual_seed(19033))
        # Explicit nonlinear parameter/path coupling exercises both the
        # regression mean and the new parameter-dependent Jacobian.
        white[:,:,20]+=.02*(white[:,:,0]/.03).square()
        reference=learn_reference(model,parent,{'white':white})
        return model,parent,reference

    def test_both_force_arms_roundtrip_and_independent_state_determinant(self):
        for force in ['marginal','joint']:
            model,parent,reference=self.fixture(force)
            target=conditional_target(model,reference);w=reference['initial_white']
            native=target.coordinates(w);physical,jac=target.model.transform(native)
            inverse,inverse_jac=target.model.transform(physical,inverse=True)
            torch.testing.assert_close(inverse,native,atol=1e-8,rtol=1e-8)
            torch.testing.assert_close(inverse_jac,jac,atol=1e-8,rtol=1e-8)
            _,root=target.model.gaussian(native)
            _,log_scale=target.model.correction(native)
            matrix=torch.linalg.solve_triangular(root.transpose(-1,-2),
                reference['conditional_bridge']['residual_root'].expand(len(w),-1,-1),upper=True)
            matrix=matrix*log_scale.exp()[:,None,None]
            # LAPACK LU expects contiguous storage for these batched solves.
            independent=torch.stack([torch.linalg.slogdet(row.contiguous()).logabsdet for row in matrix])
            torch.testing.assert_close(jac,independent,atol=1e-8,rtol=1e-8)
            torch.testing.assert_close(target.model.objective(native),model.objective(physical)-independent,atol=1e-8,rtol=1e-9)

    def test_dense_floor_normals_and_centered_interweaving_preserve_the_path(self):
        for force in ['marginal','joint']:
            model,parent,reference=self.fixture(force)
            target=conditional_target(model,reference);w=reference['initial_white']
            context=target.reflection_context(w);normal,offset=context.geometry()
            for delta in [0.,.1]:
                shifted=w.clone();shifted[:,2:model.core_dimension]+=delta
                physical=target.decode(shifted)
                actual=offset+torch.einsum('bkd,bd->bk',normal,shifted[:,2:model.core_dimension])
                torch.testing.assert_close(actual,model.floor_heights(physical).flatten(1),atol=1e-9,rtol=1e-9)
            before=target.decode(w)
            new,value,*_=centered_parameter_step(target,w,model.core_dimension,torch.Generator().manual_seed(19037))
            after=target.decode(new)
            torch.testing.assert_close(before[:,2:model.core_dimension],after[:,2:model.core_dimension],atol=1e-8,rtol=1e-8)
            torch.testing.assert_close(value,target.log_ratio_no_grad(new)[0],atol=1e-8,rtol=1e-9)

    def test_new_parameter_jacobian_derivative_matches_finite_differences(self):
        model,parent,reference=self.fixture()
        target=conditional_target(model,reference)
        z=target.coordinates(reference['initial_white'][:1]).detach().requires_grad_()
        value=target.model.objective(z);gradient,=torch.autograd.grad(value.sum(),z)
        for index in [0,model.dimension-3,20]:
            direction=torch.zeros_like(z);direction[:,index]=1;eps=2e-5
            finite=(target.model.objective(z+eps*direction)-target.model.objective(z-eps*direction))/(2*eps)
            torch.testing.assert_close(finite[0],gradient[0,index],atol=.02,rtol=2e-4)


if __name__=='__main__':unittest.main()
