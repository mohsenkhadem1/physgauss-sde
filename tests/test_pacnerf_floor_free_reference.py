from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_pacnerf_floor import model as floor_model
from pacnerf_conditioned_transport import ConditionedTransport,make_target
from pacnerf_floor_free_reference import make_reference


class FloorFreeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_guide_change_preserves_physical_states_and_full_density_with_jacobian(self):
        for force in ['marginal','joint']:
            model=floor_model(force);d=model.dimension;core=model.core_dimension
            guide=ConditionedTransport(model,torch.zeros(d,dtype=torch.float64))
            parent={'mean':torch.zeros(d,dtype=torch.float64),'precision':torch.eye(d,dtype=torch.float64),
                'anchor':guide.anchor,'guide':guide.guide,'parameter_blocked_reference':True}
            rng=torch.Generator().manual_seed(879)
            white=.02*torch.randn(4,32,d,dtype=torch.float64,generator=rng)
            old=make_target(model,parent)
            reference=make_reference(model,parent,{'white':white,'position':white[:,-1]})
            self.assertNotIn('constraint_precision',reference['guide'])
            self.assertIn('constraint_precision',parent['guide'])
            torch.testing.assert_close(reference['precision'][2:core,2:core],torch.eye(core-2,dtype=torch.float64),atol=0,rtol=0)
            target=make_target(model,reference);native=target.coordinates(reference['initial_white'])
            physical,jac=target.model.transform(native)
            torch.testing.assert_close(physical,old.decode(white[:,-1]),atol=1e-9,rtol=1e-10)
            torch.testing.assert_close(target.model.objective(native),model.objective(physical)-jac,atol=1e-8,rtol=1e-10)
            self.assertFalse(reference['proposal_training']['reference_is_posterior_mode'])
