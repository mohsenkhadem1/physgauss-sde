import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from evaluate_gaussian_active_probes import predict_control,weighted_quantiles


class CapturingForward:
    def __init__(self):self.calls=[]
    def __call__(self,rest,times,physical,boundary,family):
        self.calls.append((physical.clone(),family.clone()))
        return rest[None,None]+physical[:,None,None,16:19]*times[None,:,None,None]


class ActiveProbeTests(unittest.TestCase):
    def test_weighted_quantiles_do_not_interpolate_through_zero_mass(self):
        np.testing.assert_array_equal(weighted_quantiles([1.,4.,9.],[0.,1.,0.],[0.,.05,.5,.95,1.]),[4.]*5)
        np.testing.assert_array_equal(weighted_quantiles([1.,4.,9.],[.1,.8,.1],[.05,.5,.95]),[1.,4.,9.])
    def test_known_control_replaces_unknown_forces_and_preserves_particles(self):
        operator=CapturingForward();bundle={'device':torch.device('cpu'),'forward':operator}
        physical=torch.zeros(3,25);physical[:,16:22]=.4
        particles={'effective_physical':physical,'boundary_type':torch.tensor([0,1,3]),'force_type':torch.tensor([1,2,3])}
        reference={'rest_points_m':np.zeros((2,3))}
        menu={'known_handle_centers_m':[[.1,.2,.3],[.4,.5,.6]],'sensor':{'times_s':[0.,1.]}}
        action={'peak_force_N':[.02,0,0],'force_type_index':0}
        predicted=predict_control(bundle,particles,reference,menu,action,chunk_size=2)
        np.testing.assert_allclose(predicted[:,1,:,0],.02)
        self.assertTrue(torch.all(physical[:,16:22]==.4))
        for query,family in operator.calls:
            torch.testing.assert_close(query[:,10:16],torch.tensor([[.1,.2,.3,.4,.5,.6]]).expand(len(query),-1))
            self.assertTrue(torch.all(query[:,19:22]==0))
            self.assertTrue(torch.all(family==0))


if __name__=='__main__':unittest.main()
