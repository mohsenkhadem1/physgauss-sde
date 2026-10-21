"""CPU tests of private device binding; CUDA mechanics is a separate gate."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


SOURCE=Path(__file__).resolve().parents[3]/'experiments/campaign_cuda_reference_probe.py'
SPEC=importlib.util.spec_from_file_location('cuda_probe',SOURCE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)


class DeviceBindingTests(unittest.TestCase):
    def test_source_and_shared_globals_remain_unchanged(self):
        expected=probe.core.tensor
        x,t=probe.unit_tet();cpu=probe.core.NonlinearDynamics(x,t,1200.,.3,600.)
        wrapped,proof=probe.make_device_model(cpu,'cpu')
        self.assertIs(probe.core.tensor,expected)
        self.assertTrue(all(proof['method_code_identity'].values()))
        self.assertTrue(all(proof['helper_code_identity'].values()))
        self.assertIsNot(wrapped.step.__func__.__globals__,probe.core.NonlinearDynamics.step.__globals__)
        self.assertIsInstance(cpu.tets,np.ndarray)
        self.assertIsInstance(wrapped.tets,torch.Tensor)
        self.assertEqual(wrapped.points.dtype,torch.float64)

    def test_cpu_rebinding_bitwise_state_parity(self):
        x,t=probe.unit_tet();cpu=probe.core.NonlinearDynamics(x,t,1200.,.32,600.)
        wrapped,_=probe.make_device_model(cpu,'cpu')
        a,b=cpu.initial_state(),wrapped.initial_state()
        f=cpu.body_mass.numpy()[:,None]*np.array([.03,-.02,.01])
        for _ in range(3):
            a,ra=cpu.step(a,.01,nodal_forces=f)
            b,rb=wrapped.step(b,.01,nodal_forces=f)
            for name,value in probe.state_arrays(a).items():
                np.testing.assert_array_equal(value,probe.state_arrays(b)[name])
            self.assertEqual(ra,rb)

    def test_foreign_allocation_rejected_before_backend_call(self):
        with self.assertRaises(ValueError):probe.DeviceFactories('cpu').zeros(3,device='cuda:0')

    def test_copied_buffer_mutation_does_not_touch_cpu_model(self):
        x,t=probe.unit_tet();cpu=probe.core.NonlinearDynamics(x,t,1200.,.3,600.)
        wrapped,_=probe.make_device_model(cpu,'cpu')
        expected=cpu.points.clone();wrapped.points[0,0]=123
        torch.testing.assert_close(cpu.points,expected,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
