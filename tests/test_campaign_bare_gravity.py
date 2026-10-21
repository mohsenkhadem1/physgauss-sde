import unittest
import numpy as np
from experiments.campaign_bare_gravity import integrate_held_acceleration,clock_audit

class GravityReferenceTests(unittest.TestCase):
    def test_mass_independent_uniform_acceleration_and_both_position_conventions(self):
        a=np.tile([.01,-.02,.005],(100,1));dt=np.full(100,.01)
        v,exact,kick=integrate_held_acceleration(a,dt)
        np.testing.assert_allclose(v[-1],a[0],atol=1e-16)
        np.testing.assert_allclose(exact[-1],.5*a[0],atol=1e-16)
        np.testing.assert_allclose(kick[-1],.505*a[0],atol=1e-16)

    def test_nonuniform_actual_dt_uses_each_observed_advance(self):
        a=np.array([[1.,0,0],[0,2,0],[-1,0,1.]])
        dt=np.array([.1,.2,.05]);v,p,k=integrate_held_acceleration(a,dt)
        np.testing.assert_allclose(v[-1],[.05,.4,.05])
        np.testing.assert_allclose(p[-1],[.02875,.06,.00125])
        with self.assertRaises(ValueError):integrate_held_acceleration(a,np.array([.1,0,.05]))

    def test_clock_uses_float32_actual_step_not_double_nominal_getter(self):
        dt=float(np.float32(1/120));after=np.arange(1,361)*dt;before=np.arange(360)*dt
        self.assertLess(clock_audit(before,after,120)['max_cumulative_error_vs_float32_dt_s'],1e-15)
        broken=after.copy();broken[200]+=.001
        with self.assertRaises(ValueError):clock_audit(before,broken,120)
