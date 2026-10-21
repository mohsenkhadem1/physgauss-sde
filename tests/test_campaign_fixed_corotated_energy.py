import unittest
import numpy as np
from experiments.campaign_fixed_corotated_energy import fixed_corotated_energy

class FixedCorotatedEnergyTests(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        self.tets=np.array([[0,1,2,3]])
        self.E=4249.367739516093;self.nu=.3407184360170016
        self.mu=self.E/(2*(1+self.nu));self.lam=self.E*self.nu/((1+self.nu)*(1-2*self.nu))

    def test_uniform_volume_dilation_uses_det_term(self):
        scale=1.1;q=(self.rest*scale)[None]
        actual=fixed_corotated_energy(self.rest,self.tets,q,self.E,self.nu)[0]
        expected=(self.mu*3*(scale-1)**2+.5*self.lam*(scale**3-1)**2)/6
        self.assertAlmostEqual(actual,expected,places=12)
        common=(self.mu*3*(scale-1)**2+.5*self.lam*(3*scale-3)**2)/6
        self.assertGreater(abs(actual-common),.01*actual)

    def test_small_strain_tangent_matches_isotropic_linear_elasticity(self):
        A=np.array([[.7,.2,0],[.2,-.1,.1],[0,.1,.3]])
        expected=(self.mu*np.sum(A*A)+.5*self.lam*np.trace(A)**2)/6
        eps=np.array([1e-2,1e-3,1e-4]);q=np.array([self.rest@(np.eye(3)+e*A).T for e in eps])
        tangent=fixed_corotated_energy(self.rest,self.tets,q,self.E,self.nu)/(eps**2)
        error=np.abs(tangent/expected-1)
        self.assertTrue(np.all(error[1:]<.11*error[:-1]))
        self.assertLess(error[-1],1e-4)

    def test_proper_rotation_has_zero_energy_and_inversion_is_rejected(self):
        theta=.7;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
        self.assertLess(fixed_corotated_energy(self.rest,self.tets,(self.rest@R.T)[None],self.E,self.nu)[0],1e-24)
        reflected=self.rest.copy();reflected[:,0]*=-1
        with self.assertRaises(ValueError):fixed_corotated_energy(self.rest,self.tets,reflected[None],self.E,self.nu)
