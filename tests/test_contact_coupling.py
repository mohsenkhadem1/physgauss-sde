import numpy as np
from scripts.contact_benchmark_v12.coupled_qualification import CoupledSystem,run


def test_contact_reactions_and_friction_power():
    s=CoupledSystem();s.center=np.array([-.03,0,.045]);s.v[:]=[.1,.2,0]
    f,r,t,depth,power,count=s.contact_forces()
    assert depth>0 and count>0 and power<0
    np.testing.assert_allclose(f.sum(0)+r,0,atol=1e-14)
    np.testing.assert_allclose(np.cross(s.x,f).sum(0)+np.cross(s.center,r)+t,0,atol=1e-14)


def test_no_tensile_contact_when_separated():
    s=CoupledSystem();s.center[:]=[1,0,0]
    f,r,t,depth,power,count=s.contact_forces()
    assert count==0 and depth==0
    np.testing.assert_array_equal(f,np.zeros_like(f));np.testing.assert_array_equal(r,np.zeros(3))


def test_isolated_momentum_balance():
    r=run(dt=2e-4,duration=.1,mode='isolated')
    assert r['contact_samples']>0
    assert r['max_linear_momentum_error']<1e-10
    assert r['max_angular_momentum_error']<1e-10
    assert r['friction_work_j']<0
