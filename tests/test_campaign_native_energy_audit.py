import numpy as np
import pytest
from experiments.campaign_native_energy_audit import held_force_work,continuum_proxies


def test_held_force_work_includes_final_step_and_world_rotation_invariance():
    before=np.array([[[0.,0.,0.]],[[.2,0.,0.]],[[.5,.1,0.]]])
    force=np.array([[[2.,0.,0.]],[[3.,1.,0.]],[[0.,4.,0.]]])
    final=np.array([[.5,.4,0.]])
    work,_=held_force_work(force,before,final,np.arange(4))
    np.testing.assert_allclose(work,[0.,.4,1.4,2.6],atol=1e-15)
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rotated,_=held_force_work(force@R.T,before@R.T+[2.,3.,4.],final@R.T+[2.,3.,4.],np.arange(4))
    np.testing.assert_allclose(rotated,work,atol=1e-14)


def test_corotated_proxy_has_zero_rigid_energy_and_exact_uniform_dilation():
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    tets=np.array([[0,1,2,3]]);angle=.8
    R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    E,nu,rho=1200.,.3,600.;scale=1.01
    q=np.stack([rest@R.T+[2.,3.,4.],rest*scale]);v=np.ones_like(q)*np.array([.2,.3,.4])
    U,T,mass=continuum_proxies(rest,tets,q,v,E,nu,rho)
    mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    expected=(3*mu+4.5*lam)*(scale-1)**2/6
    assert U[0]<1e-25;assert U[1]==pytest.approx(expected)
    assert mass==pytest.approx(rho/6)
    np.testing.assert_allclose(T,.5*mass*(.2**2+.3**2+.4**2))


def test_inverted_states_and_nonintegral_frame_indices_are_rejected():
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    q=rest.copy();q[1,0]=-1
    with pytest.raises(ValueError,match='Inverted'):continuum_proxies(rest,[[0,1,2,3]],q[None],np.zeros((1,4,3)),1000.,.3,500.)
    with pytest.raises(ValueError,match='integer'):held_force_work(np.zeros((2,1,3)),np.zeros((2,1,3)),np.zeros((1,3)),np.array([.5]))
