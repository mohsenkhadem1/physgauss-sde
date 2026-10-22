import numpy as np
import pytest
from experiments.campaign_newton_vbd_transient_v02 import helper, physical_velocity


def test_BDF2_elimination_matches_independent_first_order_block_system():
    h=.013;mass=.7;stiffness=23.;force=.4
    q=np.array([[.12]]);oldq=np.array([[.117]]);v=np.array([[.2]]);oldv=np.array([[.23]])
    heff,hv=helper('BDF2',2,h,q,v,oldq,oldv)
    y=q+heff*hv
    qnew=(mass*y/heff**2+force)/(mass/heff**2+stiffness)
    vnew=physical_velocity('BDF2',2,h,qnew,q,oldq)
    # Solve original D2q=v, M D2v=F-Kq simultaneously, independent of elimination.
    A=np.array([[3/(2*h),-1.],[stiffness,3*mass/(2*h)]])
    rhs=np.array([float((4*q-oldq).item()/(2*h)),force+mass*(4*v-oldv).item()/(2*h)])
    expected=np.linalg.solve(A,rhs)
    np.testing.assert_allclose([qnew.item(),vnew.item()],expected,atol=2e-15,rtol=0)


def test_BE_startup_and_BDF2_quadratic_chord_are_exact():
    h=.01;q=np.array([[.3,-.2,.1]]);v=np.array([[.4,.1,-.3]])
    heff,hv=helper('BDF2',1,h,q,v)
    assert heff==h
    np.testing.assert_array_equal(hv,v)
    hv[:]=0;np.testing.assert_array_equal(v,[[.4,.1,-.3]])
    # BDF2 differentiates an exact quadratic at the new time.
    f=lambda t:np.array([[.2+3*t+2*t*t,-.1+t*t,.4-2*t]])
    np.testing.assert_allclose(physical_velocity('BDF2',3,h,f(3*h),f(2*h),f(h)),[[3+4*3*h,2*3*h,-2]],atol=6e-15)
    with pytest.raises(ValueError):helper('wrong',1,h,q,v)
