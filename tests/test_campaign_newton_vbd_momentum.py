import numpy as np
import pytest
from experiments.campaign_newton_vbd_momentum import momentum_velocity,kinematic_defect


def oscillator(n=200,h=.0005):
    # Independent 2x2 linear-system oracle, not the eliminated helper formula.
    m=.7;k=23.;force=.4;rest=.2;q=rest;v=0.;oldq=oldv=None;Q=[q];V=[v]
    for step in range(1,n+1):
        c=1/h if step==1 else 3/(2*h)
        rq=q/h if step==1 else (4*q-oldq)/(2*h)
        rv=m*v/h if step==1 else m*(4*v-oldv)/(2*h)
        new=np.linalg.solve(np.array([[c,-1],[k,m*c]]),[rq,force+k*rest+rv])
        oldq,oldv=q,v;q,v=new;Q.append(q);V.append(v)
    return np.array(Q),np.array(V),m,k,force,rest


def xyz(x):return np.array([[x,0.,0.]])


def test_momentum_and_kinematics_equal_exact_discrete_oscillator_oracle():
    Q,V,m,k,f,r=oscillator(40,h=.002);v=xyz(V[0]);oldv=None
    for step in range(1,len(Q)):
        new=momentum_velocity('BDF2',step,.002,v,oldv,xyz(-k*(Q[step]-r)),xyz(f),[m],np.array([True]))
        np.testing.assert_allclose(new,xyz(V[step]),atol=2e-13,rtol=0)
        defect=kinematic_defect('BDF2',step,.002,xyz(Q[step]),xyz(Q[step-1]),xyz(Q[step-2]) if step>1 else None,new)
        assert abs(defect).max()<2e-16
        oldv,v=v,new


def test_float32_position_quantization_is_unchanged_recovery_has_small_defect():
    Q,V,m,k,f,r=oscillator();q=Q.astype(np.float32).astype(np.float64);unchanged=q.copy();v=xyz(0.);oldv=None;rec=[0.];defects=[];chord=[0.];h=.0005
    for step in range(1,len(q)):
        new=momentum_velocity('BDF2',step,h,v,oldv,xyz(-k*(q[step]-r)),xyz(f),[m],np.array([True]))
        defects.append(abs(kinematic_defect('BDF2',step,h,xyz(q[step]),xyz(q[step-1]),xyz(q[step-2]) if step>1 else None,new)).max())
        chord.append((q[step]-q[step-1])/h if step==1 else (3*(q[step]-q[step-1])-(q[step-1]-q[step-2]))/(2*h));rec.append(new[0,0]);oldv,v=v,new
    np.testing.assert_array_equal(q,unchanged)
    # Controlled supplied-q error diagnosis, not a promise for a nonlinear solve.
    assert max(defects)<1e-7
    assert np.max(abs(np.asarray(rec)-V))<1e-7
    assert np.max(abs(np.asarray(chord)-V))>100*np.max(abs(np.asarray(rec)-V))


def test_fixed_nodes_zero_causal_history_inputs_not_mutated_and_invalid_rejected():
    v=np.array([[.1,.2,.3],[1,2,3.]]);old=np.zeros_like(v);original=v.copy();active=np.array([True,False]);m=np.ones(2)
    answer=momentum_velocity('BDF2',2,.01,v,old,np.zeros_like(v),np.ones_like(v),m,active)
    np.testing.assert_array_equal(answer[1],0);np.testing.assert_array_equal(v,original)
    with pytest.raises(ValueError):momentum_velocity('BDF2',2,.01,v,None,np.zeros_like(v),np.zeros_like(v),m,active)


def test_integrated_position_and_residual_defect_identity():
    from experiments.campaign_newton_vbd_momentum import kinematic_position
    h=.001;m=.7;q=xyz(.202);oldq=xyz(.201);v=xyz(.1);oldv=xyz(.09);qnew=xyz(.20217);f=xyz(.4)
    vm=momentum_velocity('BDF2',2,h,v,oldv,xyz(0),f,[m],np.array([True]));heff=2*h/3
    chord=(3*(qnew-q)-(q-oldq))/(2*h);R=m*(chord-(4*v-oldv)/3)/heff-f
    np.testing.assert_allclose(chord-vm,heff*R/m,atol=5e-16,rtol=0)
    defect=kinematic_defect('BDF2',2,h,qnew,q,oldq,vm)
    np.testing.assert_allclose(defect,heff**2*R/m,atol=1e-18,rtol=0)
    np.testing.assert_allclose(qnew-kinematic_position('BDF2',2,h,q,oldq,vm),defect,atol=4*np.spacing(qnew[0,0]),rtol=0)
