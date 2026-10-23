import numpy as np
import pytest
from experiments.campaign_long_trajectory_design import schedule,convergence,full_mass_momentum_ledger


def test_nine_clocks_are_exact_accepted_endpoints_and_not_midpoint_force_clocks():
    for n in [32,64,128]:
        s=schedule(n)
        np.testing.assert_array_equal(s['clock_steps']/n,s['clocks_s'])
        assert len(s['midpoint_times_s'])==n and not set(s['clocks_s'])&set(s['midpoint_times_s'])
    with pytest.raises(ValueError):schedule(48)


def test_manufactured_second_order_q_and_v_then_velocity_only_failure():
    z=np.zeros((9,63,3));q=[z+a for a in [8e-6,2e-6,.5e-6]];v=[z+a for a in [8e-6,2e-6,.5e-6]]
    a=convergence(*q,*v,z,z+1e-6,z,z+2e-6,.02,.016)
    assert a['passed'] and a['quantities']['q']['observed_temporal_order']==pytest.approx(2.)
    b=convergence(*q,z,z+1e-3,z+2e-3,z,z,z,z,.02,.016)
    assert not b['passed'] and b['quantities']['q']['passed'] and not b['quantities']['v']['passed']
    c=convergence(z,z,z,z,z,z,z,z+4e-6,z,z,.02,.016)
    assert not c['passed'] # each-coordinate comparison would wrongly pass: vector norm matters
    with pytest.raises(ValueError):convergence(*q,*v,z,z,z,z,.016,.02)


def test_full_mass_angular_momentum_and_reaction_control_detects_row_lumping():
    # Manufactured free three-body constant nodal forces with non-diagonal mass.
    # Exact midpoint update independently formed from M^-1 F; no FE/source solver.
    M=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.]])
    x=np.array([[0.,0.,0.],[1.,0.,.1],[0.,1.,.2]])
    u=np.zeros_like(x);v=np.array([[.1,.2,0.],[-.3,.1,.1],[.1,-.2,.3]])
    F=np.array([[.2,-.1,.3],[.1,.2,-.2],[-.1,.1,.2]]);R=np.zeros_like(x);h=.03
    w=v+h*np.linalg.solve(M,F);y=u+h*(v+w)/2
    a=full_mass_momentum_ledger(x,u,v,y,w,M,F,R,h)
    assert a['linear_residual_N_s']<1e-15 and a['angular_residual_N_m_s']<1e-15 and a['kinematic_residual_m']<1e-15
    from scipy import sparse
    sparse_row=full_mass_momentum_ledger(x,u,v,y,w,sparse.csr_matrix(M),F,R,h)
    assert all(abs(sparse_row[k]-a[k])<1e-15 for k in a)
    b=full_mass_momentum_ledger(x,u,v,y,w,np.diag(M.sum(1)),F,R,h)
    assert b['angular_residual_N_m_s']>1e-4
    # Constrain node0 exactly, keeping the inertial M_cf reaction.
    w=v.copy();v[0]=0;w[0]=0;w[1:]=v[1:]+h*np.linalg.solve(M[1:,1:],F[1:]);y=u+h*(v+w)/2
    R[0]=(M@((w-v)/h)-F)[0]
    c=full_mass_momentum_ledger(x,u,v,y,w,M,F,R,h)
    assert c['linear_residual_N_s']<1e-15 and c['angular_residual_N_m_s']<1e-15
    d=full_mass_momentum_ledger(x,u,v,y,w,M,F,np.zeros_like(R),h)
    assert d['linear_residual_N_s']>1e-4


def test_public_C3_pulse_actual_history_q_and_v_against_independent_duhamel_integral():
    from scipy.integrate import quad
    from experiments.campaign_p2_smooth_midpoint import smooth_bump
    # Constant gravity equilibrium removed analytically; force is known C3 pulse.
    omega=2.7;A=np.array([[0.,1.],[-omega**2,0.]])
    clocks=np.arange(9)/8
    reference=np.array([[quad(lambda s:np.sin(omega*(t-s))/omega*256*s**4*(1-s)**4,0,t,epsabs=1e-14)[0],
                         quad(lambda s:np.cos(omega*(t-s))*256*s**4*(1-s)**4,0,t,epsabs=1e-14)[0]] for t in clocks])
    errors=[]
    for n in [32,64,128]:
        h=1/n;state=np.zeros(2);saved=[state.copy()]
        for k in range(n):
            f=np.array([0.,smooth_bump((k+.5)*h)])
            state=np.linalg.solve(np.eye(2)-h*A/2,(np.eye(2)+h*A/2)@state+h*f)
            if (k+1)%(n//8)==0:saved.append(state.copy())
        errors.append(abs(np.array(saved)-reference).max(0))
    errors=np.array(errors)
    assert np.all(errors[:-1]/errors[1:]>3.9)
    # Independent expanded degree8 polynomial has first three endpoint derivatives0.
    from numpy.polynomial import Polynomial
    p=256*Polynomial([0,0,0,0,1])*Polynomial([1,-4,6,-4,1])
    for order in range(4):
        assert p.deriv(order)(0)==0 and p.deriv(order)(1)==0
