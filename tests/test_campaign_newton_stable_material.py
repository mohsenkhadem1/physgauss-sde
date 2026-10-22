import numpy as np
import pytest
from experiments.campaign_newton_stable_material import stable_material,uniaxial_oracle,p1_tet_response,StableP2Block
from experiments.campaign_static_uniaxial_bar import lame


def test_rest_stress_tangent_matches_authored_E_nu_and_objectivity():
    E=5000.;nu=.35;mu,lam=lame(E,nu);I=np.eye(3);W,P,A=stable_material(I,E,nu)
    assert W==0;np.testing.assert_array_equal(P,np.zeros((3,3)))
    G=np.array([[.1,.2,-.3],[.05,-.03,.01],[.15,.02,.07]])
    np.testing.assert_allclose(np.einsum('iJkL,kL->iJ',A,G),mu*(G+G.T)+lam*np.trace(G)*I,atol=2e-12)
    a=.8;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    F=I+G*.3;w,p,_=stable_material(F,E,nu);wr,pr,_=stable_material(Q@F,E,nu)
    assert wr==pytest.approx(w,abs=1e-12);np.testing.assert_allclose(pr,Q@p,atol=3e-12)
    wr,pr,_=stable_material(Q,E,nu);assert abs(wr)<1e-12;assert np.linalg.norm(pr)<1e-11


def test_full_energy_force_tangent_and_vbd_exact_self_block():
    rng=np.random.default_rng(93);F=np.eye(3)+rng.normal(size=(3,3))*.04;direction=rng.normal(size=(3,3));direction/=np.linalg.norm(direction);h=1e-6
    w,p,A=stable_material(F,5000,.35);plus=stable_material(F+h*direction,5000,.35);minus=stable_material(F-h*direction,5000,.35)
    assert (plus[0]-minus[0])/(2*h)==pytest.approx(np.sum(p*direction),rel=1e-8,abs=1e-8)
    np.testing.assert_allclose((plus[1]-minus[1])/(2*h),np.einsum('iJkL,kL->iJ',A,direction),rtol=1e-8,atol=2e-6)
    mu,lam=lame(5000,.35);cof=np.linalg.det(F)*np.linalg.inv(F).T
    truncated=mu*np.einsum('ik,JL->iJkL',np.eye(3),np.eye(3))+(lam+mu)*np.einsum('iJ,kL->iJkL',cof,cof)
    g=np.array([1.,-2.,.7]);np.testing.assert_allclose(np.einsum('J,iJkL,L->ik',g,A,g),np.einsum('J,iJkL,L->ik',g,truncated,g),atol=2e-11)
    different=np.array([.2,.1,-.3]);assert np.linalg.norm(np.einsum('J,iJkL,L->ik',g,A-truncated,different))>1
    X=np.array([[0,0,0],[.1,0,0],[0,.08,0],[0,0,.1]]);x=X@F.T;u,f,H,_=p1_tet_response(X,x,5000,.35);v=rng.normal(size=x.shape);eps=1e-7
    pp=p1_tet_response(X,x+eps*v,5000,.35);mm=p1_tet_response(X,x-eps*v,5000,.35)
    assert (pp[0]-mm[0])/(2*eps)==pytest.approx(-np.sum(f*v),rel=1e-7)
    np.testing.assert_allclose(-(pp[1]-mm[1]).ravel()/(2*eps),H@v.ravel(),rtol=2e-7,atol=1e-7)


def test_independent_homogeneous_oracle_and_P2_exact_solution():
    m=StableP2Block((1,2,2),young=5000.,gravity=0.,traction=2.,order=3)
    m.mask[:]=False;m.mask[:,0]=abs(m.rest[:,0])<1e-13;m.mask[:,1]=abs(m.rest[:,1])<1e-13;m.mask[:,2]=abs(m.rest[:,2])<1e-13;m.free=np.flatnonzero(~m.mask.ravel())
    o=uniaxial_oracle(2.);assert .06<o['stretch']-1<.09;assert abs(o['lateral_stress_Pa'])<1e-10
    a=m.solve();exact=m.rest*np.array([o['stretch'],o['lateral_stretch'],o['lateral_stretch']]);np.testing.assert_allclose(a['positions'],exact,atol=2e-12,rtol=0)
    # Stable law polynomial is defined at J=0, unlikelogJ. PositiveJstatic
    # branch is a separately declared guard, not an altered constitutive law.
    W,P,A=stable_material(np.diag([0.,1.,1.]),5000,.35);assert np.isfinite(W) and np.isfinite(P).all() and np.isfinite(A).all()
    with pytest.raises(ValueError,match='Jacobian'):m.elastic(m.rest@np.diag([-.2,1,1]))
