import numpy as np
from scipy import sparse
from experiments.campaign_static_p2_block import shapes,duffy
from experiments.campaign_p2_consistent_transient import ConsistentP2Transient,P2State,advance_consistent,force_coefficient


def test_full_mass_independent_positive_quadrature_and_spd_coupling():
    c=ConsistentP2Transient((1,1,1));m=c.model;b,w=duffy(4);N=shapes(b);expected=np.zeros(c.mass.shape)
    for T,V in zip(m.elements,m.volumes):expected[np.ix_(T,T)]+=m.density*V*np.einsum('q,qa,qb->ab',w,N,N)
    np.testing.assert_allclose(c.mass.toarray(),expected,atol=1e-16,rtol=1e-13)
    np.testing.assert_allclose(expected,expected.T,atol=1e-17);assert np.linalg.eigvalsh(expected).min()>0
    assert np.any(expected-np.diag(np.diag(expected))<0)
    assert np.any(expected.sum(1)<0) and abs(expected.sum()-.96)<2e-15
    np.testing.assert_allclose(c.mass@np.ones(len(m.rest)),m.density*m.body_integral_weights,atol=2e-16)


def test_direct_displacement_elasticity_matches_frozen_physical_assembler():
    c=ConsistentP2Transient((1,1,1));m=c.model;u=m.rest@np.diag([.03,-.006,.004]);u[:,1]+=.002*np.sin(m.rest[:,0]*8)
    a=c.elastic_u(u,True);b=m.elastic(m.rest+u,True)
    np.testing.assert_allclose(a[0],b[0],atol=2e-15);np.testing.assert_allclose(a[1],b[1],atol=3e-14);np.testing.assert_allclose(a[2].toarray(),b[2].toarray(),atol=2e-12)
    d=np.random.default_rng(17).normal(size=u.shape);eps=1e-7
    ap=c.elastic_u(u+eps*d);am=c.elastic_u(u-eps*d)
    np.testing.assert_allclose((ap[0]-am[0])/(2*eps),np.sum(a[1]*d),atol=3e-9)
    np.testing.assert_allclose((ap[1]-am[1])/(2*eps),(a[2]@d.ravel()).reshape(u.shape),atol=2e-7,rtol=1e-8)


def test_BDF2_full_coupled_mass_against_independent_first_order_modal_oracle():
    M=np.array([[2.,.4],[.4,1.3]]);K=np.array([[7.,-1.],[-1.,4.]]);mass=sparse.csr_matrix(M);mask=np.ones((2,3),bool);mask[:,0]=False;F=np.zeros((2,3));F[:,0]=[.1,-.03];s=P2State(np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,3)));prev=None
    A=np.block([[np.zeros((2,2)),np.eye(2)],[-np.linalg.solve(M,K),np.zeros((2,2))]]);drive=np.r_[np.zeros(2),np.linalg.solve(M,F[:,0])];Y=np.zeros(4);Yprev=None;h=.02
    def elastic(u,tangent=False):
        g=np.zeros_like(u);g[:,0]=K@u[:,0];H=np.zeros((6,6));H[np.ix_([0,3],[0,3])]=K
        return .5*u[:,0]@K@u[:,0],g,sparse.csr_matrix(H) if tangent else None,np.ones(1)
    for n in range(12):
        expected=np.linalg.solve(np.eye(4)-h*A,Y+h*drive) if n==0 else np.linalg.solve(3*np.eye(4)-2*h*A,4*Y-Yprev+2*h*drive)
        new,d=advance_consistent(elastic,mass,mask,s,prev,h,F,1.)
        np.testing.assert_allclose(np.r_[new.u[:,0],new.v[:,0]],expected,atol=2e-14)
        assert d['physical_original_free_R_N']<1e-12
        prev,s=s,new;Yprev,Y=Y,expected


def test_real_consistent_P2_step_residual_and_exact_clamp_known_force():
    c=ConsistentP2Transient((1,1,1));s=c.initial();prev=None
    for i in range(3):
        f=c.model.load*force_coefficient('smooth_ramp',(i+1)/480);new,d=c.step(s,prev,1/480,f)
        assert d['objective_free_R_N']<=1e-10 and d['physical_original_free_R_N']<=1e-9 and d['minimum_J']>.1
        np.testing.assert_array_equal(new.u[c.model.mask],0.);np.testing.assert_array_equal(new.v[c.model.mask],0.)
        prev,s=s,new
    assert np.linalg.norm(s.u)>0
    assert force_coefficient('step',0)==1 and force_coefficient('smooth_ramp',0)==0 and force_coefficient('smooth_ramp',.05)==1
