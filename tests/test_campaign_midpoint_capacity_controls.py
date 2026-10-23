import numpy as np
from scipy import sparse
from experiments.campaign_midpoint_capacity_controls import cpu_midpoint


def test_sparse_halfstep_reconstruction_matches_independent_constrained_cayley_and_reactions():
    M=np.array([[2.,.3],[.3,1.]]);K=np.array([[4.,-1.],[-1.,3.]]);mask=np.zeros((2,3),bool);mask[0]=True;F=np.array([[0.,0.,0.],[.07,-.04,.1]]);u=np.array([[0.,0.,0.],[.03,.02,-.01]]);v=np.array([[0.,0.,0.],[-.02,.01,.04]]);h=.02
    def elastic(x,tangent=False):return .5*np.sum(x*(K@x)),K@x,sparse.kron(K,sparse.eye(3),format='csr') if tangent else None,np.ones(1)
    a=cpu_midpoint(elastic,sparse.csr_matrix(M),mask,u,v,h,F,1.)
    A=np.block([[np.zeros((3,3)),np.eye(3)],[-3*np.eye(3),np.zeros((3,3))]]);y=np.r_[u[1],v[1]];b=np.r_[np.zeros(3),F[1]];expected=np.linalg.solve(np.eye(6)-h*A/2,(np.eye(6)+h*A/2)@y+h*b)
    np.testing.assert_allclose(np.r_[a['u'][1],a['v'][1]],expected,atol=2e-15,rtol=1e-13)
    assert np.array_equal(a['u'][0],u[0]) and np.array_equal(a['v'][0],v[0])
    r=M@((a['v']-v)/h)+K@a['midpoint_u']-F
    np.testing.assert_allclose(a['ledger']['fixed_reaction_resultant_N'],r[0],atol=1e-14)
    assert abs(a['ledger']['ledger_identity_defect_J'])<1e-15 and abs(a['ledger']['total_energy_minus_midpoint_work_J'])<1e-15
