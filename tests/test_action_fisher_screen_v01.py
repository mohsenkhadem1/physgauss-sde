from types import SimpleNamespace
import numpy as np
from experiments.campaign_action_fisher_screen_v01 import basis_and_derivative,atlas_jacobian,linear_jacobian,screen
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas,unit_axis
from experiments.campaign_surface_p2_inference_v01 import LabelledModal


def test_analytic_barycentric_derivative_at_nodes_and_interiors():
    x=unit_axis(9);q=np.r_[x,[.123,.789,np.nextafter(1.,0.)]];B,D=basis_and_derivative(q,x)
    for degree in range(9):
        expected=np.zeros_like(q)if degree==0 else degree*q**(degree-1)
        np.testing.assert_allclose(B@(x**degree),q**degree,atol=2e-15)
        np.testing.assert_allclose(D@(x**degree),expected,atol=6e-13)


def test_atlas_derivative_quadratic_polynomial_and_nested_refinement():
    x=unit_axis(9);a,b=np.meshgrid(x,x,indexing='ij');values=(a*a+3*a*b+2*b*b)[:,:,None,None]*np.array([1e-6,2e-6])[None,None,None,:]
    atlas=NonlinearResidualAtlas(values,['force'],provenance_fingerprint='toy');q=np.array([[0,1],[.1,.8],[.5,.5],[1,0]])
    J=atlas_jacobian(atlas,q,['force']);expected=np.stack((2*q[:,0]+3*q[:,1],3*q[:,0]+4*q[:,1]),axis=-1)[:,None,None,:]*np.array([1e-6,2e-6])[None,None,:,None]
    np.testing.assert_allclose(J,expected,atol=2e-19)
    np.testing.assert_allclose(J,atlas_jacobian(atlas.coarsen(5),q,['force']),atol=2e-19)


def test_exact_modal_jacobian_independent_finite_differences_including_zero_mode():
    modal=SimpleNamespace(eigenvalues=np.array([0.,.4,2.]),coefficients=np.arange(12,dtype=float).reshape(4,3)+1,output_shape=(2,2))
    model=LabelledModal(modal,['a','b']);q=np.array([[.2,.7],[.8,.3]])
    J=linear_jacobian(model,q,['b','a']);predict=model.select(['b','a'])
    for h in(1e-3,5e-4):
        for d in range(2):
            step=np.eye(2)[d]*h;fd=(-predict(q+2*step)+8*predict(q+step)-8*predict(q-step)+predict(q-2*step))/(12*h)
            np.testing.assert_allclose(J[...,d],fd,rtol=2e-10,atol=1e-14)


def test_fisher_trace_matches_direct_explicit_matrix_and_duplicate_reset():
    P=np.array([[[2e-4,1e-4],[1e-4,-2e-4]]]);A=np.stack((P,P*.5),axis=1);T=np.array([[[3e-4,1e-4],[2e-4,-1e-4],[0,1e-4]]])
    out=screen(P,A,T);C=np.linalg.inv(12*np.eye(2)+P[0].T@P[0]/(20e-6)**2)
    expected=np.trace(T[0]@C@T[0].T)/3
    np.testing.assert_allclose(out['prefix_risk_m2'],[expected],rtol=2e-15)
    twice=np.linalg.inv(12*np.eye(2)+2*P[0].T@P[0]/(20e-6)**2)
    np.testing.assert_allclose(out['post_risk_m2'][0,0],np.trace(T[0]@twice@T[0].T)/3,rtol=2e-15)
    assert out['winner_mask'].tolist()==[[True,False]]


def test_equal_actions_are_explicit_ties_no_arbitrary_winner():
    P=np.zeros((2,3,2));A=np.ones((2,3,4,2))*1e-5;T=np.ones((2,5,2))*1e-4
    out=screen(P,A,T)
    assert out['winner_mask'].all()and np.all(out['top_margin_m2']==0)
