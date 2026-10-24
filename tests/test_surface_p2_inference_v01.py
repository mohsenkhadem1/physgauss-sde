"""Synthetic controls only; no production observations or posterior outcomes."""
import numpy as np
import pytest
from scipy.special import ndtr, ndtri
from experiments.campaign_surface_p2_inference_v01 import (
    LabelledModal, IntegrationPlan, qualify_integrals, physical_parameters,
    cache_storage_estimate)
from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference


def test_analytic_bounded_gaussian_and_predictive_moments():
    mu=np.array([.43,.61]);sd=np.array([.13,.17])
    def ll(u):return -.5*np.sum(((u-mu)/sd)**2,axis=1)-np.log(sd*np.sqrt(2*np.pi)).sum()
    def predict(u):return 1e-3*u
    answer=qualify_integrals(ll,predict,plan=IntegrationPlan((33,65,129),(24,48,96),512))
    assert answer['qualified']
    final=answer['local_final_simpson'];norm=ndtr((1-mu)/sd)-ndtr(-mu/sd)
    assert abs(final['log_evidence']-np.log(norm).sum())<1e-7
    a=-mu/sd;b=(1-mu)/sd;phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    mean=mu+sd*(phi(a)-phi(b))/norm
    variance=sd**2*(1+(a*phi(a)-b*phi(b))/norm-((phi(a)-phi(b))/norm)**2)
    assert np.max(abs(final['predictive_mean']-mean*1e-3))<1e-10
    assert np.max(abs(final['predictive_SD']-np.sqrt(variance)*1e-3))<1e-10
    q=mu[:,None]+sd[:,None]*ndtri(ndtr(a)[:,None]+norm[:,None]*np.array([.05,.5,.95]))
    assert np.max(abs(final['unit_quantiles']-q))<.0003
    assert len(answer['diagnostics']['levels']['Simpson'])==3


def test_underresolved_posterior_fails_closed_keeps_negative_levels():
    def ll(u):return -.5*np.sum(((u-[.37213,.61937])/.0007)**2,axis=1)
    answer=qualify_integrals(ll,lambda u:u*1e-3,
        plan=IntegrationPlan((9,17),(8,16),64))
    assert not answer['qualified']
    assert answer['diagnostics']['qualified'] is False
    assert len(answer['diagnostics']['levels']['Simpson'])==2
    assert answer['diagnostics']['successive']['Simpson'][-1]['passed'] is False


def test_label_selection_repeated_independent_joint_observations():
    modal=ModalStaticReference(np.diag([2.,3.,4.]),np.diag([1.,.2,.7]),
        np.array([[1.,2.],[3.,4.],[5.,6.]]),np.eye(3))
    model=LabelledModal(modal,['A','B']);u=np.array([[.2,.7],[.8,.3]])
    expected=modal(physical_parameters(u)).transpose(0,2,1)
    assert np.allclose(model.select(['B','A'])(u),expected[:,[1,0],:])
    y=np.array([1.,2.,3.])*1e-5;z=y+.3e-5
    ll=model.loglikelihood([('A',y),('A',z),('B',y)])
    residual=np.stack([expected[:,0]-y,expected[:,0]-z,expected[:,1]-y],axis=1)/20e-6
    direct=-.5*np.sum(residual**2,axis=(1,2))-9*np.log(20e-6*np.sqrt(2*np.pi))
    assert np.allclose(ll(u),direct,rtol=1e-14)
    assert model._selected[('B','A')].shape==(6,3)
    model.select(['A'])
    assert model._selected[('A',)].shape==(3,3)
    with pytest.raises(ValueError):model.loglikelihood([('A',y)],sigma=1e-5)


def test_extra_exact_schur_particular_term_not_lost():
    class Particular:
        output_shape=(2,2);modes=2
        eigenvalues=np.array([.1,.2,0.])
        coefficients=np.arange(12.).reshape(4,3)
    model=LabelledModal(Particular(),['A','B']);u=np.array([[.2,.3]])
    theta=physical_parameters(u);full=(1/(theta[:,0,None]+theta[:,1,None]*model.modal.eigenvalues))@model.modal.coefficients.T
    assert np.allclose(model.select(['B'])(u),full.reshape(1,2,2).transpose(0,2,1)[:,1:2])


def test_p2_tiny_force_direct_sparse_parity():
    from experiments.campaign_surface_p2_static_v01 import SparseStaticForward, surface_traction, field_queries
    fwd=SparseStaticForward(1)
    points=np.array([[.09,.02,.08],[.24,.08,.02]])
    loads={'tiny_normal':surface_traction(1,[0,0,-1],1e-5)[0],
           'tiny_axial':surface_traction(1,[1,0,0],1e-5)[0]}
    model=LabelledModal.from_forward(fwd,loads=loads,points=points)
    unit=np.array([[.25,.65],[.7,.2]])
    u=fwd.staticforward(physical_parameters(unit),np.array(list(loads.values())))
    expected=field_queries(1,u,points).reshape(2,2,-1)
    assert np.max(abs(model.select(tuple(loads))(unit)-expected))<1e-17
    assert model.modal.diagnostics['all_boundary_modes_retained']


def test_plan_validation_and_cache_budget():
    with pytest.raises(ValueError):IntegrationPlan((33,17),(16,32))
    with pytest.raises(ValueError):physical_parameters([[1.1,.2]])
    budget=cache_storage_estimate()
    assert budget['total_counted_bytes']<4*1024**3
    assert not budget['real_cache_execution_authorized']
    assert budget['raw_response_tables_bytes']==(1025**2+128**2+256**2+512**2)*240*8
