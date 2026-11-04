"""Lightweight independent contracts for the audited object backend revision."""
import numpy as np
import pytest
import torch
from scripts.gaussian_boundary_flow_v02.object_backend import ObjectBoundaryBackend
from scripts.gaussian_boundary_flow_v02.object_backend_revision import ObjectBoundaryBackendRevision


def payload():
    nodes=np.array(list(__import__('itertools').product((-.1,.1),repeat=3)),float)
    return {'nodes':nodes,'observed':np.tile(nodes,(3,1,1)),'train_end':3,
            'observation_matrix':np.eye(8),'observed_initial':nodes.copy(),
            'attachment_weights':np.ones((1,8)),
            'controller_targets':np.tile(nodes,(4,1,1,1)),
            'attachment_active':np.ones((4,1)),'floor':-.1}


def test_explicit_floor_off_removes_latent_and_penalty_exactly():
    on=ObjectBoundaryBackendRevision(payload(),floor_mode='inferred')
    off=ObjectBoundaryBackendRevision(payload(),floor_mode='off')
    assert on.parameter_dim==4 and off.parameter_dim==3
    assert not any('floor' in s for s in off.names)
    th_on=on.bind(np.zeros(4));th_off=off.bind(np.zeros(3))
    assert th_off[off.model.i_floor].exp()==0
    x=on.model.rest.clone();x[:,2]-=.1
    _,ron,Hon=on.model.evaluate(x,th_on,0,True)
    _,roff,Hoff=off.model.evaluate(x,th_off,0,True)
    assert not on.model.contact and not off.model.contact
    assert (ron[:,2]-roff[:,2]).sum()<0
    # At the translated shape, the difference is exactly the normal floor law.
    penetration=torch.minimum(x[:,2]-on.model.floor,torch.zeros_like(x[:,2]))
    expected=th_on[on.model.i_floor].exp()*on.model.mass*penetration
    assert torch.allclose(ron[:,2]-roff[:,2],expected,atol=1e-12)


def test_inferred_revision_matches_base_parameter_binding():
    old=ObjectBoundaryBackend(payload());new=ObjectBoundaryBackendRevision(payload(),floor_mode='inferred')
    z=np.array([-.2,.6,.9,-.7]);a=old.bind(z);b=new.bind(z)
    assert torch.equal(a,b)
    assert torch.equal(old.model.weights,new.model.weights)


def test_canonical_schedule_is_invariant_even_for_path_sensitive_solver():
    class Fake:
        scale=1.;origin=0.
        def solve(self,theta,frame,initial,maxiter):
            state=torch.zeros(4,3) if initial is None else initial
            state=state+(frame+1)**2
            return state,None,{'residual':0.,'minJ':1.,'iterations':1}
    b=ObjectBoundaryBackendRevision.__new__(ObjectBoundaryBackendRevision)
    b.model=Fake();b.data={'controller_targets':np.zeros((5,1,4,3))};b.bind=lambda z:torch.zeros(1)
    sparse,_=b.rollout(np.zeros(1),np.array([0,2,4]));dense,_=b.rollout(np.zeros(1),np.arange(5))
    assert np.array_equal(sparse,dense[[0,2,4]])
    assert b.last_path_diagnostics['frames_solved']==5
    with pytest.raises(ValueError):b.rollout(np.zeros(1),[2,1])


def test_projection_must_precede_posterior_quantiles():
    # Each node's rare excursion is below its marginal 95% quantile, but their
    # disjoint events exceed 5% at an interpolated observation point.
    samples=np.zeros((100,2));samples[:4,0]=1;samples[4:8,1]=1
    A=np.array([.5,.5])
    wrong=A@np.quantile(samples,.95,axis=0)
    correct=np.quantile(samples@A,.95)
    assert wrong==0 and correct==.5
