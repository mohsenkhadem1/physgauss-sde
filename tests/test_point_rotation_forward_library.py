from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from experiments.campaign_point_rotation_forward import PointRotationForward,signed_tet_diagnostics
from experiments.campaign_batched_point_rotation import synthetic,retract_positions
from experiments.campaign_material_gaussians import MaterialGaussians
from experiments import prepare_point_rotation_library as builder


class ToyPacked:
    def __init__(self,rest):self.rest=rest;self.calls=0
    def predict(self,clock,ids,theta,device='cpu'):
        self.calls+=1;X=self.rest.to(device);r=X[None]-X[ids.to(device),None]
        axis=theta.to(device)[:,0,None]*theta.new_tensor([.1,-.2,.05]).to(device)
        u=torch.linalg.cross(axis[:,None].expand_as(r),r,dim=-1)
        phase=torch.linspace(0,1,61,dtype=torch.float64,device=device)
        q=X+u[:,None]*phase[None,:,None,None]
        return dict(positions_m=q,times_s=phase*3,handle_centers_m=torch.ones((len(ids),61,2,3)),
                    handle_rotation_coordinates_rad=torch.ones((len(ids),61,2,3)))


def toy():
    X,_,_=synthetic();rest=torch.tensor(X);packed=ToyPacked(rest)
    ids=torch.tensor([0,2],dtype=torch.int64);theta=torch.tensor([[1.,625.,.35,.6],[.8,625.,.35,.6]],dtype=torch.float64)
    return rest,packed,ids,theta


def test_wrapper_corrects_each_explicit_pin_and_never_claims_corrected_hardware():
    X,packed,ids,theta=toy();wrapper=PointRotationForward(packed,X)
    result=wrapper.predict(None,ids,theta)
    assert set(result)=={'positions_m','times_s'} and packed.calls==1
    for b,pin in enumerate(ids):torch.testing.assert_close(result['positions_m'][b,:,pin],X[pin].expand(61,3),atol=0,rtol=0)
    raw=wrapper.predict(None,ids,theta,return_uncorrected=True)
    expected=retract_positions(X,raw['uncorrected_positions_m'],ids)['positions_m']
    torch.testing.assert_close(raw['positions_m'],expected,atol=0,rtol=0)
    assert not torch.equal(raw['positions_m'],raw['uncorrected_positions_m'])
    theta.requires_grad_();loss=wrapper.predict(None,ids,theta)['positions_m'].square().sum()
    assert torch.isfinite(torch.autograd.grad(loss,theta)[0]).all()


def test_transport_and_full_frame_inversion_diagnostics_agree_without_filtering():
    X,packed,ids,theta=toy();q=PointRotationForward(packed,X).predict(None,ids,theta)['positions_m']
    tets=torch.tensor([[0,1,2,3]],dtype=torch.int64);g=MaterialGaussians.from_tetrahedra(X,tets,625.)
    state=g.transport(q);diagnostic=signed_tet_diagnostics(q,X,tets)
    torch.testing.assert_close(diagnostic['minimum_J'],state.jacobians.amin((1,2)),atol=1e-15,rtol=1e-13)
    assert torch.count_nonzero(diagnostic['nonpositive_tet_frame_count'])==0
    torch.testing.assert_close(state.volumes*state.bulk_density,g.masses.expand_as(state.volumes),atol=1e-16,rtol=1e-13)
    # A finite inverted candidate is recorded for everyframe, never removed/repaired.
    inverted=X[None,None].repeat(1,61,1,1);inverted[:,:,:,0]*=-1
    result=signed_tet_diagnostics(inverted,X,tets)
    assert result['minimum_J'].item()==pytest.approx(-1)
    assert result['nonpositive_tet_frame_count'].item()==61


def test_old_archive_member_allowlist_never_opens_sensor_likelihood_or_weight_members():
    requested=[];ids=np.array([11],dtype=np.int64);theta=np.array([[5000.,625.,.35,.6]])
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __getitem__(self,key):
            requested.append(key)
            if key not in ['geometry_ids','physical_coordinates']:raise AssertionError('Forbidden old observation/prediction/posterior member')
            return ids if key=='geometry_ids' else theta
    with patch.object(builder.np,'load',return_value=Archive()):
        got_ids,got_theta=builder.load_coordinates_only('synthetic-old.npz')
    assert requested==['geometry_ids','physical_coordinates']
    np.testing.assert_array_equal(got_ids,ids);np.testing.assert_array_equal(got_theta,theta)


def test_exact_prior_stratum_mass_and_nested_prefix_no_renormalization_or_dropping():
    p={11:.2,22:.8};ids=np.repeat([11,22],4).astype(np.int64);theta=np.repeat([[5000.,625.,.35,.6]],8,axis=0)
    records=builder.integration_records(ids,theta,p,count=4,prefix=2)
    np.testing.assert_array_equal(records['prefix_indices_256'],[0,1,4,5])
    np.testing.assert_allclose(records['prior_mass_1024'],[.05]*4+[.2]*4,rtol=0,atol=0)
    assert records['prior_mass_1024'].sum()==pytest.approx(1) and records['prior_mass_256'].sum()==pytest.approx(1)
    with pytest.raises(ValueError,match='complete ordered'):builder.integration_records(ids[:-1],theta[:-1],p,count=4,prefix=2)


def test_ray_renderer_receives_corrected_surface_before_diagnostic_and_no_truth_inputs():
    X,packed,ids,theta=toy();wrapper=PointRotationForward(packed,X);seen=[]
    def sensor(q,faces,queries,**kwargs):
        seen.append(q.clone())
        return SimpleNamespace(visible=np.ones((2,72),dtype=bool),residual_m=np.zeros((2,72)))
    with patch.object(builder,'predict_ranges',side_effect=sensor):
        _,diagnostic=builder.corrected_range_batch(wrapper,None,ids,theta,X,[[0,1,2,3]],[],[],device='cpu')
    assert packed.calls==1 and len(seen)==1 and len(diagnostic['minimum_J'])==2
    raw=packed.predict(None,ids,theta)['positions_m'];expected=retract_positions(X,raw,ids)['positions_m']
    torch.testing.assert_close(seen[0],expected,atol=0,rtol=0)
    assert not torch.equal(seen[0],raw)
