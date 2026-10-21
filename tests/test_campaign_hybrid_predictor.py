"""Static TRAIN-only hybrid contracts; no observed/deformation labels loaded."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.campaign_nonlinear_dynamics_case import load_case_inputs,hardware_descriptor
from experiments.campaign_hybrid_predictor import HybridGaussianPredictor,differentiable_surface_weights
from experiments.campaign_boundary_gaussians import normalized_surface_weights
from experiments.campaign_material_gaussians import MaterialGaussians


@pytest.fixture(scope='module')
def static_case():
    # This route regenerates static mesh/spec from the public recipe and verifies
    # its hashes. It never opens motion, images or withheld response arrays.
    return load_case_inputs('t0001',None,Path('configs/gaussian_independent_diagnostic_inputs_v01.json'))[:3]


def inputs(static_case,*,dtype=torch.float64,device='cpu',frames=6):
    case,spec,arrays=static_case;rest=torch.as_tensor(arrays['rest_points_m'],dtype=dtype,device=device)
    hardware=hardware_descriptor(arrays['rest_points_m'].astype(float),spec)
    tensor=lambda x:torch.as_tensor(x,dtype=dtype,device=device)
    support=tensor([[0.,0.,0.],[0.,0.,0.]]);active=torch.zeros(2,dtype=torch.bool,device=device)
    original=spec['support_centers_m'];support[:len(original)]=tensor(original);active[:len(original)]=True
    fixture=torch.zeros(len(rest),dtype=torch.bool,device=device);fixture[spec['fixture_nodes']]=True
    material=case['material']
    data={'rest':rest[None],'handle_patches':torch.tensor(spec['load_patches'],device=device)[None],
        'handle_centers':tensor([h['reference_origin_m'] for h in hardware])[None],
        'support_centers':support[None],'support_active':active[None],'fixture_mask':fixture[None],
        'material':tensor([[material['youngs_modulus_Pa'],material['density_kg_m3'],material['poissons_ratio'],material['linear_damping_s_inverse']]]),
        'handle_masses':tensor([[h['mass_kg'] for h in hardware]]),
        'handle_inertias':tensor([[h['full_COM_inertia_kg_m2'] for h in hardware]]),
        'handle_com_offsets':tensor([[h['COM_offset_m'] for h in hardware]]),
        'handle_initial_rotations':tensor([[h['initial_rotation_matrix'] for h in hardware]]),
        'controls':torch.zeros(1,frames,12,dtype=dtype,device=device),
        'support_offsets':torch.zeros(1,frames,3,dtype=dtype,device=device),'times':torch.arange(frames,dtype=dtype,device=device)/20}
    tets=torch.tensor(arrays['tetrahedra'],dtype=torch.long,device=device);faces=torch.tensor(arrays['surface_faces'],dtype=torch.long,device=device)
    return data,tets,faces


def model(data,tets,faces):
    torch.manual_seed(2401)
    return HybridGaussianPredictor(data['rest'][0],tets,faces,width=8,layers=1).to(device=data['rest'].device,dtype=data['rest'].dtype)


def test_torch_boundary_quadrature_matches_independent_numpy_and_has_hypothesis_gradients(static_case):
    data,_,faces=inputs(static_case);rest=data['rest'][0]
    centers=torch.cat([data['support_centers'][0],data['handle_centers'][0]],0).detach().requires_grad_()
    covariance=(torch.eye(3,dtype=rest.dtype)[None].repeat(4,1,1)*.02**2).requires_grad_()
    active=torch.tensor([True,False,True,True])
    actual,areas=differentiable_surface_weights(rest,faces,centers,covariance,active)
    expected,expected_areas=normalized_surface_weights(rest.numpy(),faces.numpy(),centers.detach().numpy(),covariance.detach().numpy(),active.numpy())
    np.testing.assert_allclose(actual.detach().numpy(),expected,atol=2e-14,rtol=2e-12)
    np.testing.assert_allclose(areas.numpy(),expected_areas,atol=2e-16,rtol=2e-12)
    score=(actual*torch.linspace(-1,1,len(rest),dtype=rest.dtype)).sum();gc,gs=torch.autograd.grad(score,(centers,covariance))
    assert torch.isfinite(gc).all() and torch.isfinite(gs).all()
    assert gc.abs().sum()>0 and gs.abs().sum()>0
    assert torch.equal(actual[1],torch.zeros_like(actual[1]))


def test_zero_control_returns_reference_and_compatible_material_pushforward(static_case):
    data,tets,faces=inputs(static_case);network=model(data,tets,faces);result=network(data)
    torch.testing.assert_close(result['positions_m'],data['rest'][:,None].expand_as(result['positions_m']),atol=1e-14,rtol=0)
    torch.testing.assert_close(result['handle_centers_m'],data['handle_centers'][:,None].expand_as(result['handle_centers_m']),atol=1e-14,rtol=0)
    gaussians=MaterialGaussians.from_tetrahedra(data['rest'][0],tets,data['material'][0,1]);state=gaussians.transport(result['positions_m'])
    torch.testing.assert_close(state.deformation_gradients,torch.eye(3,dtype=data['rest'].dtype).expand_as(state.deformation_gradients),atol=1e-13,rtol=0)
    torch.testing.assert_close(state.covariances,gaussians.reference_covariances.expand_as(state.covariances),atol=1e-15,rtol=1e-12)


def test_rest_initial_condition_is_independent_of_force_at_first_query(static_case):
    data,tets,faces=inputs(static_case);network=model(data,tets,faces)
    data['controls'][:,0]=.07
    result=network(data)
    torch.testing.assert_close(result['positions_m'][:,0],data['rest'],atol=1e-14,rtol=0)


def test_causal_prefix_future_control_invariance_and_time_selection(static_case):
    data,tets,faces=inputs(static_case,frames=8);network=model(data,tets,faces)
    data['controls'][:,1:4,0]=.05;data['controls'][:,2:4,7]=.02
    first=network(data)['positions_m'];future=deepcopy(data);future['controls'][:,4:]=torch.randn_like(future['controls'][:,4:])
    changed=network(future)['positions_m']
    torch.testing.assert_close(changed[:,:4],first[:,:4],atol=1e-14,rtol=0)
    prefix={key:(value[:,:4] if key in ['controls','support_offsets'] else value[:4] if key=='times' else value) for key,value in data.items()}
    torch.testing.assert_close(network(prefix)['positions_m'],first[:,:4],atol=1e-14,rtol=0)
    selected=torch.tensor([0,2,5,7]);torch.testing.assert_close(network(data,time_indices=selected)['positions_m'],first[:,selected],atol=1e-14,rtol=0)


def test_exact_moving_fixture_and_finite_two_handle_patch_maps(static_case):
    data,tets,faces=inputs(static_case);network=model(data,tets,faces)
    data['controls'][:,1:,0]=.1;data['controls'][:,1:,8]=.05
    data['support_offsets'][:,1:,1]=torch.linspace(.0002,.001,5,dtype=data['rest'].dtype)
    result=network(data);positions=result['positions_m'];mask=data['fixture_mask'][0]
    expected=data['rest'][:,None]+data['support_offsets'][:,:,None]
    torch.testing.assert_close(positions[:,:,mask],expected[:,:,mask],atol=1e-14,rtol=0)
    for j in range(2):
        ids=data['handle_patches'][0,j];local=data['rest'][0,ids]-data['handle_centers'][0,j]
        R=result['handle_rotations'][0,:,j]
        mapped=local@R.transpose(-1,-2)+result['handle_centers_m'][0,:,j,None]
        torch.testing.assert_close(positions[0,:,ids],mapped,atol=1e-14,rtol=0)
        torch.testing.assert_close(R.transpose(-1,-2)@R,torch.eye(3,dtype=R.dtype).expand_as(R),atol=1e-13,rtol=0)
        torch.testing.assert_close(torch.linalg.det(R),torch.ones(len(R),dtype=R.dtype),atol=1e-13,rtol=0)


def test_batching_and_nonidentity_initial_hardware_pose_are_consistent(static_case):
    data,tets,faces=inputs(static_case)
    data={key:(value if key=='times' else value.repeat(2,*([1]*(value.ndim-1)))) for key,value in data.items()}
    # World-space rest patch offsets need the inverse initial pose before being
    # mapped by the returned absolute orientation. This covers nonidentity
    # hardware without changing any reference nodes or consuming motion labels.
    angle=torch.tensor(.7,dtype=data['rest'].dtype)
    c,s=torch.cos(angle),torch.sin(angle)
    R0=torch.stack([c,-s,c*0,s,c,c*0,c*0,c*0,c*0+1]).reshape(3,3)
    data['handle_initial_rotations'][1]=R0
    data['controls'][:,1:,0]=.06
    data['controls'][1,1:,8]=-.04
    network=model(data,tets,faces);batched=network(data)
    for b in range(2):
        single={key:(value if key=='times' else value[b:b+1]) for key,value in data.items()}
        torch.testing.assert_close(network(single)['positions_m'],batched['positions_m'][b:b+1],atol=2e-14,rtol=0)
        for j in range(2):
            ids=data['handle_patches'][b,j]
            offsets=data['rest'][b,ids]-data['handle_centers'][b,j]
            local=offsets@data['handle_initial_rotations'][b,j]
            mapped=local@batched['handle_rotations'][b,:,j].transpose(-1,-2)+batched['handle_centers_m'][b,:,j,None]
            torch.testing.assert_close(batched['positions_m'][b,:,ids],mapped,atol=2e-14,rtol=0)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:2',marks=pytest.mark.skipif(torch.cuda.device_count()<3,reason='CUDA2 unavailable'))])
def test_full_model_material_and_control_gradients_are_finite(static_case,device):
    data,tets,faces=inputs(static_case,dtype=torch.float32,device=device);network=model(data,tets,faces)
    data['controls'][:,1:,0]=.04;data['controls'].requires_grad_();data['material'].requires_grad_()
    result=network(data);gaussians=MaterialGaussians.from_tetrahedra(data['rest'][0],tets,data['material'][0,1])
    state=gaussians.transport(result['positions_m'])
    loss=(result['positions_m']-data['rest'][:,None]).square().sum()+state.covariances.square().sum()
    loss.backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in network.parameters())
    for key in ['material','controls']:
        assert data[key].grad is not None and bool(torch.isfinite(data[key].grad).all()) and bool(data[key].grad.abs().sum()>0)
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in network.parameters())
