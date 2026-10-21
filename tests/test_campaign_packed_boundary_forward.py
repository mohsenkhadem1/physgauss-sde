"""Tiny known-support mechanics only; no observations, atlas, or finite steps."""
import copy
import numpy as np
import pytest
import torch

from experiments.campaign_packed_boundary_forward import PackedBoundaryMaterialForward,modal_tensor_record
from experiments.campaign_modal_material_forward import BatchedModalMaterialForward
from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
from experiments.campaign_modal_static_lift import complete_static_lift
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
from experiments.campaign_nonlinear_dynamics import NonlinearDynamics
from experiments.campaign_exciting_commands import make_exciting_commands
from test_campaign_nonlinear_dynamics import centered_cube


def tiny_models(*,different_modes=False):
    torch.set_num_threads(2);points,tets=centered_cube()
    inertia=np.array([[1.1e-5,1e-6,3e-7],[1e-6,1.5e-5,-2e-7],[3e-7,-2e-7,2e-5]])
    models={}
    for node,modes in [(4,6),(8,8 if different_modes else 6)]:
        physical=NonlinearDynamics(points,tets,2000.,.3,700.,drag=.4,
            fixture_nodes=[node],boundary_kind='point_pin',load_patches=[[1,2,3],[5,6,7]],
            load_nodes=[1,5],handle_masses=[.02,.03],handle_inertias=[inertia,inertia*1.2],
            handle_com_offsets=[[.001,-.002,.001],[-.001,.001,.002]])
        models[node]=complete_static_lift(FixedModalLinearDynamics(LinearHypothesisDynamics(physical),modes))
    return models


@pytest.mark.parametrize('different_modes',[False,True])
def test_two_support_geometries_match_single_geometry_forward_and_preserve_order(monkeypatch,different_modes):
    monkeypatch.setattr(NonlinearDynamics,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No finite steps permitted')))
    models=tiny_models(different_modes=different_modes);packed=PackedBoundaryMaterialForward(models)
    clock=make_exciting_commands('exciting_world_a_v01');ids=torch.tensor([8,4,8,4,4],dtype=torch.int64)
    theta=torch.tensor([[4000.,650.,.31,.5],[6000.,800.,.38,.7],[1700.,450.,.28,.4],[13000.,780.,.42,.8],[3000.,600.,.35,.6]],dtype=torch.float64)
    result=packed.predict(clock,ids,theta)
    for i,id in enumerate(ids.tolist()):
        reference=BatchedModalMaterialForward(models[id]).predict(clock,theta[i:i+1])
        for key in ['positions_m','handle_centers_m','handle_rotation_coordinates_rad']:
            torch.testing.assert_close(result[key][i],reference[key][0],atol=3e-14,rtol=2e-10)
    torch.testing.assert_close(result['geometry_ids'],ids,atol=0,rtol=0)
    assert packed.catalogue()['geometry_ids']==[4,8]
    assert packed.catalogue()['categories_dropped']==[]
    assert packed.catalogue()['weights_supplied_or_estimated'] is False
    assert len(result['execution_groups'])==(2 if different_modes else 1)
    if different_modes:
        assert models[4].mode_count!=models[8].mode_count
        assert {g['mode_count'] for g in result['execution_groups']}=={m.mode_count for m in models.values()}
    assert len(packed._cache)==1
    # A query using only one category cannot remove the other from the catalogue.
    packed.predict(clock,ids[:1],theta[:1])
    assert packed.catalogue()['geometry_ids']==[4,8]


def test_packed_geometry_gather_scatter_preserves_all_four_material_gradients():
    models=tiny_models(different_modes=True);packed=PackedBoundaryMaterialForward(models)
    clock=make_exciting_commands('exciting_world_b_v01');ids=torch.tensor([8,4,8],dtype=torch.int64)
    theta=torch.tensor([[6000.,800.,.33,.6],[3500.,600.,.29,.4],[9500.,700.,.39,.7]],dtype=torch.float64,requires_grad=True)
    weights=torch.tensor([.7,1.3,2.1],dtype=torch.float64)
    def objective(values):
        displacement=packed.predict(clock,ids,values)['positions_m']-models[4].full.points
        return (displacement.square().sum(dim=(1,2,3))*weights).sum()
    gradient=torch.autograd.grad(objective(theta),theta)[0]
    references=[]
    for i,id in enumerate(ids.tolist()):
        one=theta.detach()[i:i+1].clone().requires_grad_(True)
        displacement=BatchedModalMaterialForward(models[id]).predict(clock,one)['positions_m']-models[id].full.points
        references.append(torch.autograd.grad(displacement.square().sum()*weights[i],one)[0][0])
    torch.testing.assert_close(gradient,torch.stack(references),atol=2e-15,rtol=2e-9)
    assert torch.isfinite(gradient).all() and torch.all(gradient.abs()>1e-14)
    # Relative1e-3 perturbations avoid subtracting nearly identical objectives
    # after360 solves; their central truncation is still quadratic in step.
    # Include both mode groups and a repeated category.
    for row,col,step in [(0,0,6.),(1,1,.6),(2,2,3.9e-4),(0,3,6e-4)]:
        delta=torch.zeros_like(theta);delta[row,col]=step
        finite=(objective(theta.detach()+delta)-objective(theta.detach()-delta))/(2*step)
        torch.testing.assert_close(gradient[row,col],finite,atol=3e-12,rtol=3e-5)


def test_unknown_category_invalid_material_and_hardware_mismatch_are_rejected():
    models=tiny_models();packed=PackedBoundaryMaterialForward(models);clock=make_exciting_commands('exciting_world_a_v01')
    theta=torch.tensor([[5000.,625.,.35,.5]],dtype=torch.float64)
    with pytest.raises(TypeError,match='known-actuator'):
        packed.predict(None,torch.tensor([4]),theta)
    with pytest.raises(ValueError,match='Unknown geometry'):
        packed.predict(clock,torch.tensor([7],dtype=torch.int64),theta)
    with pytest.raises(TypeError,match='int64'):
        packed.predict(clock,torch.tensor([4.]),theta)
    for j,value in [(0,0.),(1,-1.),(2,.5),(3,-.1)]:
        bad=theta.clone();bad[0,j]=value
        with pytest.raises(ValueError,match='material'):packed.predict(clock,torch.tensor([4]),bad)
    with pytest.raises(ValueError,match='exactly equal'):
        PackedBoundaryMaterialForward({4:models[8]})
    changed=copy.copy(models[8]);changed.full=copy.copy(changed.full);changed.full.model=copy.copy(changed.full.model)
    changed.full.model.com_offsets=changed.full.model.com_offsets.clone();changed.full.model.com_offsets[0,0]+=.001
    with pytest.raises(ValueError,match='public rest geometry and full hardware'):
        PackedBoundaryMaterialForward({4:models[4],8:changed})


def test_slim_tensor_cache_roundtrip_matches_live_geometry_without_full_systems(tmp_path):
    from experiments.campaign_hypothesis_inputs import known_hardware_from_public,physical_hardware_digest
    models=tiny_models(different_modes=True);p=models[4].full.model
    faces=np.concatenate([p.tets[:,ids] for ids in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]])
    unique,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
    inertia=np.array([[1.1e-5,1e-6,3e-7],[1e-6,1.5e-5,-2e-7],[3e-7,-2e-7,2e-5]])
    handles=[]
    for j,(origin,patch,mass,I,com) in enumerate(zip([1,5],[[1,2,3],[5,6,7]],[.02,.03],[inertia,inertia*1.2],[[.001,-.002,.001],[-.001,.001,.002]])):
        handles.append(dict(origin_world_m=p.points[origin].tolist(),ordered_patch_nodes=patch,mass_kg=mass,
            com_inertia_body_kg_m2=I.tolist(),com_offset_body_m=com,initial_rotation_body_to_world=np.eye(3).tolist(),origin_node_id=origin))
    public=dict(rest_points_m=p.points.tolist(),tetrahedra=p.tets.tolist(),surface_faces=unique[counts==1].tolist(),
        handles=handles,coordinate_origin_world_m=[0.,0.,0.])
    fingerprint=physical_hardware_digest(known_hardware_from_public(public))
    records={id:modal_tensor_record(m,id,fingerprint) for id,m in models.items()}
    path=tmp_path/'geometry.pt';torch.save(records,path)
    loaded=PackedBoundaryMaterialForward.from_tensor_records(torch.load(path,weights_only=True),public)
    assert not loaded.models and not loaded._single
    assert all('full' not in r and '_cache' not in r for r in records.values())
    theta=torch.tensor([[6000.,800.,.33,.6],[3500.,600.,.29,.4]],dtype=torch.float64)
    ids=torch.tensor([8,4]);clock=make_exciting_commands('exciting_world_a_v01')
    a=loaded.predict(clock,ids,theta);b=PackedBoundaryMaterialForward(models).predict(clock,ids,theta)
    for key in ['positions_m','handle_centers_m','handle_rotation_coordinates_rad']:
        torch.testing.assert_close(a[key],b[key],atol=3e-14,rtol=2e-10)
