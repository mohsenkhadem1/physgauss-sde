import numpy as np
import pytest
import torch
from scripts.complex_scenes_v18.world_queries import World,Material,Query,build_reference,build_prediction
from scripts.complex_scenes_v18.scenarios import FAMILIES


def test_material_stays_identical_across_interactions_and_overrides_private_recipe():
    for family in FAMILIES:
        mats=(Material(np.log(9200),.29,.3,.42),)*(2 if family=='deformable_pair'else 1)
        world=World('development-contract',mats);fingerprint=world.digest
        for variant in (0,1):
            scene,state,meta=build_reference(world,Query(family,variant))
            for m in scene.models:
                q=torch.tensor([[.1,.5,.5],[.9,.5,.5]],dtype=torch.float64)
                torch.testing.assert_close(m.material_latent_field(q),mats[0].provider()[1](q))
            assert meta['private_world_sha256']==fingerprint


def test_public_query_uses_explicit_material_and_scales_actions():
    with pytest.raises(ValueError):build_prediction(Query('container_drop',0),None)
    world=World('demo',(Material(np.log(12000)),))
    a,sa,_=build_reference(world,Query('rigid_impact',0))
    b,sb,_=build_reference(world,Query('rigid_impact',0,velocity_scale=1.4))
    np.testing.assert_allclose(sb.rigid['striker']['velocity'],1.4*sa.rigid['striker']['velocity'])
    _,drop,_=build_reference(world,Query('container_drop',0,initial_height_m=.025))
    np.testing.assert_allclose(drop.u[0][:,2],.025)


def test_rotated_drop_preserves_shape_and_fixture_shift_changes_support():
    world=World('demo',(Material(np.log(12000)),))
    scene,state,_=build_reference(world,Query('container_drop',0,initial_rotation_deg=30))
    rest=scene.models[0].rest;deformed=rest+state.u[0]
    np.testing.assert_allclose(np.linalg.norm(rest[:,None]-rest[None,:],axis=-1),np.linalg.norm(deformed[:,None]-deformed[None,:],axis=-1),atol=1e-12)
    a,_,_=build_reference(world,Query('clamp_release',0))
    b,_,_=build_reference(world,Query('clamp_release',0,fixture_shift=.1))
    assert np.linalg.norm((a.models[0].anchor_matrices[0]-b.models[0].anchor_matrices[0]).toarray())>1e-3


def test_mpm_initial_geometry_matches_public_affine_surface():
    from scripts.complex_scenes_v18.world_queries import drop_particle_cloud
    from scripts.complex_scenes_v18.particle_tracks import binding
    from scripts.complex_scenes_v18.pair_contact import SurfaceBinding
    query=Query('container_drop',1,initial_rotation_deg=23,initial_height_m=.02)
    world=World('demo',(Material(np.log(12000)),));scene,state,_=build_reference(world,query)
    surface=SurfaceBinding.build(scene.models[0],9);_,x=drop_particle_cloud(query);index,weight=binding(surface.q)
    np.testing.assert_allclose((x[index]*weight[...,None]).sum(1),surface.B@(scene.models[0].rest+state.u[0]),atol=1e-12)
