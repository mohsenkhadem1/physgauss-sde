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
