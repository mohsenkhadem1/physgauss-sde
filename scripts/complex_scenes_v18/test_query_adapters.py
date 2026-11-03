import numpy as np
import pytest
import torch
from scripts.complex_scenes_v18.world_queries import Query
from scripts.complex_scenes_v18.spring_scene import SpringScene
from scripts.complex_scenes_v18.phystwin_scene import PhysTwinScene


@pytest.mark.parametrize('Class',[SpringScene,PhysTwinScene])
def test_public_query_preserves_default_and_changes_fixture_response(Class):
    torch.set_num_threads(1)
    a=Class('clamp_release');b=Class('clamp_release',query=Query('clamp_release',0))
    c=Class('clamp_release',query=Query('clamp_release',0,fixture_shift=.1,actuation_scale=.9))
    for x,y in zip(a.x,b.x):torch.testing.assert_close(x,y)
    for obj in (a,b,c):obj.time=.04
    fa,_,_=a.external();fb,_,_=b.external();fc,_,_=c.external()
    np.testing.assert_allclose(fa[0],fb[0],atol=1e-10)
    assert np.linalg.norm(fc[0]-fa[0])>1e-3
