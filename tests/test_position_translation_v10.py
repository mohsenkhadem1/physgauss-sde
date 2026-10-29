import numpy as np
import torch
from scripts.position_ood.smooth_fixture import SmoothDensePositionFEM


def test_whole_object_translation_distinguishes_load_frames():
    shift=np.array([0.,.04,0.]);responses={}
    for frame in ['material','world']:
        m=SmoothDensePositionFEM(1,'cpu',field_frame=frame)
        theta=m.tensor([*np.log([20000,5000,8000]),0,0,-9.81,.1,.3,.12,0,-.09,np.log(.08)])
        poses=m.tensor(np.zeros((2,6)));translated=poses.clone();translated[:,:3]+=m.tensor(shift)
        with torch.no_grad():u,_=m.solve(theta,poses,1);v,_=m.solve(theta,translated,1,initial=u+m.tensor(shift))
        responses[frame]=(v-u-m.tensor(shift)).numpy()
    np.testing.assert_allclose(responses['material'],0,atol=1e-10)
    assert np.sqrt(np.mean(responses['world']**2))>1e-4
