import json
import numpy as np,torch
from .model import SceneQueryModel
from . import data

def test_target_geometry_and_held_camera_do_not_enter_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'BASE',tmp_path);p=tmp_path/'prepared/example';p.mkdir(parents=True)
    C=np.repeat(np.eye(4)[None],3,0);K=np.repeat(np.eye(3)[None],3,0)
    (p/'geometry.json').write_text(json.dumps(dict(camera_to_world=C.tolist(),intrinsics=K.tolist(),WH=[100,100],frames=2,train_end=1)))
    np.save(p/'features.npy',np.zeros((2,2,256,12,12),np.float16))
    np.savez(p/'labels.npz',query_uv=np.full((4,2),.5,np.float32),tracks=np.zeros((2,4,3),np.float32),valid=np.ones((2,4),bool))
    s=data.Sequence('example');a,y=s.sample(1,np.arange(4));s.labels['tracks']+=100;s.C[2]+=500;b,z=s.sample(1,np.arange(4))
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    assert not np.array_equal(y['position'],z['position'])

def test_image_blind_control_is_independent_of_rgb():
    torch.set_num_threads(1);m=SceneQueryModel(True).eval()
    x=dict(features=torch.randn(1,4,256,12,12),token_geometry=torch.zeros(1,576,19),uv=torch.rand(1,4,2),rays=torch.ones(1,4,3),camera0=torch.eye(4)[None],time_s=torch.tensor([1.]))
    with torch.no_grad():a=m(**x)[0];x['features']*=100;b=m(**x)[0]
    torch.testing.assert_close(a,b)

def test_source_time_has_zero_displacement():
    m=SceneQueryModel().eval();x=dict(features=torch.zeros(1,4,256,12,12),token_geometry=torch.zeros(1,576,19),uv=torch.rand(1,3,2),rays=torch.ones(1,3,3),camera0=torch.eye(4)[None],time_s=torch.zeros(1))
    with torch.no_grad():m.delta.bias.fill_(1);p,s=m(**x)
    torch.testing.assert_close(p,s)
