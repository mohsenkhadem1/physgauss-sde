import numpy as np
from scripts.contact_benchmark_v12.contact_rollout import integrate_pose
from scripts.contact_benchmark_v12.scene import StaticScene

def block(z):
    return np.array([[x,y,z+q] for x in [-.02,.02] for y in [-.02,.02] for q in [0,.04]])

def test_freefall_translation_has_no_artificial_tether():
    p=block(1);p[:,0]+=2
    x,v,w,d=integrate_pose(p,np.zeros(3),np.zeros(3),.02,StaticScene(floor=-10),substeps=20)
    assert np.max(np.abs(x[:,:2]-p[:,:2]))<1e-12
    assert abs(v[2]+9.81*.02)<1e-12

def test_landing_nonpenetration_and_separation():
    p=block(.001);scene=StaticScene(floor=0)
    x,v,w,d=integrate_pose(p,np.array([0.,0.,-.5]),np.zeros(3),.05,scene,substeps=50)
    assert x[:,2].min()>=-1e-12
    y,_,_,_=integrate_pose(x,np.array([0.,0.,1.]),np.zeros(3),.01,scene,substeps=10)
    assert y[:,2].min()>x[:,2].min()

def test_static_contact_friction_slows_sliding():
    scene=StaticScene(floor=0);p=block(0);v=np.array([1.,0.,0.])
    _,vf,_,_=integrate_pose(p,v,np.zeros(3),.1,scene,friction=.5,substeps=100)
    _,v0,_,_=integrate_pose(p,v,np.zeros(3),.1,scene,friction=0,substeps=100)
    assert abs(vf[0])<abs(v0[0])
