import numpy as np
from scripts.compositional_neural_mechanics.visual_metric_bridge import Camera,render_scene,visual_errors,calibrated_cameras
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec

def mesh(z,color):
 # Camera eye origin, target+x: camera xaxis=-worldy,yaxis=-worldz.
 return dict(vertices=np.array([[z,1,1],[z,-1,1],[z,-1,-1],[z,1,-1]]),triangles=np.array([[0,1,2],[0,2,3]]),color=np.array(color))

def test_camera_occlusion_optical_depth():
 cam=Camera((0.,0.,0.),(1.,0.,0.),32,24,20.)
 front=mesh(2,[1,0,0]);back=mesh(3,[0,1,0]);a=render_scene([front,back],cam);b=render_scene([back,front],cam)
 assert np.array_equal(a['rgb'],b['rgb']);assert np.array_equal(a['depth'],b['depth'])
 assert a['depth'][12,16]==2;assert np.array_equal(a['rgb'][12,16],[1,0,0])
 e=visual_errors(a,b);assert e['silhouette_iou']==1 and e['rgb_foreground_union_rmse']==0 and e['depth_joint_rmse_m']==0
 eye,R=cam.pose();assert np.allclose((np.array([2,0,0])-eye)@R.T,[0,0,2])

def test_appearance_has_no_stiffness_channel_and_metric_missing_pixels():
 cam=Camera((0.,0.,0.),(1.,0.,0.),32,24,20.)
 a=render_scene([mesh(2,[.2,.4,.6])],cam);b=render_scene([mesh(4,[.2,.4,.6])],cam)
 e=visual_errors(b,a);assert e['silhouette_iou']<1 and e['missing_predicted_pixels']>0 and e['depth_joint_rmse_m']==2
 for c in calibrated_cameras(GeometrySpec()).values():
  eye,R=c.pose();assert np.allclose(R@R.T,np.eye(3));assert np.linalg.det(R)>0

def test_exact_common_scale_same_images_different_reactions():
 import copy
 from scripts.compositional_neural_mechanics.scenes import sample_scene,make_model
 from scripts.compositional_neural_mechanics.visual_metric_bridge import render_state
 scene=sample_scene((0,0,0,0),np.random.default_rng(926991),'unit-scale-test',gravity=(0.,0.,0.));scene['interface_latents'][0][2]=0;scene['interface_latents'][1][2]=0
 model=make_model(scene,n=1);pose=np.zeros((2,6));pose[1,0]=.004;model.set_controls(pose);state=model.solve(force_tolerance_N=1e-10);u=state['displacement_local_m']
 scaled=copy.deepcopy(scene);scaled['bulk_latent'][0]+=np.log(2)
 for z in scaled['interface_latents']:z[0]+=np.log(2)
 other=make_model(scaled,n=1);other.set_controls(pose)
 assert np.linalg.norm(other.evaluate(u,False)['residual'])<1e-8
 w=model.reaction_wrenches(state);v=other.reaction_wrenches(dict(state,displacement_local_m=u))
 assert np.linalg.norm(w['forces_N'])>1e-4
 assert np.allclose(v['forces_N'],2*w['forces_N'],rtol=1e-10,atol=1e-12)
 assert np.allclose(v['torques_Nm'],2*w['torques_Nm'],rtol=1e-10,atol=1e-12)
 cams=calibrated_cameras(model.geometry,64,48,60.)
 a=render_state(model,u,cams,cells=(6,2,2));b=render_state(other,u,cams,cells=(6,2,2))
 for name in a:
  assert np.array_equal(a[name]['rgb'],b[name]['rgb'])
  assert np.array_equal(a[name]['depth'],b[name]['depth'])
