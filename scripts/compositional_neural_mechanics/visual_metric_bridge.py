"""Known-geometry synthetic RGB-D metrics from the SAME solved Q2 state.

No learned reconstruction, image residual warp, material-label appearance or
sensor-calibrated likelihood. Pixel sampling and surface tessellation are finite.
"""
from dataclasses import asdict
import numpy as np
from scripts.physical_refinement_v15.observations import Camera,render_scene

def surface_grid(cells=(24,8,8)):
 axes=[np.linspace(0,1,n+1)for n in cells];vertices=[];triangles=[]
 for axis in range(3):
  other=[j for j in range(3)if j!=axis]
  for side in (0,-1):
   begin=len(vertices);n=len(axes[other[1]])
   for a in axes[other[0]]:
    for b in axes[other[1]]:
     q=np.zeros(3);q[axis]=axes[axis][side];q[other]=[a,b];vertices.append(q)
   for i in range(len(axes[other[0]])-1):
    for j in range(n-1):
     a=begin+i*n+j;b=a+n;triangles.extend(((a,b,b+1),(a,b+1,a+1)))
 return np.asarray(vertices),np.asarray(triangles,dtype=int)

def state_surface(model,displacement,*,cells=(24,8,8)):
 q,tri=surface_grid(cells);u=np.asarray(displacement,float)
 if u.shape!=model.rest.shape or not np.isfinite(u).all():raise ValueError('Finite solved nodal displacement required')
 positions=model.geometry.map(q)+model.sample_material(u,q)
 # Appearance is tied only to material coordinates, identical for all hypotheses.
 colors=.2+.6*q
 return dict(vertices=positions,triangles=tri,color=colors)

def calibrated_cameras(geometry,width=256,height=192,focal=250.):
 center=geometry.map(np.array([[.5,.5,.5]]))[0];radius=max(geometry.dimensions)*2.4
 offsets={'front':np.array([0.,-1.,.35]),'rear':np.array([0.,1.,.35]),'oblique':np.array([-.85,-.7,.55])}
 return {name:Camera(tuple(center+radius*d/np.linalg.norm(d)),tuple(center),width,height,focal)for name,d in offsets.items()}

def render_state(model,displacement,cameras=None,*,cells=(24,8,8)):
 mesh=state_surface(model,displacement,cells=cells);cameras=calibrated_cameras(model.geometry)if cameras is None else cameras
 result={}
 for name,cam in cameras.items():
  image=render_scene([mesh],cam);eye,R=cam.pose()
  image['calibration']=dict(K=[[cam.focal,0,cam.width/2],[0,cam.focal,cam.height/2],[0,0,1]],world_to_camera_R=R.tolist(),world_to_camera_t=(-R@eye).tolist(),pixel_centers='integer+.5',depth='optical-axis metres;inf background')
  result[name]=image
 return result

def visual_errors(prediction,reference):
 a=np.isfinite(prediction['depth']);b=np.isfinite(reference['depth']);joint=a&b;union=a|b
 if prediction['rgb'].shape!=reference['rgb'].shape or prediction['depth'].shape!=reference['depth'].shape:raise ValueError('Identical image dimensions required')
 if prediction.get('camera')!=reference.get('camera'):raise ValueError('Identical calibrated camera required')
 difference=prediction['rgb']-reference['rgb']
 return dict(rgb_global_rmse=float(np.sqrt(np.mean(difference**2))),rgb_foreground_union_rmse=float(np.sqrt(np.mean(difference[union]**2)))if union.any()else None,depth_joint_rmse_m=float(np.sqrt(np.mean((prediction['depth'][joint]-reference['depth'][joint])**2)))if joint.any()else None,silhouette_iou=float(joint.sum()/union.sum())if union.any()else 1.,foreground_union_pixels=int(union.sum()),joint_depth_pixels=int(joint.sum()),missing_predicted_pixels=int((b&~a).sum()),extra_predicted_pixels=int((a&~b).sum()),scope='Finite raster/surface approximation; foreground-union RGB andjoint-depth withmissingcoverageexplicit.')

def physical_errors(predicted_displacement,reference_displacement,predicted_wrench,reference_wrench):
 d=np.asarray(predicted_displacement)-np.asarray(reference_displacement);w=np.asarray(predicted_wrench)-np.asarray(reference_wrench)
 if w.shape!=(2,6)or d.ndim!=2 or d.shape[1]!=3 or not np.isfinite(d).all()or not np.isfinite(w).all():raise ValueError('Finite matched physical arrays required')
 return dict(position_coordinate_rmse_m=float(np.sqrt(np.mean(d*d))),reaction_force_coordinate_rmse_N=float(np.sqrt(np.mean(w[:,:3]**2))),reaction_torque_coordinate_rmse_Nm=float(np.sqrt(np.mean(w[:,3:]**2))))
