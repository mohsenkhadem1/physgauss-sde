"""Training-only reconstruction-to-mechanics input adapter for PIDG Balls Reaction."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import numpy as np,torch
from PIL import Image
from scipy.ndimage import distance_transform_edt,gaussian_filter,maximum_filter
from scipy.spatial import cKDTree
from scene_PIDG import GaussianModel,DeformModel
from utils.general_utils import quaternion_to_matrix
base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction';doc=json.loads((base/'fit/transforms_train.json').read_text());frames=doc['frames'];run=ROOT/'runs/pidg-comparison-v02/refined-pidg';meta=json.loads((run/'run.json').read_text());cfg={};exec((run/'config.py').read_text(),cfg)
g=GaussianModel(3);g.load_ply(str(run/'model/point_cloud/iteration_5000/point_cloud.ply'),og_number_points=100000);d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(run/'model'),5000)
with torch.no_grad():
 def map_at(t):
  v=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),t,device='cuda'))['d_xyz'];return ((quaternion_to_matrix(v[:,:4])@g.get_xyz[...,None]).squeeze(-1)+v[:,4:]).cpu().numpy()
 first=map_at(frames[0]['time']);opacity=g.get_opacity.detach().cpu().numpy()[:,0]
im=np.asarray(Image.open(base/'released'/(frames[0]['file_path']+'.png')).resize((640,360)).convert('RGBA'))/255
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import label,binary_fill_holes
hsv=rgb_to_hsv(im[:,:,:3]);h=hsv[:,:,0];sat=hsv[:,:,1]
chosen=[]
for lo,hi,count in [(-.01,.15,2),(.20,.48,1),(.50,.635,2),(.635,.85,1)]:
 region=(h>lo)&(h<hi)&(sat>.25)&(im[:,:,3]>.5)
 if lo<0:region|=(h>.9)&(sat>.25)&(im[:,:,3]>.5)
 lab,n=label(region);areas=np.bincount(lab.ravel());areas[0]=0
 ids=np.argsort(-areas)[:count]
 for k in ids:
  assert areas[k]>20,(lo,hi,areas[k])
  yy,xx=np.nonzero(lab==k);cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2;radius=max(np.ptp(xx),np.ptp(yy))/2
  chosen.append(np.array([cx,cy,radius]))
assert len(chosen)==6
seeds=np.array(chosen);matrix=np.linalg.inv(frames[0]['transform_matrix']);cam=first@matrix[:3,:3].T+matrix[:3,3];focal=320/np.tan(doc['camera_angle_x']/2);depth=-cam[:,2];uv=np.stack([320+focal*cam[:,0]/depth,180-focal*cam[:,1]/depth],1)
score=np.linalg.norm((uv[:,None]-seeds[None,:,:2])/seeds[None,:,2:3],axis=-1);label=score.argmin(1);accepted=(score.min(1)<.90)&(opacity>.2)&(depth>0)
centers=[];radii=[];colors=[];groups=[];clouds=[]
for b,s in enumerate(seeds):
 ids=np.where(accepted&(label==b))[0];assert len(ids)>20,(b,len(ids));z=float(np.median(depth[ids]));camera=np.array([(s[0]-320)*z/focal,-(s[1]-180)*z/focal,-z]);pose=np.array(frames[0]['transform_matrix']);center=pose[:3,:3]@camera+pose[:3,3];radius=float(s[2]*z/focal);centers.append(center);radii.append(radius);groups.append(ids);clouds.append(first[ids]);colors.append(im[int(s[1]),int(s[0]),:3])
centers=np.array(centers);radii=np.array(radii);targets=[];affines=[]
with torch.no_grad():
 for frame in frames:
  moved=map_at(frame['time']);body=[]
  for b,ids in enumerate(groups):
   x=clouds[b];y=moved[ids];xc=np.median(x,axis=0);yc=np.median(y,axis=0);X=x-xc;Y=y-yc
   # Ridge regularization is a declared geometry initialization, not measured mechanics.
   ridge=len(x)*radii[b]**2*.02;A=np.linalg.solve(X.T@X+ridge*np.eye(3),X.T@Y+ridge*np.eye(3)).T
   u,s,vt=np.linalg.svd(A);u[:,-1]*=np.linalg.det(u@vt);A=u@np.diag(np.clip(s,.7,1.4))@vt
   c=A@(centers[b]-xc)+yc;body.append((A,c))
  affines.append(np.array([x[0] for x in body]));targets.append(np.array([x[1] for x in body]))
out=ROOT/'runs/pidg-mechanics-v01';np.savez_compressed(out/'training-geometry.npz',centers=centers,radii=radii,colors=colors,times=np.array([f['time'] for f in frames]),translations=targets,affines=affines,seeds=seeds)
protocol={'dataset':'PIDG balls_reaction','fit_frames':[f['file_path'] for f in frames],'geometry_source':'Training-only PIDG fitted checkpoint; first fitting frame segmentation; six spherical volume initialization hypotheses','motion_inputs':'Training reconstruction affine tracks; fitted implicit Gaussian attachment targets; interpolated only after training extraction','identifiability':'Monocular radius/depth and material/boundary ambiguity remain; not measured true geometry or material.','initialization_bounds':'Affine singular values clipped [0.7,1.4] before mechanics fitting; disclosed stabilizing geometry prior','teacher_run':str(run),'teacher_training_frames':101,'teacher_validation_frames':11,'test_images_read':False,'gaussians_per_group':[len(x) for x in groups],'centers':centers.tolist(),'radii':radii.tolist(),'input_sha256':hashlib.sha256((out/'training-geometry.npz').read_bytes()).hexdigest()}
(out/'input-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(protocol))
