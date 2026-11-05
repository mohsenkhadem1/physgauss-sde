"""Training RGB silhouette controls; no test images, no invented novel-view targets."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import rgb_to_hsv
source=ROOT/'runs/pidg-mechanics-v01';out=ROOT/'runs/pidg-mechanics-v03';out.mkdir(exist_ok=True);g=np.load(source/'training-geometry.npz');base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction';doc=json.loads((base/'fit/transforms_train.json').read_text());focal=320/np.tan(doc['camera_angle_x']/2);translations=[];affines=[];tracks=[];prev=g['seeds'][:,:2].copy();velocity=prev*0
for f in doc['frames']:
 image=np.asarray(Image.open(base/'released'/(f['file_path']+'.png')).resize((640,360)).convert('RGBA'))/255;hsv=rgb_to_hsv(image[:,:,:3]);h=hsv[:,:,0];sat=hsv[:,:,1];current=np.zeros((6,2));radii=np.zeros((6,2))
 for lo,hi,ids in [(-.01,.15,[0,1]),(.20,.48,[2]),(.50,.635,[3,4]),(.635,.85,[5])]:
  region=(h>lo)&(h<hi)&(sat>.35)&(image[:,:,3]>.5)
  if lo<0:region|=(h>.9)&(sat>.35)&(image[:,:,3]>.5)
  lab,n=label(region);components=[]
  for k in range(1,n+1):
   yy,xx=np.nonzero(lab==k)
   if len(xx)<40:continue
   low=np.array([xx.min(),yy.min()]);high=np.array([xx.max(),yy.max()]);center=(low+high)/2;rad=(high-low)/2
   if min(rad)<3:continue
   components.append((center,rad,len(xx)))
  assert len(components)>=len(ids),(f['file_path'],ids)
  cost=np.array([[np.linalg.norm(c[0]-(prev[b]+velocity[b]))/max(g['seeds'][b,2],5)+.2*abs(np.log(c[2]/(np.pi*g['seeds'][b,2]**2*.65))) for c in components] for b in ids])
  ii,jj=linear_sum_assignment(cost)
  for k,j in zip(ii,jj):
   b=ids[k];current[b],radii[b],_=components[j]
 pose=np.array(f['transform_matrix']);body=[];stretch=[]
 for b in range(6):
  rx,ry=radii[b];r=np.sqrt(rx*ry);depth=g['radii'][b]*focal/r
  camera=np.array([(current[b,0]-320)*depth/focal,-(current[b,1]-180)*depth/focal,-depth]);body.append(pose[:3,:3]@camera+pose[:3,3])
  A=pose[:3,:3]@np.diag([rx/r,ry/r,1.])@pose[:3,:3].T;stretch.append(A)
 translations.append(body);affines.append(stretch);tracks.append(np.c_[current,radii]);velocity=.5*velocity+.5*(current-prev);prev=current
np.savez_compressed(out/'training-geometry.npz',centers=np.array(translations)[0],radii=g['radii'],colors=np.array([[.65,.2,.17],[.65,.2,.17],[.15,.75,.08],[.12,.4,.85],[.12,.4,.85],[.3,.07,.8]]),times=g['times'],translations=translations,affines=affines,seeds=g['seeds'],tracks=tracks)
(out/'input-protocol.json').write_text(json.dumps({'fit_images_only':True,'fit_frame_count':len(translations),'source':'Hue-separated object tracks from training RGB; six-object spherical volume prior, initial radius gauge from v01 reconstruction','boundary_inputs':'Six Gaussian attachment footprints; targets inferred from silhouette centers and ellipse axes, not supplied real actuators','gauge':'Radius/depth fixed by initial reconstruction; apparent radius implies depth, ellipse anisotropy is volume-preserving initial strain hypothesis','inference_limit':'Neither true material nor actual contact/force is identified. Camera-unseen pixel accuracy has no released reference.'},indent=2))
print('Prepared',len(translations),'training observations')
