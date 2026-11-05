from pathlib import Path
import numpy as np,json,torch
from scipy.ndimage import label as components
from scipy.spatial import cKDTree
from PIL import Image
from matplotlib.colors import rgb_to_hsv
root=Path.cwd();source=root/'runs/pidg-appearance-transfer-v01';out=root/'runs/pidg-appearance-transfer-v02';out.mkdir(exist_ok=True)
for name in ['transferred-appearance.pt','physical-cache.npz','binding-reference-audit.json']:
 if not (out/name).exists():(out/name).symlink_to((source/name).resolve())
g=np.load(root/'runs/pidg-mechanics-v03/training-geometry.npz');doc=json.loads((root/'runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_train.json').read_text());f=doc['frames'][0];im=np.asarray(Image.open(root/'runs/pidg-comparison-v01/data/balls_reaction/released'/(f['file_path']+'.png')).resize((640,360)).convert('RGBA'))/255;hsv=rgb_to_hsv(im[:,:,:3]);h=hsv[:,:,0];sat=hsv[:,:,1];coords=[];classes=[]
for lo,hi,ids in [(-.01,.15,[0,1]),(.20,.48,[2]),(.50,.635,[3,4]),(.635,.85,[5])]:
 mask=(h>lo)&(h<hi)&(sat>.35)&(im[:,:,3]>.5)
 if lo<0:mask|=(h>.9)&(sat>.35)&(im[:,:,3]>.5)
 lab,n=components(mask);regions=[]
 for i in range(1,n+1):
  yy,xx=np.nonzero(lab==i)
  if len(xx)>40:regions.append(np.c_[xx,yy])
 for b in ids:
  region=min(regions,key=lambda x:np.linalg.norm(x.mean(0)-g['seeds'][b,:2]));coords.extend(region);classes.extend([b]*len(region))
x=torch.load(source/'transferred-appearance.pt',map_location='cpu',weights_only=False)['xyz'].numpy();mat=np.linalg.inv(f['transform_matrix']);p=x@mat[:3,:3].T+mat[:3,3];focal=320/np.tan(doc['camera_angle_x']/2);uv=np.c_[320+focal*p[:,0]/(-p[:,2]),180-focal*p[:,1]/(-p[:,2])];_,idx=cKDTree(coords).query(uv);labels=np.array(classes)[idx];np.save(out/'body-labels.npy',labels)
(out/'binding-identity.json').write_text(json.dumps({'source':'First fitting image only: hue-connected object regions; nearest projected region for each Gaussian','counts':np.bincount(labels,minlength=6).tolist(),'held_images_read':False},indent=2));print(np.bincount(labels))
