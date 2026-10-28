"""Software RGB-D renderer and color-fiducial tracking for a controlled bridge.

Pixel quantization and independent depth noise are real observation errors here.
Known marker identities/cameras replace raw-video correspondence estimation.
"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.sequential_reference import BASE,protocol,write
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_policy import field_indices
from scripts.phystwin.render_qualified_predictions import triangles
WIDTH,HEIGHT,FOCAL=640,480,650.

def camera(side=0):
 eye=np.array([.12 if side==0 else (-.32 if side<0 else .56),-.5,.2]);target=np.array([.12,0,0]);z=target-eye;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
 return eye,np.stack([x,y,z])
def project(p,cam):
 eye,R=cam;q=(p-eye)@R.T;return np.c_[FOCAL*q[:,0]/q[:,2]+WIDTH/2,FOCAL*q[:,1]/q[:,2]+HEIGHT/2],q[:,2]
def backproject(pixel,depth,cam):
 eye,R=cam;xy=(pixel-[WIDTH/2,HEIGHT/2])/FOCAL;return np.c_[xy*depth[:,None],depth]@R+eye

def render(positions,side,ids):
 cam=camera(side);uv,z=project(positions,cam);depth=np.full((HEIGHT,WIDTH),np.inf);rgb=np.full((HEIGHT,WIDTH,3),245,np.uint8)
 for tri in triangles():
  p=uv[tri];zz=z[tri];lo=np.maximum(np.floor(p.min(0)).astype(int),0);hi=np.minimum(np.ceil(p.max(0)).astype(int),[WIDTH-1,HEIGHT-1])
  if np.any(hi<lo):continue
  xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
  if abs(den)<1e-12:continue
  a=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den;b=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den;c=1-a-b
  valid=(a>=-1e-8)&(b>=-1e-8)&(c>=-1e-8);inv=a/zz[0]+b/zz[1]+c/zz[2];d=1/np.maximum(inv,1e-12);tile=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];take=valid&(d<tile);tile[take]=d[take]
  normal=np.cross(positions[tri[1]]-positions[tri[0]],positions[tri[2]]-positions[tri[0]]);normal/=max(np.linalg.norm(normal),1e-12);shade=.5+.4*abs(normal@cam[1][2]);rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1][take]=np.array([80,145,190])*shade
 for i in ids:
  px,py=np.rint(uv[i]).astype(int)
  if not (3<=px<WIDTH-3 and 3<=py<HEIGHT-3):raise RuntimeError('Marker out of view')
  local=depth[py-1:py+2,px-1:px+2];finite=local[np.isfinite(local)]
  if len(finite) and z[i]>finite.max()+.003:raise RuntimeError(f'Marker {i} occluded')
  # Distinct color IDs are intentionally supplied synthetic fiducials.
  color=np.array([20+2*i,30+(7*i)%180,40+(11*i)%180],np.uint8)
  yy,xx=np.mgrid[-2:3,-2:3];mask=xx*xx+yy*yy<=4;rgb[py-2:py+3,px-2:px+3][mask]=color;depth[py-2:py+3,px-2:px+3][mask]=z[i]
 return rgb,depth,cam

def track(rgb,depth,cam,ids,rng):
 pixels=[];depths=[]
 for i in ids:
  color=np.array([20+2*i,30+(7*i)%180,40+(11*i)%180],np.uint8);yy,xx=np.where(np.all(rgb==color,axis=2))
  if not len(xx):raise RuntimeError(f'Marker {i} missing')
  pixel=np.array([xx.mean(),yy.mean()]);ix,iy=np.rint(pixel).astype(int);pixels.append(pixel);depths.append(depth[iy,ix]+rng.normal(0,.0003))
 return backproject(np.array(pixels),np.array(depths),cam)

def main():
 spec=protocol();out=BASE/'vision';out.mkdir(parents=True,exist_ok=True);ix=field_indices();rest=points();diagnostics={}
 for split in ['development','calibration','test']:
  rng=np.random.default_rng({'development':910,'calibration':911,'test':912}[split]);measurements=[];truths=[];errors=[]
  for obj in range(len(spec[split])):
   fields=np.load(BASE/'fields'/f'{split}-{obj}-reference.npz')['field'];selected=fields.ravel()[ix];images=[]
   for action in range(7):
    ids=(ix[:9]//3)[::3] if action==0 else ((ix[9+3*(action-1):12+3*(action-1)]%243)//3)[::3]
    side=0 if action==0 else (-1 if rest[ids[0],0]==0 else 1);rgb,depth,cam=render(rest+fields[action],side,ids);images.append((rgb,depth,cam,ids))
    folder=out/f'{split}-{obj}';folder.mkdir(exist_ok=True);Image.fromarray(rgb).save(folder/f'{action}-rgb.png');np.savez_compressed(folder/f'{action}-depth.npz',depth_m=depth,camera_eye=cam[0],camera_rotation=cam[1],focal_pixels=FOCAL)
   for repeat in range(spec['noise_repeats']):
    obs=np.concatenate([(track(rgb,depth,cam,ids,rng)-rest[ids]).ravel() for rgb,depth,cam,ids in images]);measurements.append(obs);truths.append(selected[27:]);errors.append(obs-selected[:27])
  np.savez_compressed(out/f'{split}.npz',measurements=measurements,truth=truths)
  diagnostics[split]=dict(coordinate_observation_rmse_mm=float(1000*np.sqrt(np.mean(np.array(errors)**2))),frames=len(spec[split])*7,observations=len(measurements));print('VISION',split,diagnostics[split],flush=True)
 write(out/'summary.json',dict(scope='640x480 perspective rasterization, surface depth, explicit colored synthetic markers, pixel centroid quantization, 0.3mm depth noise, known cameras and rest geometry. One initial view and one action-facing view per acquired probe. Correspondence identities are supplied; this is not unconstrained RGB-D reconstruction.',splits=diagnostics))
if __name__=='__main__':main()
