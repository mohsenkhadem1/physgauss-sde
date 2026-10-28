"""Camera/marker functions copied unchanged from the sealed v06 renderer for Python 3.10 PhysTwin compatibility. No inference imports."""

import numpy as np

WIDTH,HEIGHT,FOCAL=640,480,650.

def triangles():
    grid=np.arange(81).reshape(9,3,3);faces=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        for side in [0,grid.shape[axis]-1]:
            for i in range(grid.shape[other[0]]-1):
                for j in range(grid.shape[other[1]]-1):
                    vertices=[]
                    for a,b in [(0,0),(1,0),(1,1),(0,1)]:
                        index=[0,0,0];index[axis]=side;index[other[0]]=i+a;index[other[1]]=j+b;vertices.append(grid[tuple(index)])
                    faces.extend([[vertices[k] for k in [0,1,2]],[vertices[k] for k in [0,2,3]]])
    return np.array(faces)

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
