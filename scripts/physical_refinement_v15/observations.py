"""Opaque RGB-D observation operator; generic v06 perspective-correct z buffer.
No material/support identifiers or mechanical labels are emitted as observations.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Camera:
    eye: tuple
    target: tuple
    width: int=160
    height: int=120
    focal: float=170.
    def pose(self):
        eye=np.asarray(self.eye,float);z=np.asarray(self.target,float)-eye;z/=np.linalg.norm(z)
        x=np.cross(z,[0.,0.,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
        return eye,np.stack([x,y,z])

def box_mesh(center,size,color):
    center=np.asarray(center,float);size=np.asarray(size,float)
    vertices=np.array([[i,j,k]for i in [-1,1]for j in [-1,1]for k in [-1,1]])*size/2+center
    quads=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    faces=np.array([[q[0],q[1],q[2]]for q in quads]+[[q[0],q[2],q[3]]for q in quads])
    return dict(vertices=vertices,triangles=faces,color=np.asarray(color,float))

def render_scene(meshes,camera):
    """Meshes are actual deformed material/rigid surfaces, not annotation proxies."""
    eye,R=camera.pose();H,W=camera.height,camera.width;depth=np.full((H,W),np.inf);rgb=np.full((H,W,3),.96)
    for mesh in meshes:
        xyz=np.asarray(mesh['vertices']);q=(xyz-eye)@R.T
        uv=q[:,:2]/q[:,2,None]*camera.focal+[W/2,H/2]
        colors=np.broadcast_to(mesh['color'],(len(xyz),3))
        for tri in np.asarray(mesh['triangles']):
            if np.any(q[tri,2]<=1e-6):raise ValueError('Near-plane clipping unsupported; choose an outside camera')
            p=uv[tri];z=q[tri,2];lo=np.maximum(np.floor(p.min(0)).astype(int),0);hi=np.minimum(np.ceil(p.max(0)).astype(int),[W-1,H-1])
            if np.any(hi<lo):continue
            xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
            den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
            if abs(den)<1e-12:continue
            a=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den
            b=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den;c=1-a-b
            valid=(a>=-1e-9)&(b>=-1e-9)&(c>=-1e-9);inv=a/z[0]+b/z[1]+c/z[2];distance=1/np.maximum(inv,1e-12)
            tile=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];take=valid&(distance<tile-1e-12)
            weights=np.stack([a/z[0],b/z[1],c/z[2]],-1)/np.maximum(inv[...,None],1e-12)
            color=np.einsum('hwk,kc->hwc',weights,colors[tri]);rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1][take]=color[take];tile[take]=distance[take]
    return dict(rgb=rgb,depth=depth,camera=dict(eye=list(camera.eye),target=list(camera.target),focal=camera.focal,width=W,height=H),depth_convention='camera optical-axis metres; infinity means no return')

def compare_rgbd(a,b):
    finite_a=np.isfinite(a['depth']);finite_b=np.isfinite(b['depth']);joint=finite_a&finite_b
    return dict(rgb_rmse=float(np.sqrt(np.mean((a['rgb']-b['rgb'])**2))),depth_rmse_m=float(np.sqrt(np.mean((a['depth'][joint]-b['depth'][joint])**2)))if joint.any()else None,silhouette_disagreement_pixels=int(np.count_nonzero(finite_a!=finite_b)),shared_depth_pixels=int(joint.sum()))
