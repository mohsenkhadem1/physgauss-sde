"""True six-node P2 surface interpolation, with explicit tessellation checks.

Curved patches are evaluated quadratically; ray tests use a piecewise planar
approximation at caller-declared levels. Successive agreement is a numerical
sensor check, not a rigorous global curved-surface error bound.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from experiments.campaign_ray_sensor import intersect_mesh_rays

EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def quadratic_boundary_faces(elements,faces):
    e=np.asarray(elements);f=np.asarray(faces)
    if e.ndim!=2 or e.shape[1]!=10 or f.ndim!=2 or f.shape[1] not in (3,6) or e.dtype.kind not in 'iu' or f.dtype.kind not in 'iu':raise ValueError('P2 tetrahedra and corner or six-node faces required')
    edge={}
    for tet in e:
        for k,(i,j) in enumerate(EDGES):
            key=tuple(sorted(map(int,tet[[i,j]])));node=int(tet[4+k])
            if key in edge and edge[key]!=node:raise ValueError('Nonconforming shared P2 edge')
            edge[key]=node
    result=[]
    for face in f:
        corners=face[:3];ids=[edge[tuple(sorted(map(int,corners[[i,j]])))] for i,j in ((0,1),(0,2),(1,2))];six=np.r_[corners,ids]
        if len(face)==6 and not np.array_equal(face,six):raise ValueError('Six-node face order must be vertices,e01,e02,e12')
        result.append(six)
    return np.asarray(result,dtype=np.int64).reshape(-1,6)


@dataclass(frozen=True)
class P2SurfaceBinding:
    node_ids:np.ndarray
    weights:np.ndarray
    triangles:np.ndarray
    subdivisions:int
    def evaluate(self,positions):
        if isinstance(positions,torch.Tensor):
            ids=torch.as_tensor(self.node_ids,device=positions.device);w=torch.as_tensor(self.weights,dtype=positions.dtype,device=positions.device)
            return (positions[...,ids,:]*w[...,None]).sum(-2)
        q=np.asarray(positions)
        return np.einsum('sa,...sai->...si',self.weights,q[...,self.node_ids,:])


def bind_surface(faces6,subdivisions):
    f=np.asarray(faces6)
    if f.ndim!=2 or f.shape[1]!=6 or f.dtype.kind not in 'iu' or type(subdivisions) is not int or subdivisions<1:raise ValueError('Six-node surface and positive integer subdivision required')
    n=subdivisions;ids=[];weights=[];triangles=[];lookup={}
    for face in f:
        grid={}
        for i in range(n+1):
            for j in range(n+1-i):
                b=np.array([i,j,n-i-j],dtype=np.int64);num=np.r_[b*(2*b-n),4*b[0]*b[1],4*b[0]*b[2],4*b[1]*b[2]]
                key=tuple(sorted((int(node),int(w)) for node,w in zip(face,num) if w))
                if key not in lookup:lookup[key]=len(ids);ids.append(face.copy());weights.append(num/n**2)
                grid[i,j]=lookup[key]
        for i in range(n):
            for j in range(n-i):
                triangles.append([grid[i,j],grid[i+1,j],grid[i,j+1]])
                if i+j<n-1:triangles.append([grid[i+1,j],grid[i+1,j+1],grid[i,j+1]])
    return P2SurfaceBinding(np.asarray(ids,dtype=np.int64).reshape(-1,6),np.asarray(weights).reshape(-1,6),np.asarray(triangles,dtype=np.int64).reshape(-1,3),n)


def checked_p2_depth(positions,faces6,origins,directions,*,levels=(2,4,8),depth_tolerance_m=2e-5,**chunks):
    """No images/truth lookup; returns finest depths plus explicit finite check.

    All adjacent levels must agree in visibility and depth. Never silently drop
    a ray whose visibility differs. The selected levels/budget must freeze
    before using physical outcomes; this routine does not tune them.
    """
    if len(levels)<2 or any(type(x) is not int or x<1 for x in levels) or any(b<=a for a,b in zip(levels,levels[1:])) or not np.isfinite(depth_tolerance_m) or depth_tolerance_m<=0:raise ValueError('Prospective increasing surface levels and physical depth tolerance required')
    reports=[];last=None
    for level in levels:
        binding=bind_surface(faces6,level);q=binding.evaluate(positions);value=intersect_mesh_rays(q,binding.triangles,origins,directions,**chunks)
        if last is not None:
            same=value.visible==last.visible;both=value.visible&last.visible
            delta=(value.depth_m-last.depth_m).abs()[both]
            maximum=0. if not delta.numel() else float(delta.max())
            reports.append(dict(coarse=previous,fine=level,visibility_disagreements=int((~same).sum()),maximum_visible_depth_difference_m=maximum,passed=bool(same.all()) and maximum<=depth_tolerance_m))
        previous=level;last=value
    return last,dict(passed=all(x['passed'] for x in reports),comparisons=reports,depth_tolerance_m=depth_tolerance_m,claim='Successive tessellation sensor check, not exact curved-ray intersection or rigorous global bound',physical_midnode_geometry_used=True)
