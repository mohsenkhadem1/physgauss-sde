"""Conservative two-deformable-body contact on explicit sampled closed surfaces.

Symmetric vertex-to-triangle penalty with reference-area weights. Closest-point
and inside classifications are piecewise smooth, not differentiable at feature
switches. This is sampled finite-penetration contact, not CCD/nonpenetration.
All forces derive from one energy, including forces on the target triangles.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy import sparse
from scripts.phystwin.tensor_fixture import basis


def material_surface(k):
    if k<3:raise ValueError('At least three samples per face edge')
    vertices=[];faces=[];lookup={}
    for axis in range(3):
        others=[j for j in range(3)if j!=axis]
        for side in (0,1):
            ids=np.empty((k,k),int)
            for i in range(k):
                for j in range(k):
                    q=np.zeros(3);q[axis]=side;q[others]=[i/(k-1),j/(k-1)]
                    key=tuple(q)
                    if key not in lookup:lookup[key]=len(vertices);vertices.append(q)
                    ids[i,j]=lookup[key]
            for i in range(k-1):
                for j in range(k-1):
                    for tri in [[ids[i,j],ids[i+1,j],ids[i+1,j+1]], [ids[i,j],ids[i+1,j+1],ids[i,j+1]]]:
                        a,b,c=np.asarray(vertices)[tri]
                        if np.cross(b-a,c-a)[axis]*(2*side-1)<0:tri=tri[::-1]
                        faces.append(tri)
    return np.asarray(vertices),np.asarray(faces)


def interpolation_matrix(model,q):
    old=q*np.array([.24,.08,.08])+np.array([0,-.04,-.04])
    cell=np.stack([np.clip(np.searchsorted(a,old[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(model.axes)],1)
    lo=np.stack([a[cell[:,i]]for i,a in enumerate(model.axes)],1)
    width=np.stack([np.diff(a)[cell[:,i]]for i,a in enumerate(model.axes)],1)
    N,_=basis(2*(old-lo)/width-1);ids=model.elements[np.ravel_multi_index(cell.T,tuple(model.cells))]
    return sparse.coo_matrix((N.ravel(),(np.repeat(np.arange(len(q)),27),ids.ravel())),shape=(len(q),len(model.rest))).tocsr()


def closest_squared(points,vertices,faces):
    """Exact closest point on all nondegenerate triangles, including edges."""
    a,b,c=vertices[faces].unbind(1);ab=b-a;ac=c-a;p=points[:,None,:];ap=p-a
    d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1)
    den=d00*d11-d01*d01
    if bool((den<1e-24).any()):raise FloatingPointError('Degenerate contact surface triangle')
    d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1)
    v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
    proj=a+v[...,None]*ab+w[...,None]*ac
    plane=(p-proj).square().sum(-1)
    plane=torch.where((v>=0)&(w>=0)&(v+w<=1),plane,torch.full_like(plane,float('inf')))
    candidates=[plane]
    for x,y in [(a,b),(b,c),(c,a)]:
        edge=y-x;t=((p-x)*edge).sum(-1)/edge.square().sum(-1)
        cp=x+torch.clamp(t,0.,1.)[...,None]*edge
        candidates.append((p-cp).square().sum(-1))
    return torch.stack(candidates,-1).min(-1).values.min(-1).values


def inside_closed_surface(points,vertices,faces):
    """Solid-angle winding; supports nonconvex consistently oriented meshes."""
    with torch.no_grad():
        a,b,c=(vertices[faces][None,:,:,:]-points[:,None,None,:]).unbind(2)
        la=torch.linalg.vector_norm(a,dim=-1);lb=torch.linalg.vector_norm(b,dim=-1);lc=torch.linalg.vector_norm(c,dim=-1)
        num=(a*torch.linalg.cross(b,c)).sum(-1)
        den=la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
        winding=(2*torch.atan2(num,den)).sum(-1)/(4*np.pi)
        return winding.abs()>.5


@dataclass
class SurfaceBinding:
    B: object
    triangles: np.ndarray
    weights: np.ndarray
    q: np.ndarray

    @classmethod
    def build(cls,model,samples=5):
        q,tri=material_surface(samples);B=interpolation_matrix(model,q);X=B@model.rest
        area=.5*np.linalg.norm(np.cross(X[tri[:,1]]-X[tri[:,0]],X[tri[:,2]]-X[tri[:,0]]),axis=1)
        if np.any(area<=0):raise ValueError('Degenerate reference surface')
        weights=np.zeros(len(X));np.add.at(weights,tri.ravel(),np.repeat(area/3,3))
        return cls(B,tri,weights,q)


class DeformablePairContact:
    def __init__(self,models,samples=5,penalty_N_m3=4e7,device='cpu'):
        if len(models)!=2:raise ValueError('This pair operator takes exactly two separate bodies')
        if not np.isfinite(penalty_N_m3)or penalty_N_m3<=0:raise ValueError('Positive penalty required')
        self.bindings=[SurfaceBinding.build(m,samples)for m in models];self.k=penalty_N_m3;self.device=device
        self.tri=[torch.tensor(b.triangles,dtype=torch.long,device=device)for b in self.bindings]
        self.weights=[torch.tensor(b.weights,dtype=torch.float64,device=device)for b in self.bindings]

    def evaluate(self,positions):
        sampled=[b.B@x for b,x in zip(self.bindings,positions)]
        # Exact AABB rejection is safe only for separated sampled surfaces.
        if np.any(sampled[0].max(0)<sampled[1].min(0)) or np.any(sampled[1].max(0)<sampled[0].min(0)):
            return 0.,[np.zeros_like(x)for x in positions],dict(active_vertices=0,max_sampled_penetration_m=0.)
        x=[torch.tensor(v,dtype=torch.float64,device=self.device,requires_grad=True)for v in sampled]
        energies=[];depths=[];count=0
        for i,j in [(0,1),(1,0)]:
            inside=inside_closed_surface(x[i],x[j],self.tri[j]);count+=int(inside.sum())
            if bool(inside.any()):
                dsq=closest_squared(x[i][inside],x[j],self.tri[j])
                energies.append(.25*self.k*(self.weights[i][inside]*dsq).sum());depths.append(float(torch.sqrt(dsq.detach()).max()))
        if not energies:return 0.,[np.zeros_like(v)for v in positions],dict(active_vertices=0,max_sampled_penetration_m=0.)
        energy=sum(energies);grad=torch.autograd.grad(energy,x)
        force=[-b.B.T@g.detach().cpu().numpy()for b,g in zip(self.bindings,grad)]
        return float(energy.detach()),force,dict(active_vertices=count,max_sampled_penetration_m=max(depths),net_contact_force_N=float(np.linalg.norm(sum((f.sum(0)for f in force),np.zeros(3)))),net_contact_torque_Nm=float(np.linalg.norm(sum((np.cross(v,f).sum(0)for v,f in zip(positions,force)),np.zeros(3)))))
