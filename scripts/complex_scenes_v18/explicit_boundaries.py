"""Explicit derivatives of the common boundary energy, avoiding nested AD.

Forces remain differentiable in positions/rigid centers for training. Closest
triangle feature selection is piecewise constant, as in the energy implementation.
"""
import numpy as np
import torch
from scripts.complex_scenes_v18.torch_boundaries import TorchBoundaries
from scripts.complex_scenes_v18.pair_contact import inside_closed_surface
from scripts.complex_scenes_v18.geometry import RigidBowl,RigidCylinder
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere


def closest_barycentric(points,vertices,faces):
    a,b,c=vertices[faces].unbind(1);ab=b-a;ac=c-a;p=points[:,None];ap=p-a
    d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1);den=d00*d11-d01.square()
    d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1);v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
    cp=[a+v[...,None]*ab+w[...,None]*ac];bary=[torch.stack([1-v-w,v,w],-1)]
    distance=[torch.where((v>=0)&(w>=0)&(v+w<=1),(p-cp[0]).square().sum(-1),torch.full_like(v,float('inf')))]
    for j,(x,y)in enumerate([(a,b),(b,c),(c,a)]):
        edge=y-x;t=(((p-x)*edge).sum(-1)/edge.square().sum(-1)).clamp(0,1);zero=torch.zeros_like(t)
        cp.append(x+t[...,None]*edge);distance.append((p-cp[-1]).square().sum(-1))
        bary.append(torch.stack(([1-t,t,zero]if j==0 else[zero,1-t,t]if j==1 else[t,zero,1-t]),-1))
    choice=torch.stack(distance,-1).reshape(len(points),-1).argmin(-1);row=torch.arange(len(points),device=points.device)
    closest=torch.stack(cp,-2).reshape(len(points),-1,3)[row,choice]
    weights=torch.stack(bary,-2).reshape(len(points),-1,3)[row,choice]
    return closest,faces[choice//4],weights


class ExplicitTorchBoundaries(TorchBoundaries):
    def evaluate(self,positions,rigid_centers,t,create_graph=True):
        energy=sum(x.sum()*0 for x in positions);forces=[torch.zeros_like(x)for x in positions];rf={k:torch.zeros_like(v)for k,v in rigid_centers.items()}
        for i,(m,x)in enumerate(zip(self.scene.models,positions)):
            motion=self.scene.motions[i]
            if motion is None:poses=np.zeros((2,6));gates=np.ones(2)
            else:poses,_,gates,_=motion(t)
            if np.any(poses[:,3:]):raise ValueError('Translation-only actuator path')
            for j,A in enumerate(self.anchors[i]):
                delta=x-self.rest[i]-self.tensor(poses[j,:3]);reaction=float(gates[j])*(A@delta)
                energy=energy+.5*(delta*reaction).sum();forces[i]=forces[i]-reaction
            B,weights=self.patches[i];sample=B@x
            for c in m.colliders:
                center=rigid_centers.get(c.name)
                if center is None:center=self.tensor(c.point if isinstance(c,RigidPlane)else c.center)
                y=sample-center
                if isinstance(c,RigidPlane):
                    normal=self.tensor(c.normal);normal=normal/normal.norm();gap=(y*normal).sum(-1);gradient=normal.expand_as(y)
                elif isinstance(c,RigidSphere):
                    r=y.norm(dim=-1);gap=r-c.radius;gradient=y/r[:,None]
                elif isinstance(c,RigidCylinder):
                    axis=self.tensor(c.axis);axis=axis/axis.norm();radial=y-(y*axis).sum(-1,keepdim=True)*axis;r=radial.norm(dim=-1);gap=r-c.radius;gradient=radial/r[:,None]
                elif isinstance(c,RigidBowl):
                    R=c.curvature_radius;r2=y[:,:2].square().sum(-1);s=torch.sqrt(1+r2/R**2);h=y[:,2]-r2/(2*R);gap=h/s
                    xy=-y[:,:2]/(R*s[:,None])-h[:,None]*y[:,:2]/(R**2*s[:,None]**3);gradient=torch.cat([xy,(1/s)[:,None]],-1)
                else:raise ValueError('Unsupported geometry')
                penetration=gap.clamp_max(0);traction=(weights*c.penalty_N_m3*penetration)[:,None]*gradient
                energy=energy+.5*c.penalty_N_m3*(weights*penetration.square()).sum();forces[i]=forces[i]-B.T@traction
                if c.name in rf:rf[c.name]=rf[c.name]+traction.sum(0)
        if self.pair is not None:
            sampled=[b['B']@x for b,x in zip(self.pair,positions)];sf=[torch.zeros_like(x)for x in sampled]
            separated=bool((sampled[0].max(0).values<sampled[1].min(0).values).any()or(sampled[1].max(0).values<sampled[0].min(0).values).any())
            if not separated:
                for i,j in [(0,1),(1,0)]:
                    inside=inside_closed_surface(sampled[i],sampled[j],self.pair[j]['tri']);ids=torch.where(inside)[0]
                    if len(ids):
                        cp,tri,bary=closest_barycentric(sampled[i][ids],sampled[j],self.pair[j]['tri']);delta=sampled[i][ids]-cp;weight=self.pair[i]['weights'][ids]
                        energy=energy+.25*self.scene.pair.k*(weight*delta.square().sum(-1)).sum();reaction=.5*self.scene.pair.k*weight[:,None]*delta
                        sf[i]=sf[i].index_add(0,ids,-reaction);sf[j]=sf[j].index_add(0,tri.reshape(-1),(bary[:,:,None]*reaction[:,None,:]).reshape(-1,3))
                forces=[f+b['B'].T@s for f,b,s in zip(forces,self.pair,sf)]
        if not create_graph:return energy.detach(),[f.detach()for f in forces],{k:v.detach()for k,v in rf.items()}
        return energy,forces,rf
