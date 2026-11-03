"""Differentiable common boundaries for per-spring baseline calibration.

Same reference quadrature, compliant fixture energy and pair energy as the CPU
scene adapter. No bulk material law is evaluated here. Gravity stays in the native
integrator. All contact forces, including reactions, derive from one scalar energy.
"""
from dataclasses import replace
import numpy as np
import torch
from scripts.complex_scenes_v18.pair_contact import inside_closed_surface,closest_squared


class TorchBoundaries:
    def __init__(self,scene,device='cuda',dtype=torch.float32):
        self.scene=scene;self.device=device;self.dtype=dtype
        self.tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
        self.rest=[self.tensor(m.rest)for m in scene.models]
        self.anchors=[[self.tensor(A.toarray())for A in matrices]for matrices in scene.anchor_base]
        self.patches=[]
        for m in scene.models:
            matrices=[];weights=[]
            for p in m.contact_patches:
                nq,ne=p['weights'].shape;B=np.zeros((nq*ne,len(m.rest)))
                rows=np.broadcast_to(np.arange(nq*ne).reshape(nq,ne,1),(nq,ne,27))
                cols=np.broadcast_to(p['ids'][None],(nq,ne,27));values=np.broadcast_to(p['N'][:,None],(nq,ne,27))
                np.add.at(B,(rows.ravel(),cols.ravel()),values.ravel());matrices.append(B);weights.append(p['weights'].ravel())
            self.patches.append((self.tensor(np.concatenate(matrices)),self.tensor(np.concatenate(weights))))
        self.pair=None
        if scene.pair is not None:
            self.pair=[dict(B=self.tensor(b.B.toarray()),weights=self.tensor(b.weights),tri=torch.as_tensor(b.triangles,dtype=torch.long,device=device))for b in scene.pair.bindings]

    def evaluate(self,positions,rigid_centers,t,create_graph=True):
        if any(not x.requires_grad for x in positions)or any(not x.requires_grad for x in rigid_centers.values()):raise ValueError('Differentiable state tensors required')
        energy=sum(x.sum()*0 for x in positions)
        for i,(m,x)in enumerate(zip(self.scene.models,positions)):
            motion=self.scene.motions[i]
            if motion is None:poses=np.zeros((2,6));gates=np.ones(2)
            else:poses,_,gates,_=motion(t)
            if np.any(poses[:,3:]):raise ValueError('Translation-only public actuator path')
            for j,A in enumerate(self.anchors[i]):
                delta=x-self.rest[i]-self.tensor(poses[j,:3]);energy=energy+.5*float(gates[j])*(delta*(A@delta)).sum()
            B,weights=self.patches[i];sample=B@x
            for collider in m.colliders:
                if collider.name in rigid_centers:collider=replace(collider,center=rigid_centers[collider.name])
                gap=collider.signed_distance(sample);energy=energy+.5*collider.penalty_N_m3*(weights*gap.clamp_max(0).square()).sum()
        if self.pair is not None:
            sampled=[b['B']@x for b,x in zip(self.pair,positions)]
            separated=bool((sampled[0].max(0).values<sampled[1].min(0).values).any()or(sampled[1].max(0).values<sampled[0].min(0).values).any())
            if not separated:
                for i,j in [(0,1),(1,0)]:
                    inside=inside_closed_surface(sampled[i],sampled[j],self.pair[j]['tri'])
                    if bool(inside.any()):
                        d2=closest_squared(sampled[i][inside],sampled[j],self.pair[j]['tri'])
                        energy=energy+.25*self.scene.pair.k*(self.pair[i]['weights'][inside]*d2).sum()
        names=list(rigid_centers);variables=list(positions)+[rigid_centers[n]for n in names]
        gradients=torch.autograd.grad(energy,variables,create_graph=create_graph,allow_unused=True)
        forces=[-g if g is not None else torch.zeros_like(x)for g,x in zip(gradients,variables)]
        return energy,forces[:len(positions)],dict(zip(names,forces[len(positions):]))
