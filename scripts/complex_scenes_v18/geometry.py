"""Explicit world collision geometry; energies use the differentiable gap.

The bowl's normalized implicit gap is not an exact Euclidean signed distance.
It defines a conservative compliant boundary and is reported as such.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scripts.compositional_neural_mechanics.contact_mechanics import vector, RigidPlane, RigidSphere
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel


@dataclass(frozen=True)
class RigidBowl:
    center: object=(.06, 0., -.08)
    curvature_radius: float=.10
    penalty_N_m3: float=4e7
    name: str='bowl'

    def origin(self, x): return vector(self.center, x)

    def signed_distance(self, x):
        if not np.isfinite(self.curvature_radius) or self.curvature_radius<=0:
            raise ValueError('Positive bowl curvature radius required')
        y=x-self.origin(x); r2=y[..., :2].square().sum(-1); R=self.curvature_radius
        return (y[..., 2]-r2/(2*R))/torch.sqrt(1+r2/R**2)


@dataclass(frozen=True)
class RigidCylinder:
    center: object=(.06, 0., -.05)
    axis: object=(0., 1., 0.)
    radius: float=.025
    penalty_N_m3: float=4e7
    name: str='cylinder'

    def origin(self,x): return vector(self.center,x)

    def signed_distance(self,x):
        a=vector(self.axis,x); length=torch.linalg.vector_norm(a)
        if float(length)<=1e-12 or not np.isfinite(self.radius) or self.radius<=0:
            raise ValueError('Valid cylinder axis/radius required')
        a=a/length;y=x-self.origin(x);rad=y-(y*a).sum(-1,keepdim=True)*a
        return torch.linalg.vector_norm(rad,dim=-1)-self.radius


class SceneBody(DynamicContactModel):
    def __init__(self,*args,colliders=(),bulk_quadrature='gauss3',**kwargs):
        if any(not isinstance(c,(RigidPlane,RigidSphere,RigidBowl,RigidCylinder)) for c in colliders):
            raise ValueError('Explicit supported collider required')
        if len({c.name for c in colliders})!=len(colliders):raise ValueError('Collider identities must be unique')
        super().__init__(*args,colliders=(),**kwargs)
        if bulk_quadrature!='gauss3':
            from scripts.complex_scenes_v18.quadrature import install_lobatto
            if bulk_quadrature not in ('lobatto4','lobatto5'):raise ValueError('Explicit supported volume quadrature required')
            install_lobatto(self,int(bulk_quadrature[-1]))
        self.colliders=tuple(colliders)
