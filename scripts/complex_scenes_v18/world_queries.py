"""Separate public interaction recipes from private, immutable world materials.

Development API, not a frozen held protocol. A prediction builder must receive a
material provider explicitly: the private generator material is never a default.
All units are SI. Recipe families denote visible scene geometry, not latent labels.
"""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
import torch
from scripts.complex_scenes_v18.scenarios import build_case,FAMILIES
from scripts.complex_scenes_v18.scene import CoupledScene
from scripts.compositional_neural_mechanics.mechanics import AnalyticNeoHookean


@dataclass(frozen=True)
class Material:
    log_young:float
    poisson:float=.35
    contrast:float=0.
    transition:float=.5

    def provider(self):
        def field(q):return (self.log_young+self.contrast*torch.tanh(8*(q[...,0]-self.transition)))[...,None]
        return AnalyticNeoHookean(self.poisson),field


@dataclass(frozen=True)
class World:
    world_id:str
    materials:tuple[Material,...]

    def material_provider(self,index):return self.materials[index].provider()

    @property
    def digest(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Query:
    family:str
    variant:int
    actuation_scale:float=1.
    velocity_scale:float=1.
    initial_height_m:float=0.
    initial_rotation_deg:float=0.
    fixture_shift:float=0.

    def __post_init__(self):
        if self.family not in FAMILIES or self.variant not in (0,1):raise ValueError('Unknown public scene recipe')
        if not np.isfinite([self.actuation_scale,self.velocity_scale,self.initial_height_m,self.initial_rotation_deg,self.fixture_shift]).all():raise ValueError('Nonfinite public action')
        if self.actuation_scale<=0 or self.velocity_scale<=0:raise ValueError('Positive action scales required')
        if self.initial_height_m and self.family!='container_drop':raise ValueError('Height offset only supported for free drops')
        if self.initial_rotation_deg and self.family!='container_drop':raise ValueError('Initial rotation only supported for free drops')
        if abs(self.fixture_shift)>.15:raise ValueError('Fixture shift out of range')
        if self.fixture_shift and self.family not in ('clamp_release','curved_push','rigid_impact','heterogeneous'):raise ValueError('Scene has no fixtures')


def build_prediction(query,material_provider,*,n=1,surface_samples=5,bulk_quadrature='gauss3',contact_device='cpu'):
    """Build from visible query plus inferred material; no World argument allowed."""
    if material_provider is None:raise ValueError('An inferred material provider is mandatory')
    scene,state,meta=build_case(query.family,n=n,variant=query.variant,surface_samples=surface_samples,bulk_quadrature=bulk_quadrature,contact_device=contact_device,material_provider=material_provider,fixture_shift=query.fixture_shift)
    motions=[]
    for motion in scene.motions:
        if motion is None:motions.append(None)
        else:
            def scaled(t,motion=motion):
                poses,velocity,gates,rates=motion(t)
                return poses*query.actuation_scale,velocity*query.actuation_scale,gates,rates
            motions.append(scaled)
    from dataclasses import replace
    rigid=[replace(body,velocity=body.velocity*query.velocity_scale)for body in scene.rigid.values()]
    scene=CoupledScene(scene.models,pair_contact=scene.pair,rigid_bodies=rigid,motions=motions)
    u=[a.copy()for a in state.u];v=[a*query.velocity_scale for a in state.v]
    if query.initial_rotation_deg:
        angle=np.deg2rad(query.initial_rotation_deg);c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        for m,a in zip(scene.models,u):
            center=m.rest.mean(0);a[:]=(m.rest+a-center)@rotation.T+center-m.rest
    if query.initial_height_m:
        for a in u:a[:,2]+=query.initial_height_m
    state=scene.initial(u,v)
    meta.update(public_query=asdict(query),material_source='explicit provider; no private world read')
    return scene,state,meta


def build_reference(world,query,**kwargs):
    expected=2 if query.family=='deformable_pair'else 1
    if len(world.materials)!=expected:raise ValueError('World object count differs from public scene')
    scene,state,meta=build_prediction(query,world.material_provider,**kwargs)
    meta.update(world_id=world.world_id,private_world_sha256=world.digest)
    return scene,state,meta


def build_boundary_geometry(query,**kwargs):
    """Spring/particle adapters use only geometry, mass, boundaries and initial state.

    The explicit placeholder bulk is never used as their internal material force.
    No private World is accessed; native learned springs supply internal forces.
    """
    placeholder=Material(log_young=0.)
    scene,state,meta=build_prediction(query,lambda index:placeholder.provider(),**kwargs)
    meta['material_source']='Geometry-only placeholder; adapter supplies internal force'
    return scene,state,meta
