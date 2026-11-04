"""Camera-independent scene edits; explicit SI forces and position controls.

Initial integration pilot, not inferred scene understanding or calibrated posterior.
"""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
import torch
from scripts.complex_scenes_v18.scene import CoupledScene
from scripts.complex_scenes_v18.geometry import SceneBody
from scripts.complex_scenes_v18.scenarios import smooth
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,AttachmentSpec,AnalyticNeoHookean
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane

@dataclass(frozen=True)
class PhysicalHypothesis:
    young_pa:float
    poisson:float=.35
    density_kg_m3:float=625.
    def __post_init__(self):
        if not np.isfinite([self.young_pa,self.poisson,self.density_kg_m3]).all() or self.young_pa<=0 or self.density_kg_m3<=0 or not -1<self.poisson<.5:raise ValueError('Invalid material')

@dataclass(frozen=True)
class Intervention:
    kind:str
    gravity:tuple=(0.,0.,-9.81)
    displacement_m:tuple=(0.,0.,.025)
    force_N:tuple=(.25,0.,0.)
    force_center_m:tuple=(.09,0.,0.)
    force_width_m:float=.025
    start_s:float=.02
    end_s:float=.08
    release_s:float=.10
    floor_z_m:float=-.08
    def __post_init__(self):
        if self.kind not in ('lift_release','force_pulse','drop'):raise ValueError('Unknown intervention')
        vals=[*self.gravity,*self.displacement_m,*self.force_N,*self.force_center_m,self.force_width_m,self.start_s,self.end_s,self.release_s,self.floor_z_m]
        if not np.isfinite(vals).all() or self.force_width_m<=0 or not 0<=self.start_s<self.end_s or self.release_s<0:raise ValueError('Invalid action')
        if any(len(v)!=3 for v in (self.gravity,self.displacement_m,self.force_N,self.force_center_m)):raise ValueError('3D vectors required')
    @property
    def digest(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class Camera:
    world_from_camera:np.ndarray
    intrinsics:np.ndarray
    def project(self,points):
        C=np.asarray(self.world_from_camera);K=np.asarray(self.intrinsics)
        if C.shape!=(4,4) or K.shape!=(3,3):raise ValueError('Camera shape')
        x=(np.asarray(points)-C[:3,3])@C[:3,:3]
        if np.any(x[...,2]<=0):raise ValueError('Points behind camera')
        q=x@K.T
        return q[...,:2]/q[...,2,None]

class ForcedScene(CoupledScene):
    def __init__(self,models,action,motions):
        self.action=action
        rest=models[0].rest;weights=np.exp(-.5*np.sum((rest-np.array(action.force_center_m))**2,axis=1)/action.force_width_m**2)
        weights*=models[0].nodal_volumes
        self.force_weights=weights/weights.sum()
        super().__init__(models,motions=motions)
    def load(self,t):
        a=self.action
        if a.kind!='force_pulse' or not a.start_s<t<a.end_s:return np.zeros(3),np.zeros(3)
        phase=np.pi*(t-a.start_s)/(a.end_s-a.start_s)
        return np.asarray(a.force_N)*np.sin(phase)**2,np.asarray(a.force_N)*np.sin(2*phase)*np.pi/(a.end_s-a.start_s)
    def evaluate(self,u,rigid,t):
        forces,rf,U,power,diag=super().evaluate(u,rigid,t)
        F,dF=self.load(t);nodal=self.force_weights[:,None]*F
        x=self.models[0].rest+u[0]
        forces[0]+=nodal;U-=float(np.sum(nodal*x))
        power-=float(np.sum(self.force_weights[:,None]*dF*x))
        diag['applied_force_N']=F.tolist()
        return forces,rf,U,power,diag

def build(hypothesis,action,device='cpu',n=1):
    if not isinstance(hypothesis,PhysicalHypothesis) or not isinstance(action,Intervention):raise TypeError('Explicit material and action required')
    attached=action.kind=='lift_release'
    m=SceneBody(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=3,surface_order=4,
        density=hypothesis.density_kg_m3,device=device,bulk_energy=AnalyticNeoHookean(hypothesis.poisson),
        material_latent_field=lambda q:q.new_full((*q.shape[:-1],1),np.log(hypothesis.young_pa)),
        gravity=action.gravity,support_mode='bilateral' if attached else 'free',
        attachments=(AttachmentSpec(6000,(.5,.5),(.32,.32)),)*2,
        colliders=(RigidPlane(point=(0,0,action.floor_z_m),penalty_N_m3=4e7,name='floor'),))
    def motion(t):
        rise,rate=smooth(t,0,.06);released,dr=smooth(t,action.release_s,.02)
        poses=np.zeros((2,6));poses[:,:3]=rise*np.asarray(action.displacement_m)
        velocity=np.broadcast_to(rate*np.asarray(action.displacement_m),(2,3)).copy()
        return poses,velocity,np.full(2,1-released),np.full(2,-dr)
    scene=ForcedScene([m],action,[motion if attached else None]);return scene,scene.initial()
