"""Declared development scenes. No held-world data or recipe selection here."""
import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,AttachmentSpec,AnalyticNeoHookean
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere
from scripts.compositional_dynamics_v17.dynamics import SphereBody
from scripts.complex_scenes_v18.geometry import SceneBody,RigidBowl,RigidCylinder
from scripts.complex_scenes_v18.pair_contact import DeformablePairContact
from scripts.complex_scenes_v18.scene import CoupledScene

FAMILIES=['clamp_release','curved_push','container_drop','rigid_impact','deformable_pair','heterogeneous']


def smooth(t,start,duration):
    a=np.clip((t-start)/duration,0.,1.);f=np.clip(a**3*(10-15*a+6*a*a),0.,1.)
    v=30*a*a*(1-a)**2/duration if 0<a<1 else 0.
    return f,v


def build_case(name,n=1,surface_samples=5,variant=0,device='cpu',contact_device=None):
    if name not in FAMILIES or variant not in (0,1):raise ValueError('Undeclared development scenario')
    dimensions=(.12,.06,.06);base=dict(n=n,order=3,surface_order=4,density=625,device=device,bulk_energy=AnalyticNeoHookean(.35),material_latent_field=lambda q:q.new_full((*q.shape[:-1],1),np.log(12000.)))
    specs=(AttachmentSpec(6000,(.5,.5),(.32,.32)),AttachmentSpec(4500,(.55,.6),(.3,.3)))
    def body(**kw):return SceneBody(GeometrySpec(dimensions=dimensions),**dict(base,**kw))
    motion=None;rigid=[];pair=None;duration=.16;g=(0.,0.,0.)
    if name in ('clamp_release','heterogeneous'):
        displacement=np.array([[.012,0,-.005],[-.012,0,.012]])
        if variant:displacement[:,1]=[.009,-.009]
        def motion(t):
            f,df=smooth(t,0,.07);pose=np.zeros((2,6));pose[:,:3]=f*displacement
            gates=np.ones(2);rates=np.zeros(2)
            if name=='clamp_release':
                release,dr=smooth(t,.09,.03);gates[1]=1-release;rates[1]=-dr
            return pose,df*displacement,gates,rates
        if name=='heterogeneous':
            base['material_latent_field']=lambda q:(np.log(12000.)+np.log(3.)*torch.tanh(8*(q[...,0]-(.35 if variant else .6))))[...,None]
        models=[body(support_mode='bilateral',attachments=specs,gravity=g)]
    elif name=='curved_push':
        obstacle=RigidCylinder(center=(.06 if not variant else .08,0,-.065),radius=.025 if not variant else .032)
        displacement=np.array([[.004,0,-.027],[-.004,0,-.019]])
        def motion(t):
            f,df=smooth(t,0,.10);pose=np.zeros((2,6));pose[:,:3]=f*displacement
            return pose,df*displacement,np.ones(2),np.zeros(2)
        models=[body(support_mode='bilateral',attachments=specs,gravity=g,colliders=(obstacle,))]
    elif name=='container_drop':
        g=(0.,0.,-9.81);duration=.24
        colliders=[RigidBowl(center=(.06,0,-.08 if variant else -.07),curvature_radius=.10 if variant else .12)]
        if variant:colliders.append(RigidSphere(center=(.09,.035,-.06),radius=.015,name='obstacle'))
        models=[body(support_mode='free',gravity=g,colliders=colliders)]
    elif name=='rigid_impact':
        radius=.025 if not variant else .02;center=np.array([.06 if not variant else .08,0,.075]);duration=.12
        sphere=RigidSphere(center=center,radius=radius,name='striker')
        rigid=[SphereBody('striker',.08 if not variant else .12,center,np.array([.1 if variant else 0,0,-1.]),np.zeros(3))]
        models=[body(support_mode='bilateral',gravity=g,colliders=(sphere,),attachments=specs)]
    else:
        if variant:g=(0,0,-9.81);duration=.20
        else:duration=.16
        floor=(RigidPlane(point=(0,0,-.031),penalty_N_m3=4e7,name='floor'),)if variant else ()
        models=[body(support_mode='free',gravity=g,colliders=floor),body(support_mode='free',gravity=g,colliders=floor,material_latent_field=lambda q:q.new_full((*q.shape[:-1],1),np.log(18000.)))]
        samples=tuple(2*c+1 for c in models[0].cells)if surface_samples==0 else surface_samples
        pair=DeformablePairContact(models,samples=samples,penalty_N_m3=4e7,device=contact_device or device)
    scene=CoupledScene(models,pair_contact=pair,rigid_bodies=rigid,motions=[motion]*len(models))
    u=[np.zeros_like(m.rest)for m in models];v=[np.zeros_like(m.rest)for m in models]
    if name=='deformable_pair':
        if variant:u[1][:]=[.008,.006,.09]
        else:
            u[1][:]=[.18,.004,.003];v[0][:]=[.4,0,0];v[1][:]=[-.4,0,0]
    return scene,scene.initial(u,v),dict(family=name,variant=variant,duration_s=duration,dimensions_m=dimensions,development_only=True,reference_qualified=False,contact_note='Finite-penalty sampled contact; no friction or CCD')
