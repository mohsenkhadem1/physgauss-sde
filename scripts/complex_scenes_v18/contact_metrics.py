"""Common saved-surface contact diagnostics; no force/physics fitting.

Bowl gaps use the declared normalized implicit gap. Pair penetration is sampled
vertex-to-triangle depth, not continuous collision detection or exact overlap.
"""
from dataclasses import replace
import numpy as np
import torch
from scripts.complex_scenes_v18.world_queries import build_boundary_geometry
from scripts.complex_scenes_v18.pair_contact import material_surface,closest_squared,inside_closed_surface


def trace(query,positions,times,rigid_centers=None):
    scene,_,_=build_boundary_geometry(query);positions=np.real_if_close(positions);times=np.asarray(times)
    if positions.shape!=(len(times),len(scene.models),386,3)or not np.isfinite(positions).all():raise ValueError('Common surface ordering and finite states required')
    names=list(scene.rigid)
    if names and (rigid_centers is None or np.asarray(rigid_centers).shape!=(len(times),len(names),3)):raise ValueError('Predicted moving rigid state is mandatory')
    gaps={};faces=torch.tensor(material_surface(9)[1],dtype=torch.long)
    for frame,xyz in enumerate(positions):
        points=[torch.tensor(x,dtype=torch.float64)for x in xyz]
        for body,(model,x)in enumerate(zip(scene.models,points)):
            for collider in model.colliders:
                if collider.name in names:collider=replace(collider,center=np.asarray(rigid_centers)[frame,names.index(collider.name)])
                key=f'body_{body}:{collider.name}';gap=float(collider.signed_distance(x).min())
                gaps.setdefault(key,[]).append(gap)
        if len(points)==2:
            minimum=float('inf')
            for source,target in [(points[0],points[1]),(points[1],points[0])]:
                distance=torch.sqrt(closest_squared(source,target,faces).clamp_min(0));inside=inside_closed_surface(source,target,faces)
                signed=torch.where(inside,-distance,distance);minimum=min(minimum,float(signed.min()))
            gaps.setdefault('body_0:body_1',[]).append(minimum)
    events={}
    for key,values in gaps.items():
        gap=np.asarray(values);contact=gap<=0;indices=np.flatnonzero(contact)
        events[key]=dict(minimum_gap_m=gap.tolist(),maximum_sampled_penetration_m=float(np.maximum(-gap,0).max()),first_contact_s=float(times[indices[0]])if len(indices)else None,contact_frames=int(contact.sum()))
    return dict(applicable=bool(events),events=events,temporal_resolution_s=float(np.max(np.diff(times)))if len(times)>1 else None,scope='Sampled common surface; normalized implicit bowl gap; no continuous collision or exact intersection claim')


def compare(reference,prediction):
    if set(reference['events'])!=set(prediction['events']):raise ValueError('Matching scene event identities required')
    rows=[]
    for key,a in reference['events'].items():
        b=prediction['events'][key];ta,tb=a['first_contact_s'],b['first_contact_s']
        row=dict(event=key,missed_contact=ta is not None and tb is None,extra_contact=ta is None and tb is not None,first_contact_error_s=abs(tb-ta)if ta is not None and tb is not None else None,maximum_penetration_error_m=abs(b['maximum_sampled_penetration_m']-a['maximum_sampled_penetration_m']))
        rows.append(row)
    return rows
