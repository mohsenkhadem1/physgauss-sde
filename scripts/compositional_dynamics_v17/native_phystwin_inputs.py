"""Strict shared-information export for a later matched native PhysTwin fit.
No fitting/evaluation or family labels. Static geometry and mass are known to both methods.
"""
import numpy as np

def export_prefix(*,initial_tracked_m,observed_prefix_m,visible_prefix,controller_commands_m,
                  other_surface_m,interior_m,object_masses_kg,frame_dt_s):
    initial=np.asarray(initial_tracked_m,float);prefix=np.asarray(observed_prefix_m,float)
    visible=np.asarray(visible_prefix);commands=np.asarray(controller_commands_m,float)
    surface=np.asarray(other_surface_m,float).reshape(-1,3);interior=np.asarray(interior_m,float).reshape(-1,3)
    if initial.ndim!=2 or initial.shape[1]!=3 or prefix.ndim!=3 or prefix.shape[1:]!=initial.shape:raise ValueError('Tracked point correspondence')
    K,N=prefix.shape[:2]
    if visible.dtype!=bool or visible.shape!=(K,N)or K<2:raise ValueError('Explicit prefix visibility and2frames required')
    if commands.ndim!=3 or commands.shape[2]!=3 or len(commands)<K:raise ValueError('Public controller trajectory required')
    if not np.isfinite(frame_dt_s)or frame_dt_s<=0:raise ValueError('Positive uniform frame clock')
    if not all(np.isfinite(a).all()for a in [initial,commands,surface,interior])or not np.isfinite(prefix[visible]).all():raise ValueError('Finite permitted observations/geometry')
    if not np.allclose(prefix[0][visible[0]],initial[visible[0]],atol=1e-9,rtol=0):raise ValueError('Declared knowninitial geometry must match tracked first frame')
    mass=np.asarray(object_masses_kg,float)
    if mass.shape!=(N+len(surface)+len(interior),)or not np.isfinite(mass).all()or np.any(mass<=0):raise ValueError('Known positive SI mass per objectpoint')
    T=len(commands);objects=np.broadcast_to(initial,(T,N,3)).copy();objects[:K]=np.where(visible[...,None],prefix,initial)
    visibility=np.zeros((T,N),bool);visibility[:K]=visible
    motions=np.zeros((T-1,N),bool);motions[:K-1]=visible[:-1]&visible[1:]
    data=dict(object_points=objects.astype(np.float32),object_colors=np.full((T,N,3),.5,np.float32),object_visibilities=visibility,
              object_motions_valid=motions,controller_points=commands.astype(np.float32),surface_points=surface.astype(np.float32),interior_points=interior.astype(np.float32))
    metadata=dict(train_frame_count=K,total_command_frames=T,frame_dt_s=float(frame_dt_s),gravity_m_s2=9.8,object_masses_kg=mass.tolist(),total_mass_kg=float(mass.sum()),
                  suffix_targets='Initial reference placeholders with visibility/motion-valid false; no futureobjecttrack input accepted',scope='Knowngeometry/knownmass shared prefix exporter only; native fitting and held dynamics not yet benchmarked')
    return data,metadata

def apply_known_object_masses(native_tuple,object_masses_kg):
    """Call after unchanged native _init_start; only its public mass output changes.
    Controller masses remain native placeholders; they are kinematic and not integrated.
    """
    import torch
    vertices,springs,lengths,masses,count=native_tuple
    known=torch.as_tensor(object_masses_kg,dtype=masses.dtype,device=masses.device)
    if known.ndim!=1 or len(known)>len(masses)or not torch.isfinite(known).all()or torch.any(known<=0):raise ValueError('Positive finite known objectmasses')
    updated=masses.clone();updated[:len(known)]=known
    return vertices,springs,lengths,updated,count
