"""Declared scenarios and geometry for the moving-camera Isaac dataset.

Pure NumPy helpers; no simulator imports. Forces are world-frame Newtons and
are normal/tangential to the reference surface, not follower pressure fields.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from boundary_block import block_mesh, reference_gaussians

BOUNDARIES=('point_pin','fixed_patch','two_patches','moving_patch')
LOADS=('push_pulse','pull_hold','shear_oscillation','twist_couple')
MODULI=(1500.,5000.,20000.)
SPARSE_FRAMES=(0,5,10,15,20,40,45,50,55,60)


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_plan():
    cases=[]
    for mi,modulus in enumerate(MODULI):
        for li,load in enumerate(LOADS):
            for bi,boundary in enumerate(BOUNDARIES):
                group=4*li+bi; rng=np.random.default_rng(421000+group)
                # Nuisance properties vary independently of stiffness and repeat
                # across each paired material triplet.
                material={'youngs_modulus_Pa':modulus,'poissons_ratio':float(rng.uniform(.28,.42)),
                          'density_kg_m3':float(rng.uniform(400,850)),
                          'linear_damping_s_inverse':float(rng.uniform(.35,.85)),
                          'elasticity_damping':.02}
                split=('train','train','validation','test')[(li+bi)%4]
                case={'episode_id':f'e{len(cases):04d}','group_id':f'g{group:02d}','split':split,
                      'seed':431000+group,'material':material,'boundary_type':boundary,
                      'force_type':load,'seconds':3.,'physics_hz':960,'fps':20,
                      'cells':[12,6,4],'load_patch_nodes':9,'sparse_frames':list(SPARSE_FRAMES),
                      'units':{'length':'m','time':'s','force':'N','modulus':'Pa','density':'kg/m^3'},
                      'qualification':'Fixed-discretization development simulation; no calibrated-mechanics claim'}
                case['recipe_sha256']=digest_json(case);cases.append(case)
    plan={'dataset':'isaac-boundary-orbit-v0.1','episodes':cases,'episode_count':len(cases),
          'design':'3 stiffness levels x4 world-load families x4 boundary types; independently varied nuisance properties',
          'split_rule':'group=(load,boundary,seed); Latin-square split keeps material triplets together',
          'source_policy':'Freeze code and plan before generation; never silently replace failed scenarios'}
    plan['plan_sha256']=digest_json(plan)
    return plan


def face_ids(points,axis,sign):
    value=points[:,axis].max() if sign>0 else points[:,axis].min()
    return np.flatnonzero(np.isclose(points[:,axis],value,atol=1e-9))


def compact_patch(points,candidates,center,count):
    candidates=np.asarray(candidates,dtype=int)
    ids=candidates[np.argsort(np.linalg.norm(points[candidates]-center,axis=1),kind='stable')[:count]]
    if len(ids)!=count or np.linalg.norm(points[ids]-center,axis=1).max()>.09:
        raise ValueError('Cannot form requested compact surface patch')
    return ids


def geometry(case):
    points,tetrahedra,faces=block_mesh(cells=case['cells'])
    rng=np.random.default_rng(case['seed']);bounds=np.stack([points.min(0),points.max(0)])
    middle=bounds.mean(0);size=bounds[1]-bounds[0]
    axis=int(rng.integers(3));sign=int(rng.choice([-1,1]));centers=[];fixed=[]
    bc=case['boundary_type'];support_faces=[]
    for side in ([sign,-sign] if bc=='two_patches' else [sign]):
        center=middle+rng.uniform(-.18,.18,3)*size;center[axis]=bounds[int(side>0),axis]
        count=1 if bc=='point_pin' else (9 if bc=='two_patches' else 16)
        patch=compact_patch(points,face_ids(points,axis,side),center,count)
        fixed.extend(patch.tolist());centers.append(center.tolist());support_faces.append([axis,side])
    fixed=np.asarray(sorted(set(fixed)),dtype=int)
    surface=np.unique(faces);distance=np.linalg.norm(points[surface,None]-points[fixed][None],axis=-1).min(1)
    available=surface[distance>.027];used=set(map(int,fixed));patches=[];normals=[];load_nodes=[]
    for _ in range(2):
        options=[]
        for ax in range(3):
            for side in [-1,1]:
                candidates=np.array([int(i) for i in face_ids(points,ax,side) if i in available and int(i) not in used])
                if len(candidates)>=case['load_patch_nodes']:options.append((ax,side,candidates))
        ax,side,candidates=options[int(rng.integers(len(options)))]
        # Prefer an unsupported region, with seeded jitter instead of always a corner.
        center=middle+rng.uniform(-.25,.25,3)*size;center[ax]=bounds[int(side>0),ax]
        patch=compact_patch(points,candidates,center,case['load_patch_nodes'])
        node=int(patch[0]);patches.append(patch.tolist());load_nodes.append(node);used.update(map(int,patch))
        normal=np.zeros(3);normal[ax]=side;normals.append(normal.tolist())
    load_centers=points[load_nodes]
    separation=load_centers[1]-load_centers[0]
    trial=rng.normal(size=3);couple=np.cross(separation,trial);couple/=np.linalg.norm(couple)
    normal=np.asarray(normals[0]);tangent=np.cross(normal,np.eye(3)[int(np.argmin(np.abs(normal)))])
    amplitude=float(rng.uniform(.12,.24))
    if bc=='point_pin':amplitude*=.25
    motion=rng.normal(size=3);motion[axis]=0;motion/=np.linalg.norm(motion)
    camera={'center_m':middle.tolist(),'radius_m':float(rng.uniform(.88,.94)),
            'azimuth_start_deg':float(rng.uniform(-120,-65)),'arc_deg':float(rng.uniform(85,125)),
            'elevation_deg':float(rng.uniform(20,33)),'elevation_excursion_deg':float(rng.uniform(3,7)),
            'focal_length_mm':35.,'horizontal_aperture_mm':36.,'vertical_aperture_mm':27.,
            'resolution':[640,480],'path_depends_on_hidden_motion':False}
    spec={'boundary_type':bc,'fixture_nodes':fixed.tolist(),'support_centers_m':centers,
          'support_faces':support_faces,'motion_amplitude_m':.008 if bc=='moving_patch' else 0.,
          'motion_direction_world':motion.tolist(),'load_nodes':load_nodes,'load_patches':patches,
          'reference_load_normals':normals,'force_type':case['force_type'],'peak_force_N':amplitude,
          'normal_direction_world':normal.tolist(),'shear_direction_world':tangent.tolist(),
          'couple_direction_world':couple.tolist(),'seconds':case['seconds'],'camera':camera,
          'force_convention':'World directions defined from reference geometry; no follower pressure',
          'handle_mass_kg_each':.001,'seed':case['seed']}
    return points,tetrahedra,faces,spec


def envelope(kind,time,seconds):
    u=np.asarray(time,dtype=float)/seconds
    if kind in ['push_pulse','twist_couple']:
        phase=(u-.1)/.45
        return np.where((phase>0)&(phase<1),np.sin(np.pi*np.clip(phase,0,1))**2,0.)
    if kind=='pull_hold':
        rise=np.sin(np.pi/2*np.clip((u-.1)/.2,0,1))**2
        fall=np.cos(np.pi/2*np.clip((u-.65)/.25,0,1))**2
        return rise*fall
    if kind=='shear_oscillation':
        phase=np.clip((u-.1)/.8,0,1)
        return np.sin(4*np.pi*phase)*np.sin(np.pi*phase)**2
    raise ValueError('Unknown load family')


def forces_at(spec,time):
    kind=spec['force_type'];amplitude=spec['peak_force_N']*float(envelope(kind,time,spec['seconds']))
    force=np.zeros((2,3),dtype=np.float32)
    if kind=='push_pulse':force[0]=-amplitude*np.asarray(spec['normal_direction_world'])
    elif kind=='pull_hold':force[0]=amplitude*np.asarray(spec['normal_direction_world'])
    elif kind=='shear_oscillation':force[0]=amplitude*np.asarray(spec['shear_direction_world'])
    elif kind=='twist_couple':
        force[0]=amplitude*np.asarray(spec['couple_direction_world']);force[1]=-force[0]
    else:raise ValueError('Unknown force type')
    return force


def boundary_shift(spec,time):
    phase=np.clip(float(time)/spec['seconds'],0,1)
    return spec['motion_amplitude_m']*np.sin(np.pi*phase)**2*np.asarray(spec['motion_direction_world'])


def camera_pose(spec,time):
    camera=spec['camera'];u=np.clip(float(time)/spec['seconds'],0,1)
    azimuth=np.deg2rad(camera['azimuth_start_deg']+camera['arc_deg']*(3*u*u-2*u*u*u))
    elevation=np.deg2rad(camera['elevation_deg']+camera['elevation_excursion_deg']*np.sin(np.pi*u))
    radius=camera['radius_m']+.025*np.sin(np.pi*u)
    center=np.asarray(camera['center_m']);eye=center+radius*np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
    back=(eye-center)/np.linalg.norm(eye-center);right=np.cross([0,0,1],back);right/=np.linalg.norm(right)
    up=np.cross(back,right);pose=np.eye(4);pose[:3,:3]=np.stack([right,up,back]);pose[3,:3]=eye
    return pose


def appearance(points,faces):
    grid=np.rint((points-points.min(0))/.02).astype(int)
    palette=np.array([[.12,.3,.65],[.72,.43,.08]])
    colors=palette[(grid.sum(1)%2)]
    g=reference_gaussians(points,faces)
    g['colors']=np.einsum('ni,nij->nj',g['barycentric_weights'],colors[g['face_vertex_ids']]).astype(np.float32)
    return colors,g


def load_model_inputs(episode):
    """Only public model inputs: no recipe, material label or hidden trajectory."""
    episode=Path(episode)
    with np.load(episode/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
    with np.load(episode/'observations/cameras.npz') as raw:cameras={k:raw[k] for k in raw.files}
    return {'reference':reference,'cameras':cameras,
            'rgb_paths':sorted((episode/'observations/rgb').glob('*.png')),
            'mask_paths':sorted((episode/'observations/masks').glob('*.png')),
            'depth_paths':sorted((episode/'observations/depth').glob('*.npz'))}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=make_plan();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'episodes':len(plan['episodes']),'sha256':plan['plan_sha256']}))
