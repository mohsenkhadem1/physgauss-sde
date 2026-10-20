"""Explicit RGB-D per-scene geometric baseline using public measurements only.

Projective depth/point-to-plane correspondence is recomputed from the CURRENT
predicted geometry and observed depth. No true motion, material correspondence,
boundary assignment, material value or truth visibility is used. Gaussian RBF
controls and rigid motion are fitted at observation-time knots, then directly
interpolated at query times. This is a depth-assisted geometric baseline, not an
RGB-only method, physical identification or an unseen-poke predictor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
if __name__ == '__main__' and not any(shutil.which(name) for name in ('cc','gcc','clang')):
    os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class GeometricConfig:
    controls: int = 48
    bandwidth_fraction: float = .3
    interpolation: str = 'hermite'
    depth_scale_m: float = .003
    occlusion_margin_m: float = .025
    association_gate_m: float = .05
    normal_cosine: float = .15
    depth_weight: float = 1.
    point_to_plane_weight: float = 1.
    silhouette_weight: float = .02
    color_weight: float = 0.
    silhouette_quantization_px: float = 1.
    edge_weight: float = .01
    control_weight: float = .001
    temporal_weight: float = .001
    inversion_weight: float = .1
    motion_weight: float = 1.

    def __post_init__(self):
        if self.controls<4 or self.bandwidth_fraction<=0 or self.depth_scale_m<=0:
            raise ValueError('Positive geometric dimensions and residual scale required')
        if self.interpolation not in ('linear','hermite'):
            raise ValueError('Expected linear or Hermite interpolation')
        if self.occlusion_margin_m<0 or self.association_gate_m<=0:
            raise ValueError('Invalid projective correspondence gates')


def farthest_point_indices(points, count):
    """Deterministic geometry-only control selection; never inspect trajectories."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not 1<=count<=len(points):
        raise ValueError('Invalid point set/control count')
    chosen=[int(np.argmin(np.sum((points-points.mean(0))**2,-1)))]
    distance=np.sum((points-points[chosen[0]])**2,-1)
    for _ in range(count-1):
        distance[chosen]=-np.inf;index=int(np.argmax(distance));chosen.append(index)
        distance=np.minimum(distance,np.sum((points-points[index])**2,-1))
    return np.asarray(chosen,dtype=int)


def normalized_rbf(points,centers,bandwidth):
    squared=(points[:,None]-centers[None]).square().sum(-1)
    # Subtracting the maximum exponent prevents underflow for distant queries.
    exponent=-squared/(2*bandwidth**2)
    return torch.softmax(exponent,-1)


def rotation_matrix(axis_angle):
    """Stable differentiable Rodrigues map, including zero-angle gradients."""
    x,y,z=axis_angle.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*axis_angle.shape[:-1],3,3)
    angle=torch.linalg.vector_norm(axis_angle,dim=-1)
    a=torch.sinc(angle/torch.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*torch.pi)).square())[...,None,None]
    return torch.eye(3,device=axis_angle.device,dtype=axis_angle.dtype)+a*skew+b*(skew@skew)


def interpolate_knots(knots,values,queries,mode='hermite'):
    """Interpolate only observed-time parameters; clamp outside the input span.

    Hermite uses nonuniform central secants and zero initial velocity, matching
    the declared rest start. It exactly reproduces every observation knot.
    """
    queries=torch.as_tensor(queries,device=knots.device,dtype=knots.dtype).reshape(-1)
    if len(knots)<2 or bool(torch.any(knots[1:]<=knots[:-1])):
        raise ValueError('Need at least two strictly increasing knot times')
    if values.shape[0]!=len(knots):raise ValueError('Knot/value count mismatch')
    t=queries.clamp(knots[0],knots[-1]);right=torch.searchsorted(knots,t,right=True).clamp(1,len(knots)-1)
    left=right-1;dt=knots[right]-knots[left];u=(t-knots[left])/dt
    shape=(-1,)+(1,)*(values.ndim-1);u=u.reshape(shape);duration=dt.reshape(shape)
    a,b=values[left],values[right]
    if mode=='linear':return (1-u)*a+u*b
    if mode!='hermite':raise ValueError('Unknown interpolation mode')
    slopes=torch.zeros_like(values)
    slopes[-1]=(values[-1]-values[-2])/(knots[-1]-knots[-2])
    if len(knots)>2:
        denominator=(knots[2:]-knots[:-2]).reshape((-1,)+(1,)*(values.ndim-1))
        slopes[1:-1]=(values[2:]-values[:-2])/denominator
    return (2*u**3-3*u**2+1)*a+(u**3-2*u**2+u)*duration*slopes[left] + \
        (-2*u**3+3*u**2)*b+(u**3-u**2)*duration*slopes[right]


class RBFGeometricTrajectory(nn.Module):
    def __init__(self,reference,observation_times,config=GeometricConfig()):
        super().__init__();self.config=config
        rest=torch.as_tensor(reference['rest_points_m'],dtype=torch.float32)
        knots=torch.as_tensor(np.unique(observation_times),dtype=rest.dtype)
        if len(knots)<2 or abs(float(knots[0]))>1e-7:
            raise ValueError('This controlled baseline requires a known rest frame at time zero')
        candidates=np.unique(reference['surface_faces'])
        ids=candidates[farthest_point_indices(rest[candidates].numpy(),min(config.controls,len(candidates)))]
        centers=rest[ids];length=(rest.amax(0)-rest.amin(0)).max()
        self.register_buffer('rest',rest);self.register_buffer('knots',knots)
        self.register_buffer('centers',centers);self.register_buffer('length',length)
        self.register_buffer('center',(rest.amax(0)+rest.amin(0))/2)
        self.register_buffer('basis',normalized_rbf(rest,centers,length*config.bandwidth_fraction))
        self.register_buffer('free_knots',(knots>0).to(rest.dtype))
        self.control_displacements=nn.Parameter(torch.zeros(len(knots),len(centers),3))
        self.translations=nn.Parameter(torch.zeros(len(knots),3))
        self.rotations=nn.Parameter(torch.zeros(len(knots),3))

    def knot_parameters(self):
        weight=self.control_displacements-self.control_displacements.mean(1,keepdim=True)
        return {'controls':weight*self.free_knots[:,None,None],
                'translation':self.translations*self.free_knots[:,None],
                'rotation':self.rotations*self.free_knots[:,None]}

    def forward(self,times):
        p=self.knot_parameters()
        params={k:interpolate_knots(self.knots,v,times,self.config.interpolation) for k,v in p.items()}
        local=self.rest[None]-self.center+self.length*torch.einsum('nc,tcj->tnj',self.basis,params['controls'])
        rotation=rotation_matrix(params['rotation'])
        return local@rotation.transpose(-1,-2)+self.center+self.length*params['translation'][:,None]


def project_world(points,pose,k):
    """World points to USD camera pixels (edge coordinates) and positive z."""
    camera=(points-pose[3,:3])@pose[:3,:3].T
    z=-camera[:,2];safe=z.clamp_min(1e-6)
    uv=torch.stack((k[0,0]*camera[:,0]/safe+k[0,2],k[1,2]-k[1,1]*camera[:,1]/safe),-1)
    return uv,z


def unproject_depth(uv,depth,pose,k):
    camera=torch.stack(((uv[...,0]-k[0,2])/k[0,0]*depth,
                        -(uv[...,1]-k[1,2])/k[1,1]*depth,-depth),-1)
    return camera@pose[:3,:3]+pose[3,:3]


def sample_image(image,uv):
    """Bilinear sampling at pixel-edge coordinates; no gradients through images."""
    if image.ndim==2:image=image[...,None]
    h,w=image.shape[:2]
    grid=uv.new_tensor([2/w,2/h])*uv-1
    return functional.grid_sample(image.permute(2,0,1)[None],grid[None,None],
                                  mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,0].T


def sample_axis_depth(depth,uv):
    """Perspective-correct planar sampling: inverse camera-axis depth is affine."""
    positive=torch.isfinite(depth)&(depth>0)
    inverse=torch.where(positive,depth.clamp_min(1e-12).reciprocal(),torch.zeros_like(depth))
    support=sample_image(positive.to(depth.dtype),uv)[:,0]>.999
    sampled_inverse=sample_image(inverse,uv)[:,0]
    value=torch.where(support,sampled_inverse.clamp_min(1e-12).reciprocal(),torch.zeros_like(sampled_inverse))
    return value,support


def depth_normals(depth,mask,pose,k):
    """Observed world normals from depth differences; no geometry truth input."""
    h,w=depth.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=depth.device,dtype=depth.dtype)+.5,
                         torch.arange(w,device=depth.device,dtype=depth.dtype)+.5,indexing='ij')
    points=unproject_depth(torch.stack((xx,yy),-1),depth,pose,k)
    normal=torch.zeros_like(points);valid=torch.zeros_like(mask,dtype=torch.bool)
    du=points[1:-1,2:]-points[1:-1,:-2];dv=points[2:,1:-1]-points[:-2,1:-1]
    inner=torch.linalg.cross(du,dv)
    norm=torch.linalg.vector_norm(inner,dim=-1,keepdim=True)
    inner=inner/norm.clamp_min(1e-12)
    facing=((pose[3,:3]-points[1:-1,1:-1])*inner).sum(-1,keepdim=True)
    normal[1:-1,1:-1]=torch.where(facing>=0,inner,-inner)
    confident=(depth>0)&torch.isfinite(depth)&mask.bool()
    valid[1:-1,1:-1]=(confident[1:-1,1:-1]&confident[1:-1,:-2]&confident[1:-1,2:]&
                     confident[:-2,1:-1]&confident[2:,1:-1]&(norm[...,0]>1e-10))
    # Pixel normals crossing a crease do not define a valid single-plane match.
    # This rejection uses observed depth alone, not reference/truth visibility.
    stable=valid.clone()
    for axis in (0,1):
        for offset in (-1,1):
            stable &= torch.roll(valid,offset,axis)
            stable &= (normal*torch.roll(normal,offset,axis)).sum(-1)>.98
    return normal,stable


def observed_silhouette_support(mask,directions):
    ys,xs=torch.nonzero(mask,as_tuple=True)
    if len(xs)==0:raise ValueError('Empty observed object mask')
    pixels=torch.stack((xs.to(directions.dtype)+.5,ys.to(directions.dtype)+.5),-1)
    return (pixels@directions.T).amax(0)


def projective_correspondence(points,normals,pose,k,depth,observed_mask,
                              normal_map,normal_valid,config=GeometricConfig()):
    """Return current-geometry/observed-depth ICP data, never material tracks."""
    uv,z=project_world(points,pose,k)
    observed_z,support=sample_axis_depth(depth,uv)
    support = support & (sample_image(observed_mask.to(depth.dtype),uv)[:,0]>.999)
    normal_support=sample_image(normal_valid.to(depth.dtype),uv)[:,0]>.999
    measured_normal=sample_image(normal_map,uv)
    measured_normal=measured_normal/torch.linalg.vector_norm(measured_normal,dim=-1,keepdim=True).clamp_min(1e-12)
    toward_camera=pose[3,:3]-points
    toward_camera=toward_camera/torch.linalg.vector_norm(toward_camera,dim=-1,keepdim=True).clamp_min(1e-12)
    front=(normals*toward_camera).sum(-1)>config.normal_cosine
    alignment=(normals*measured_normal).sum(-1)>config.normal_cosine
    difference=z-observed_z
    inlier=(support&normal_support&front&alignment&(z>.01)&
            (difference<config.occlusion_margin_m)&(difference.abs()<config.association_gate_m))
    target=unproject_depth(uv,observed_z,pose,k)
    return {'uv':uv,'valid':inlier,'axis_residual_m':difference,
            'plane_residual_m':((points-target)*measured_normal).sum(-1),
            'front_facing':front,'observed_support':support}


def unique_edges(tetrahedra):
    t=np.asarray(tetrahedra)
    return np.unique(np.sort(np.concatenate([t[:,[a,b]] for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]),axis=-1),axis=0)


def volume_ratios(positions,rest,tetrahedra):
    vertices=positions[...,tetrahedra,:]
    rest_vertices=rest[tetrahedra]
    numerator=torch.linalg.det((vertices[...,1:,:]-vertices[...,:1,:]).transpose(-1,-2))
    denominator=torch.linalg.det((rest_vertices[:,1:]-rest_vertices[:,:1]).transpose(-1,-2))
    return numerator/denominator


def regularization_terms(model,positions,tetrahedra,edges):
    length=torch.linalg.vector_norm(positions[:,edges[:,0]]-positions[:,edges[:,1]],dim=-1)
    reference_length=torch.linalg.vector_norm(model.rest[edges[:,0]]-model.rest[edges[:,1]],dim=-1)
    edge=torch.log((length/reference_length).clamp_min(1e-5)).square().mean()
    ratio=volume_ratios(positions,model.rest,tetrahedra)
    parameters=model.knot_parameters();temporal=positions.new_zeros(())
    for values in parameters.values():
        shape=(-1,)+(1,)*(values.ndim-1)
        dt=(model.knots[1:]-model.knots[:-1]).reshape(shape)
        velocity=(values[1:]-values[:-1])/dt
        if len(velocity)>1:
            duration=((model.knots[2:]-model.knots[:-2])/2).reshape(shape)
            temporal=temporal+((velocity[1:]-velocity[:-1])/duration).square().mean()
    distance=torch.cdist(model.centers,model.centers)/model.length
    neighbours=distance.topk(min(5,len(model.centers)),largest=False).indices[:,1:]
    controls=parameters['controls']
    difference=controls[:,:,None]-controls[:,neighbours]
    scale=distance.gather(1,neighbours).clamp_min(.05)
    smooth=(difference/scale[None,:,:,None]).square().mean()
    # Zero-mean motion ridge prevents drift along unobservable plane tangents.
    # It is a declared geometric prior, not evidence that hidden points are fixed.
    motion=((positions-model.rest)/model.length).square().mean()
    return {'edge':edge,'temporal':temporal,'control':smooth,'motion':motion,
            'inversion':torch.relu(.2-ratio).square().mean(),'minimum_volume_ratio':ratio.min()}


def fit_episode(episode,output,*,steps=600,device='cuda:3',seed=2011,
                learning_rate=.0003,config=GeometricConfig(),size=(192,160),
                query_cameras=None,appearance_calibration=None,log_every=25,
                frame_batch_size=0):
    """RGB-D fitting at observed frames only; fixed final iteration selection."""
    try:
        from gaussian_observed_features import load_observed_arrays
        from gaussian_observation import project_reference_crops
    except ImportError:
        from experiments.gaussian_observed_features import load_observed_arrays
        from experiments.gaussian_observation import project_reference_crops
    if steps<1:raise ValueError('Positive fitting budget required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),Path(__file__).with_name('gaussian_observed_features.py'),Path(__file__).with_name('gaussian_observation.py')]
    source_bytes={p.name:p.read_bytes() for p in source_paths}
    source_hashes={k:hashlib.sha256(v).hexdigest() for k,v in source_bytes.items()}
    (output/'provenance').mkdir()
    for name,value in source_bytes.items():(output/'provenance'/name).write_bytes(value)
    (output/'run-spec.json').write_text(json.dumps({'config':asdict(config),'steps':steps,'seed':seed,
                                                  'device':device,'learning_rate':learning_rate,
                                                  'frame_batch_size':frame_batch_size,
                                                  'source_sha256':source_hashes},indent=2)+'\n')
    started=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if torch.device(device).type=='cuda':
        torch.empty(0,device=device) # initialize this device before allocator counters
        torch.cuda.reset_peak_memory_stats(device)
    inputs=load_observed_arrays(episode,'rgbd',size)
    ref,cameras=inputs['reference'],inputs['cameras']
    model=RBFGeometricTrajectory(ref,cameras['times_s'],config).to(device)
    def tensor(x,integer=False):return torch.as_tensor(x,device=device,dtype=torch.long if integer else torch.float32)
    pose=tensor(cameras['camera_to_world_usd_row']);k_native=tensor(inputs['native_K'])
    k_roi=tensor(cameras['K']);times=tensor(cameras['times_s'])
    depth=tensor(inputs['depth_native_m']);masks=tensor(inputs['native_mask'])>.5
    rgb=tensor(inputs['rgb']);roi_masks=tensor(inputs['mask'])
    normal_data=[depth_normals(d,m,p,k_native) for d,m,p in zip(depth,masks,pose)]
    normal_maps=torch.stack([n for n,_ in normal_data]);normal_valid=torch.stack([v for _,v in normal_data])
    face_ids=tensor(ref['face_vertex_ids'],True);weights=tensor(ref['barycentric_weights'])
    tetrahedra=tensor(ref['tetrahedra'],True);edges=tensor(unique_edges(ref['tetrahedra']),True)
    linear=np.asarray(ref['colors']);color=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    appearance={}
    if appearance_calibration:
        appearance=json.loads(Path(appearance_calibration).read_text())
        color=color@np.asarray(appearance['color_matrix'])+np.asarray(appearance['color_bias'])
    color=tensor(np.clip(color,0,1))
    angle=torch.arange(32,device=device,dtype=torch.float32)*2*torch.pi/32
    directions=torch.stack((angle.cos(),angle.sin()),-1)
    target_support=torch.stack([observed_silhouette_support(m,directions) for m in masks])
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    history=[];rejected=[];minimum_accepted=1.;learning_rate_factor=1.
    eligible=np.flatnonzero(np.asarray(cameras['times_s'])>0)
    if not len(eligible):raise ValueError('No non-rest observations')

    def frame_components(q,index):
        tri=q[face_ids]
        means=(tri*weights[:,:,None]).sum(1)
        normals=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals=normals/torch.linalg.vector_norm(normals,dim=-1,keepdim=True).clamp_min(1e-12)
        correspondence=projective_correspondence(means,normals,pose[index],k_native,depth[index],masks[index],normal_maps[index],normal_valid[index],config)
        valid=correspondence['valid'];count=int(valid.sum())
        zero=q.sum()*0
        components={}
        for name,key in [('depth','axis_residual_m'),('point_to_plane','plane_residual_m')]:
            residual=correspondence[key][valid]/config.depth_scale_m
            components[name]=functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=1.) if count else zero
        if config.color_weight>0:
            roi_uv,_=project_world(means,pose[index],k_roi[index])
            observed_color=sample_image(rgb[index],roi_uv)
            color_valid=valid&(sample_image(roi_masks[index],roi_uv)[:,0]>.99)
            components['color']=functional.smooth_l1_loss(color[color_valid],observed_color[color_valid],beta=.1) if bool(color_valid.any()) else zero
        else:components['color']=zero
        # Convex support functions constrain silhouettes without privileged IDs or a rasterizer.
        node_uv,_=project_world(q,pose[index],k_native)
        predicted_support=(node_uv@directions.T).amax(0)
        difference=predicted_support-target_support[index]
        # A hard raster mask constrains support only within about one pixel.
        # Fitting its quantized pixel centers exactly would invent subpixel motion.
        residual=torch.relu(difference.abs()-config.silhouette_quantization_px)/3.
        components['silhouette']=functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=1.)
        return components,count,int(correspondence['front_facing'].sum())

    for step in range(1,steps+1):
        indices=eligible if frame_batch_size<=0 else rng.choice(eligible,min(frame_batch_size,len(eligible)),replace=False)
        observation_positions=model(times[indices])
        frames=[frame_components(q,int(index)) for q,index in zip(observation_positions,indices)]
        components={key:torch.stack([f[0][key] for f in frames]).mean() for key in frames[0][0]}
        count=sum(f[1] for f in frames);front_count=sum(f[2] for f in frames)
        knot_positions=model(model.knots)
        priors=regularization_terms(model,knot_positions,tetrahedra,edges)
        components.update({k:v for k,v in priors.items() if k!='minimum_volume_ratio'})
        loss=sum(getattr(config,name+'_weight')*value for name,value in components.items())
        if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        previous=[p.detach().clone() for p in model.parameters()]
        schedule=.01+.99*.5*(1+np.cos(np.pi*(step-1)/max(steps-1,1)))
        for group in optimizer.param_groups:group['lr']=learning_rate*learning_rate_factor*schedule
        optimizer.step()
        with torch.no_grad():
            accepted=model(model.knots);ratio=volume_ratios(accepted,model.rest,tetrahedra)
            finite=bool(torch.isfinite(accepted).all());minimum=float(ratio.min())
            if not finite or minimum<.05:
                for parameter,value in zip(model.parameters(),previous):parameter.copy_(value)
                optimizer.state.clear()
                learning_rate_factor*=.5
                rejected.append({'step':step,'finite':finite,'minimum_volume_ratio':minimum})
                (output/'rejected-steps.json').write_text(json.dumps(rejected,indent=2)+'\n')
                if len(rejected)>10:raise RuntimeError('More than10 invalid updates; preserve failure and inspect')
            else:minimum_accepted=min(minimum_accepted,minimum)
        if step==1 or step==steps or step%log_every==0:
            row={'step':step,'observed_frame_ids':np.asarray(cameras['frame_ids'])[indices].tolist(),'loss':float(loss.detach()),
                 'correspondence_count':count,'front_facing_count':front_count,
                 'learning_rate':optimizer.param_groups[0]['lr'],
                 'gradient_norm':float(gradient),'minimum_volume_ratio':minimum,
                 **{k:float(v.detach()) for k,v in components.items()}}
            history.append(row);print(json.dumps(row),flush=True)
            (output/'training.json').write_text(json.dumps(history,indent=2)+'\n')
    # Requested camera calibration is an allowed query; target pixels/states stay unopened.
    if query_cameras:
        with np.load(query_cameras) as raw:queries={k:raw[k] for k in raw.files}
        crop=project_reference_crops(ref,queries['camera_to_world_usd_row'],queries['K'],size)
        queries={**queries,'K':crop['K'],'crop_boxes_xyxy':crop['crop_boxes_xyxy']}
    else:queries=cameras
    with torch.no_grad():
        prediction=model(tensor(queries['times_s']));ratios=volume_ratios(prediction,model.rest,tetrahedra)
        final_minimum=float(ratios.min());finite=bool(torch.isfinite(prediction).all())
    if not finite:raise FloatingPointError('Nonfinite final query prediction')
    data={'positions_m':prediction.cpu().numpy(),'times_s':queries['times_s'],'K':queries['K'],
          'frame_ids':queries.get('frame_ids',np.arange(len(prediction))),
          'camera_to_world_usd_row':queries['camera_to_world_usd_row']}
    if 'crop_boxes_xyxy' in queries:data['crop_boxes_xyxy']=queries['crop_boxes_xyxy']
    np.savez_compressed(output/'predictions.npz',**data)
    torch.save({'model_state_dict':model.state_dict(),'config':asdict(config),
                'knots_s':model.knots.cpu().numpy(),'reference':ref},output/'model.pt')
    episode=Path(episode)
    files=[episode/'reference.npz',*sorted((episode/'observations').rglob('*'))]
    result={'baseline':'projective ICP + Gaussian RBF geometric trajectory; explicit RGB-D',
            'observation_mode':'rgbd','input_channels':'Observed camera-axis depth, RGB and supplied masks; known rest mesh/cameras',
            'hidden_state_used':False,'material_correspondence_supervision':False,'truth_visibility_used':False,
            'episode_id':episode.name,'config':asdict(config),'steps':steps,'seed':seed,
            'learning_rate':learning_rate,'checkpoint_selection':'fixed final step; no test score read',
            'learning_rate_schedule':'cosine decay to 1% of initial rate; invalid-step retries halve scale',
            'frame_batch_size':int(frame_batch_size),'frame_evaluations':steps*(len(eligible) if frame_batch_size<=0 else min(frame_batch_size,len(eligible))),
            'frozen_appearance_calibration':appearance,'rejected_updates':rejected,
            'minimum_accepted_knot_volume_ratio':minimum_accepted,'minimum_query_volume_ratio':final_minimum,
            'finite_query_positions':finite,'query_inversions':int((ratios<=0).sum()),
            'observed_frame_ids':cameras['frame_ids'].tolist(),'query_frame_ids':np.asarray(data['frame_ids']).tolist(),
            'seconds_elapsed':time.monotonic()-started,'torch_disable_native_jit':os.environ.get('TORCH_DISABLE_NATIVE_JIT'),
            'peak_gpu_memory_allocated_bytes':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
            'input_sha256':{str(p.relative_to(episode)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
            'source_sha256':source_hashes,
            'limitations':['depth-assisted secondary comparison, not RGB-only','projective associations are geometric, not material tracks',
                           'deterministic per-scene fit','silhouette support compares convex hulls',
                           'no material/boundary inference','no unseen-poke prediction','interpolation may invert between valid knots; query inversions are reported']}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def evaluate_motion(episode,prediction_file,output=None):
    """Separate evaluation ONLY: open held-out motion after predictions exist.

    Metrics never feed an optimizer/checkpoint selector. Geometry errors include
    all hidden/visible Gaussians, and temporal-gap scores are reported separately.
    This helper does not silently present motion errors as rendered image PSNR.
    """
    episode=Path(episode)
    with np.load(prediction_file) as raw:
        prediction=raw['positions_m'];ids=raw['frame_ids'];times=raw['times_s']
    with np.load(episode/'reference.npz') as raw:
        rest=raw['rest_points_m'];faces=raw['face_vertex_ids'];weights=raw['barycentric_weights']
        tetrahedra=raw['tetrahedra']
    with np.load(episode/'observations/cameras.npz') as raw:observed_ids=raw['frame_ids']
    with np.load(episode/'truth/trajectory.npz') as raw:
        truth=raw['positions_m'][ids];truth_times=raw['times_s'][ids]
    np.testing.assert_allclose(times,truth_times,atol=1e-7)
    gaussian=np.einsum('gi,tgij->tgj',weights,prediction[:,faces])
    target=np.einsum('gi,tgij->tgj',weights,truth[:,faces])
    static=np.einsum('gi,gij->gj',weights,rest[faces])[None]
    errors=np.linalg.norm(gaussian-target,axis=-1);static_errors=np.linalg.norm(static-target,axis=-1)
    heldout=~np.isin(ids,observed_ids);gap=heldout&(times>1.)&(times<2.)
    rows=[]
    for index,frame in enumerate(ids):
        rows.append({'frame_id':int(frame),'time_s':float(times[index]),'observed':bool(not heldout[index]),
                     'gaussian_mean_l2_error_m':float(errors[index].mean()),
                     'static_gaussian_mean_l2_error_m':float(static_errors[index].mean())})
    result={'evaluation':'held-out motion, performed after fixed-final-step RGBD fit',
            'episode_id':episode.name,'prediction_file':str(prediction_file),
            'withheld_frames':int(heldout.sum()),'temporal_gap_frames':int(gap.sum()),'rows':rows}
    for name,selection in [('withheld',heldout),('temporal_gap',gap),('observed',~heldout)]:
        if selection.any():
            value=float(errors[selection].mean());baseline=float(static_errors[selection].mean())
            result.update({name+'_gaussian_mean_l2_error_m':value,
                           name+'_static_gaussian_mean_l2_error_m':baseline,
                           name+'_relative_error_reduction':1-value/max(baseline,1e-12)})
    visibility_file=episode/'truth/visibility.npz'
    if visibility_file.exists():
        with np.load(visibility_file) as raw:
            key='visibility' if 'visibility' in raw.files else 'labels'
            if key in raw.files:
                visibility=raw[key][ids]
                if visibility.shape==errors.shape:
                    for name,label in [('visible',1),('hidden',0)]:
                        selection=(visibility==label)&heldout[:,None]
                        if selection.any():
                            result[name+'_withheld_gaussian_mean_l2_error_m']=float(errors[selection].mean())
                            result[name+'_withheld_static_gaussian_mean_l2_error_m']=float(static_errors[selection].mean())
    result['minimum_query_volume_ratio']=float(volume_ratios(torch.from_numpy(prediction),torch.from_numpy(rest).to(torch.float32),torch.from_numpy(tetrahedra)).min())
    if output:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode');parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--seed',type=int,default=2011);parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--frame-batch-size',type=int,default=0,help='0: all observed frames per update')
    parser.add_argument('--controls',type=int,default=48);parser.add_argument('--interpolation',choices=['linear','hermite'],default='hermite')
    parser.add_argument('--query-cameras');parser.add_argument('--appearance-calibration')
    parser.add_argument('--evaluate-predictions',help='Evaluation-only mode: saved prediction NPZ; never fits a model')
    args=parser.parse_args();config=GeometricConfig(controls=args.controls,interpolation=args.interpolation)
    if args.evaluate_predictions:
        result=evaluate_motion(args.episode,args.evaluate_predictions,args.output)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return
    result=fit_episode(args.episode,args.output,steps=args.steps,device=args.device,seed=args.seed,
                       learning_rate=args.learning_rate,config=config,query_cameras=args.query_cameras,
                       appearance_calibration=args.appearance_calibration,frame_batch_size=args.frame_batch_size)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
