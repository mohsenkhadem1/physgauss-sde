"""Held-out inverse prediction and reusable public-input posterior helpers.

Current evaluation permits TRAIN/VALIDATION only. Runtime encoding and sampling
never access private trajectory, labels, force metadata, or future image files.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from gaussian_observation import project_reference_crops,IsaacGaussianRenderer,image_metrics
from gaussian_observed_features import extract_features
from train_gaussian_inverse import load_inverse_checkpoint,load_frozen_operator,collect_cases,apply_catalog
from gaussian_operator import deformation_diagnostics


def load_model(inverse_checkpoint,forward_checkpoint,device='cpu'):
    model,checkpoint=load_inverse_checkpoint(inverse_checkpoint,device)
    forward=load_frozen_operator(forward_checkpoint,device)
    return {'inverse':model,'forward':forward,'checkpoint':checkpoint,'device':torch.device(device),
            'inverse_checkpoint':str(inverse_checkpoint),'forward_checkpoint':str(forward_checkpoint)}


def load_public_observations(episode,observation_mode='rgb',size=(192,160),max_time_s=None):
    """Read only selected public timestamps; a prefix never opens later images."""
    episode=Path(episode)
    with np.load(episode/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
    with np.load(episode/'observations/cameras.npz') as raw:c={k:raw[k] for k in raw.files}
    chosen=np.ones(len(c['frame_ids']),dtype=bool) if max_time_s is None else c['times_s']<=max_time_s+1e-8
    if not chosen.any():raise ValueError('Observation prefix is empty')
    native_k=c['K'];c={k:(v[chosen] if k!='K' else v) for k,v in c.items()}
    crop=project_reference_crops(reference,c['camera_to_world_usd_row'],native_k,size)
    c['native_K']=native_k;c['K']=crop['K'];c['crop_boxes_xyxy']=crop['crop_boxes_xyxy']
    rgb=[];mask=[];depth=[];native_mask=[]
    for index,frame in enumerate(c['frame_ids']):
        box=tuple(crop['crop_boxes_xyxy'][index])
        rgb.append(np.asarray(Image.open(episode/'observations/rgb'/f'{frame:05d}.png').convert('RGB').crop(box).resize(size,Image.Resampling.BILINEAR))/255.)
        image=Image.open(episode/'observations/masks'/f'{frame:05d}.png').convert('L')
        mask.append(np.asarray(image.crop(box).resize(size,Image.Resampling.BILINEAR))/255.)
        if observation_mode=='rgbd':
            with np.load(episode/'observations/depth'/f'{frame:05d}.npz') as raw:depth.append(raw['depth_m'])
            native_mask.append(np.asarray(image)>0)
    inputs={'reference':reference,'cameras':c,'native_K':native_k,'rgb':np.asarray(rgb,dtype=np.float32),
            'mask':np.asarray(mask,dtype=np.float32),'calibration_size':tuple(size),'observation_mode':observation_mode}
    if observation_mode=='rgbd':inputs['depth_native_m']=np.asarray(depth);inputs['native_mask']=np.asarray(native_mask)
    elif observation_mode!='rgb':raise ValueError('Expected rgb or rgbd mode')
    return inputs


def encode_inputs(bundle,inputs):
    """Encode public arrays, including externally appended acquired-view arrays."""
    device=bundle['device'];checkpoint=bundle['checkpoint'];config=checkpoint['model_kwargs']
    if inputs['observation_mode']!=config.get('observation_mode','rgb'):raise ValueError('Observation modality differs from trained checkpoint')
    c=inputs['cameras'];rest=inputs['reference']['rest_points_m']
    batch={'poses':torch.as_tensor(c['camera_to_world_usd_row'][None],device=device,dtype=torch.float32),
           'K':torch.as_tensor(c['K'][None],device=device,dtype=torch.float32),
           'times':torch.as_tensor(c['times_s'][None],device=device,dtype=torch.float32),
           'rest':torch.as_tensor(rest[None],device=device,dtype=torch.float32),'calibration_size':inputs['calibration_size']}
    if config.get('input_mode','cnn')=='features':
        extractor=extract_features
        if checkpoint.get('archived_feature_source_path'):
            specification=importlib.util.spec_from_file_location('archived_observed_features',checkpoint['archived_feature_source_path'])
            archived=importlib.util.module_from_spec(specification);specification.loader.exec_module(archived)
            extractor=archived.extract_features
        features=extractor(inputs);batch['features']=torch.as_tensor(features['features'][None],device=device)
        batch['images']=torch.zeros((1,len(c['times_s']),1,1,1),device=device)
    else:
        images=np.concatenate((inputs['rgb'],inputs['mask'][...,None]),-1).transpose(0,3,1,2)
        batch['images']=torch.as_tensor(images[None],device=device)
    with torch.no_grad():context=bundle['inverse'].context(batch)
    return {'context':context,'batch':batch,'inputs':inputs}


def encode_episode(bundle,episode,max_time_s=None):
    config=bundle['checkpoint'];mode=config.get('observation_mode',config['model_kwargs'].get('observation_mode','rgb'))
    inputs=load_public_observations(episode,mode,tuple(config['size']),max_time_s)
    return encode_inputs(bundle,inputs)


def posterior_samples(bundle,encoded,samples=64,seed=20261102):
    generator=torch.Generator(device=bundle['device']).manual_seed(seed)
    with torch.no_grad():draw=bundle['inverse'].posterior.sample(encoded['context'],samples=samples,generator=generator)
    return {key:value[0] for key,value in draw.items()}


def predict_particles(bundle,particles,reference,times,chunk_size=8):
    """Forward positions[K,T,N,3] from physical hypotheses, without FEM."""
    device=bundle['device'];rest=torch.as_tensor(reference['rest_points_m'],device=device,dtype=torch.float32)
    times=torch.as_tensor(times,device=device,dtype=torch.float32);predictions=[]
    with torch.no_grad():
        for start in range(0,len(particles['physical']),chunk_size):
            stop=start+chunk_size
            predictions.append(bundle['forward'](rest,times,particles['effective_physical'][start:stop],
                                                particles['boundary_type'][start:stop],particles['force_type'][start:stop]))
    return torch.cat(predictions)


def evaluate(roots,inverse_checkpoint,forward_checkpoint,output,catalog=None,samples=64,device='cpu',
             max_time_s=None,render_frames=(),appearance=None,split='validation'):
    if split not in ('train','validation'):raise ValueError('Test groups remain locked until final protocol')
    torch.set_num_threads(2);bundle=load_model(inverse_checkpoint,forward_checkpoint,device)
    records=collect_cases(roots)
    if catalog:records=apply_catalog(records,catalog)
    records=[r for r in records if r['recipe']['split']==split]
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
    for index,record in enumerate(records):
        episode=record['path'];encoded=encode_episode(bundle,episode,max_time_s)
        particles=posterior_samples(bundle,encoded,samples,seed=20261102+index)
        # Query camera/time calibration is provided to any predictor. Private
        # image pixels and nodal trajectories are opened after prediction only.
        with np.load(episode/'truth/cameras.npz') as raw:c={k:raw[k] for k in raw.files}
        predicted=predict_particles(bundle,particles,encoded['inputs']['reference'],c['times_s']).cpu().numpy()
        mean=predicted.mean(0);lower,upper=np.quantile(predicted,[.05,.95],axis=0)
        np.savez_compressed(output/(record['recipe']['episode_id']+'-prediction.npz'),mean_positions_m=mean,
                            lower90_m=lower,upper90_m=upper,physical=particles['physical'].cpu().numpy(),
                            boundary_type=particles['boundary_type'].cpu().numpy(),force_type=particles['force_type'].cpu().numpy())
        with np.load(episode/'truth/trajectory.npz') as raw:target=raw['positions_m']
        rest=encoded['inputs']['reference']['rest_points_m'];gap=(c['times_s']>1)&(c['times_s']<2)
        # Initialstate is deterministic; exclude it from interval coverage.
        row={'key':record['key'],'episode_id':record['recipe']['episode_id'],
             'coordinate_rmse_m':float(np.sqrt(np.mean((mean-target)**2))),
             'static_coordinate_rmse_m':float(np.sqrt(np.mean((rest-target)**2))),
             'gap_coordinate_rmse_m':float(np.sqrt(np.mean((mean[gap]-target[gap])**2))),
             'pointwise_coordinate90_coverage':float(((target[1:]>=lower[1:])&(target[1:]<=upper[1:])).mean()),
             'pointwise_coordinate90_mean_width_m':float((upper[1:]-lower[1:]).mean()),
             'particles':samples,'observed_frames':encoded['inputs']['cameras']['frame_ids'].tolist()}
        ref=encoded['inputs']['reference']
        tets=ref['tetrahedra'];audit_frames=np.array([15,30,45,60]);audit=predicted[:,audit_frames]
        determinant=np.linalg.det(audit[:,:,tets[:,1:]]-audit[:,:,tets[:,:1]])
        reference_det=np.linalg.det(rest[tets[:,1:]]-rest[tets[:,:1]])
        ratios=determinant/reference_det
        positive=(ratios>0).all((1,2))
        row['particle_positive_fraction_at_4_audit_frames']=float(positive.mean())
        row['particle_minimum_volume_ratio_at_4_audit_frames']=float(ratios.min())
        checks=deformation_diagnostics(torch.tensor(mean),torch.tensor(rest,dtype=torch.float32),torch.tensor(ref['tetrahedra'],dtype=torch.long))
        row.update({'mean_trajectory_'+k:v for k,v in checks.items()})
        image_scores=[]
        if render_frames:
            size=(128,96);crop=project_reference_crops(ref,c['camera_to_world_usd_row'],c['K'],size)
            renderer=IsaacGaussianRenderer(ref,size,device,calibration_size=size,**({k:appearance[k] for k in ['color_matrix','color_bias']} if appearance else {}))
            bg=np.median(encoded['inputs']['rgb'][0][encoded['inputs']['mask'][0]<.01],axis=0)
            for frame in render_frames:
                if frame in encoded['inputs']['cameras']['frame_ids']:raise ValueError('Appearance score requires withheld frames')
                with torch.no_grad():rendered=renderer(mean[[frame]],c['camera_to_world_usd_row'][[frame]],crop['K'][[frame]],bg)
                rendered={k:v[0].cpu().numpy() for k,v in rendered.items()}
                box=tuple(crop['crop_boxes_xyxy'][frame]);rgb=np.asarray(Image.open(episode/'truth/rgb'/f'{frame:05d}.png').crop(box).resize(size,Image.Resampling.BILINEAR))/255.
                mask=np.asarray(Image.open(episode/'truth/masks'/f'{frame:05d}.png').crop(box).resize(size,Image.Resampling.BILINEAR))/255.
                image_scores.append(image_metrics(rendered,rgb,mask))
        if image_scores:row['appearance']={k:float(np.mean([s[k] for s in image_scores])) for k in image_scores[0]}
        rows.append(row);print(json.dumps(row),flush=True)
    aggregate={key:float(np.mean([r[key] for r in rows])) for key in ['coordinate_rmse_m','static_coordinate_rmse_m','gap_coordinate_rmse_m','pointwise_coordinate90_coverage','pointwise_coordinate90_mean_width_m']}
    result={'episodes':rows,'aggregate':aggregate,'inverse_checkpoint':str(inverse_checkpoint),'forward_checkpoint':str(forward_checkpoint),
            'checkpoint_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('inverse',inverse_checkpoint),('forward',forward_checkpoint)]},
            'max_observed_time_s':max_time_s,'observation_mode':bundle['checkpoint'].get('observation_mode','rgb'),
            'qualification':'Finite-sample posterior predictive intervals are uncalibrated; pointwise coordinates are correlated; one known cuboid geometry; TRAIN/VAL only'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--inverse-checkpoint',required=True);parser.add_argument('--forward-checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--catalog');parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--device',default='cpu');parser.add_argument('--max-time-s',type=float)
    parser.add_argument('--render-frames',type=int,nargs='*',default=[]);parser.add_argument('--appearance')
    parser.add_argument('--split',choices=['train','validation'],default='validation');args=parser.parse_args()
    if args.appearance:args.appearance=json.loads(Path(args.appearance).read_text())
    result=evaluate(**vars(args));print(json.dumps(result['aggregate'],indent=2))
