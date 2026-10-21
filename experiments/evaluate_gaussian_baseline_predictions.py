"""Matched RGB evaluation of already saved model predictions, never fitting.

Use the same known-rest ROI, fixed TRAIN appearance correction, renderer,
background and target sensor images for geometric/ridge/flow/PIDG predictions.
Kept orbit frames mix time and viewpoint; an optional actual acquisition pool
also supplies a genuinely different camera at exactly the same physical time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image


def read_prediction(path,node_count,query_times):
    """Read a completed point prediction, accepting common repository schemas."""
    with np.load(path) as raw:
        key='positions_m' if 'positions_m' in raw.files else 'mean_positions_m'
        if key not in raw.files:raise ValueError('Prediction needs positions_m or mean_positions_m')
        q=raw[key]
        if 'times_s' in raw.files:np.testing.assert_allclose(raw['times_s'],query_times,atol=1e-7)
        if 'frame_ids' in raw.files:np.testing.assert_array_equal(raw['frame_ids'],np.arange(len(query_times)))
    if q.shape!=(len(query_times),node_count,3) or not np.isfinite(q).all():
        raise ValueError('Expected finite dense query positions [frames,reference_nodes,3]')
    return q


def evaluate_case(episode,predictions,output,*,pool=None,frames=(25,30,35),device='cuda:3',
                  size=(192,160),appearance=None):
    import torch
    started=time.monotonic()
    try:
        from gaussian_observation import IsaacGaussianRenderer,load_rgb_inputs,project_reference_crops,image_metrics
        from isaac_acquisition_views import load_public_view_queries
    except ImportError:
        from experiments.gaussian_observation import IsaacGaussianRenderer,load_rgb_inputs,project_reference_crops,image_metrics
        from experiments.isaac_acquisition_views import load_public_view_queries
    root=Path(output);root.mkdir(parents=True,exist_ok=False);episode=Path(episode)
    inputs=load_rgb_inputs(episode,size,roi=True);ref=inputs['reference'];rest=ref['rest_points_m']
    appearance_data=json.loads(Path(appearance).read_text()) if appearance else {}
    # Camera metadata is a query. Hidden image/trajectory values are read only
    # after all fixed predictions have been loaded and checked below.
    with np.load(episode/'truth/cameras.npz') as raw:cameras={k:raw[k] for k in raw.files}
    query_times=cameras['times_s'];hashes={};positions={}
    for name,path in predictions.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Unsafe model identifier')
        positions[name]=read_prediction(path,len(rest),query_times)
        hashes[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    frames=tuple(map(int,frames))
    if any(f in inputs['cameras']['frame_ids'] or f<0 or f>=len(query_times) for f in frames):
        raise ValueError('Appearance targets must be withheld valid query frames')
    crop=project_reference_crops(ref,cameras['camera_to_world_usd_row'],cameras['K'],size)
    queries=[]
    for frame in frames:
        queries.append({'name':f'orbit_{frame:05d}','kind':'withheld_orbit',
                        'time_s':float(query_times[frame]),'frame':frame,
                        'pose':cameras['camera_to_world_usd_row'][frame],
                        'K':crop['K'][frame],'box':crop['crop_boxes_xyxy'][frame],
                        'rgb':episode/'truth/rgb'/f'{frame:05d}.png',
                        'mask':episode/'truth/masks'/f'{frame:05d}.png'})
    if pool:
        public=load_public_view_queries(pool,episode.name)
        for index in np.flatnonzero(public['roles']=='evaluation'):
            time_s=float(public['times_s'][index]);found=np.flatnonzero(np.isclose(query_times,time_s,atol=1e-7))
            if len(found)!=1:raise ValueError('Same-state evaluation needs an exact saved prediction time')
            pose=public['camera_to_world_usd_row'][index]
            view_crop=project_reference_crops(ref,pose[None],public['K'],size)
            folder=Path(pool)/'truth'/episode.name/'views'/str(public['view_ids'][index])
            queries.append({'name':str(public['view_ids'][index]),'kind':'independent_same_state_view',
                            'time_s':time_s,'frame':int(found[0]),'pose':pose,
                            'K':view_crop['K'][0],'box':view_crop['crop_boxes_xyxy'][0],
                            'rgb':folder/'rgb.png','mask':folder/'mask.png'})
    background=np.median(inputs['rgb'][0][inputs['mask'][0]<.01],axis=0)
    renderer=IsaacGaussianRenderer(ref,size=size,device=device,calibration_size=size,
                                    **{k:appearance_data[k] for k in ['color_matrix','color_bias'] if k in appearance_data})
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('gaussian_observation.py')]}
    report={'episode_id':episode.name,'prediction_sha256':hashes,'source_sha256':source_hashes,
            'resolution':list(size),'roi':'known rest projection +30 native pixel margin',
            'appearance_calibration':appearance_data,'fitting_performed':False,
            'qualification':'Gaussian/Isaac shader discrepancy remains; same metric/rendering for all models; orbit and same-state new views reported separately',
            'models':{}}
    for name,q in positions.items():
        model_folder=root/name;model_folder.mkdir();scores=[]
        for query in queries:
            with torch.no_grad():
                rendered=renderer(q[[query['frame']]],query['pose'][None],query['K'][None],background)
                static=renderer(rest[None],query['pose'][None],query['K'][None],background)
            rendered={k:v[0].cpu().numpy() for k,v in rendered.items()}
            static={k:v[0].cpu().numpy() for k,v in static.items()}
            box=tuple(map(int,query['box']))
            target=np.asarray(Image.open(query['rgb']).convert('RGB').crop(box).resize(size,Image.Resampling.BILINEAR))/255.
            mask=np.asarray(Image.open(query['mask']).convert('L').crop(box).resize(size,Image.Resampling.BILINEAR))/255.
            record={k:query[k] for k in ['name','kind','time_s','frame']}
            record['prediction']=image_metrics(rendered,target,mask);record['static']=image_metrics(static,target,mask)
            scores.append(record)
            # Actual sensor and renderer outputs; no motion or appearance edits.
            triptych=np.concatenate((static['rgb'],rendered['rgb'],target),1)
            Image.fromarray(np.rint(np.clip(triptych,0,1)*255).astype(np.uint8)).save(model_folder/(query['name']+'-static_prediction_target.png'))
        aggregate={}
        for kind in sorted({s['kind'] for s in scores}):
            selected=[s for s in scores if s['kind']==kind]
            aggregate[kind]={method:{key:float(np.mean([s[method][key] for s in selected])) for key in selected[0][method]} for method in ['prediction','static']}
        # Geometry diagnostics/score are separate from image fitting and metrics.
        with np.load(episode/'truth/trajectory.npz') as raw:target=raw['positions_m']
        tetra=ref['tetrahedra'];reference_det=np.linalg.det(rest[tetra[:,1:]]-rest[tetra[:,:1]])
        ratio=np.linalg.det(q[:,tetra[:,1:]]-q[:,tetra[:,:1]])/reference_det
        report['models'][name]={'queries':scores,'aggregate':aggregate,
                                'coordinate_rmse_m':float(np.sqrt(np.mean((q-target)**2))),
                                'static_coordinate_rmse_m':float(np.sqrt(np.mean((rest-target)**2))),
                                'minimum_volume_ratio':float(ratio.min()),'inverted_tetrahedron_frames':int((ratio<=0).sum())}
        if hashlib.sha256(Path(predictions[name]).read_bytes()).hexdigest()!=hashes[name]:
            raise RuntimeError('Prediction file changed during evaluation')
    inputs_to_hash=[episode/'reference.npz',episode/'observations/cameras.npz',
                    episode/'truth/cameras.npz',episode/'truth/trajectory.npz']
    inputs_to_hash+=list((episode/'observations/rgb').glob('*.png'))
    inputs_to_hash+=list((episode/'observations/masks').glob('*.png'))
    inputs_to_hash+=[query[k] for query in queries for k in ['rgb','mask']]
    if appearance:inputs_to_hash.append(Path(appearance))
    if pool:inputs_to_hash.append(Path(pool)/'public'/episode.name/'cameras.npz')
    report['evaluation_input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs_to_hash)) if p.is_file()}
    report['seconds_elapsed']=time.monotonic()-started
    report['device']=device
    provenance=root/'provenance';provenance.mkdir()
    for name in source_hashes:(provenance/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True);parser.add_argument('--predictions',nargs='+',required=True,metavar='NAME=NPZ')
    parser.add_argument('--output',required=True);parser.add_argument('--pool');parser.add_argument('--appearance')
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--frames',type=int,nargs='+',default=[25,30,35])
    args=parser.parse_args();predictions=dict(item.split('=',1) for item in args.predictions)
    report=evaluate_case(args.episode,predictions,args.output,pool=args.pool,appearance=args.appearance,device=args.device,frames=args.frames)
    print(json.dumps({name:{'coordinate_rmse_m':row['coordinate_rmse_m'],'aggregate':row['aggregate']} for name,row in report['models'].items()},indent=2))


if __name__=='__main__':main()
