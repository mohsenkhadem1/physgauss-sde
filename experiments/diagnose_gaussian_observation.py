"""TRAIN/VAL-only oracle geometry diagnostic; never use this as model inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gaussian_observation import (load_rgb_inputs,IsaacGaussianRenderer,image_metrics,
                                  project_reference_crops,fit_train_appearance)


def diagnose(root,output,device='cuda:3',size=(192,160),frames=(0,15,25,30,35,45,60),
             episodes=None,roi=True,appearance=None):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());cases=[c for c in plan['episodes'] if c['split']!='test']
    if episodes:
        byid={c['episode_id']:c for c in plan['episodes']}
        if any(byid[e]['split']=='test' for e in episodes):raise ValueError('Oracle diagnostic forbids test groups')
        cases=[c for c in cases if c['episode_id'] in episodes]
    rows=[];t0=time.perf_counter()
    for case in cases:
        eid=case['episode_id'];p=root/'episodes'/eid
        inputs=load_rgb_inputs(p,size,roi=roi);ref=inputs['reference']
        kwargs={'size':size,'device':device,'chunk_size':256,'calibration_size':size if roi else (640,480)}
        rawrenderer=IsaacGaussianRenderer(ref,**kwargs)
        calibrated=None
        if appearance:
            calibrated=IsaacGaussianRenderer(ref,**kwargs,color_matrix=appearance['color_matrix'],color_bias=appearance['color_bias'])
        bg=np.median(inputs['rgb'][0][inputs['mask'][0]<.01],axis=0)
        with np.load(p/'truth/cameras.npz') as f:c={k:f[k] for k in f.files}
        crop=project_reference_crops(ref,c['camera_to_world_usd_row'],c['K'],size) if roi else None
        with np.load(p/'truth/trajectory.npz') as f:q=f['positions_m']
        panel=[]
        for frame in frames:
            image=Image.open(p/'truth/rgb'/f'{frame:05d}.png');maskimage=Image.open(p/'truth/masks'/f'{frame:05d}.png')
            if roi:
                box=tuple(crop['crop_boxes_xyxy'][frame]);image=image.crop(box);maskimage=maskimage.crop(box)
            target=np.asarray(image.resize(size,Image.Resampling.BILINEAR))/255.
            mask=np.asarray(maskimage.resize(size,Image.Resampling.BILINEAR))/255.
            metrics={};line=[target]
            renderers=[('raw',rawrenderer)]
            if calibrated is not None:renderers.append(('calibrated',calibrated))
            for suffix,renderer in renderers:
                for name,pos in [('static',ref['rest_points_m']),('oracle',q[frame])]:
                    k=crop['K'][[frame]] if roi else c['K']
                    with torch.no_grad():pred=renderer(pos[None],c['camera_to_world_usd_row'][[frame]],k,bg)
                    pred={key:v[0].cpu().numpy() for key,v in pred.items()}
                    metrics[f'{name}_{suffix}']=image_metrics(pred,target,mask);line.append(pred['rgb'])
            rows.append({'episode_id':eid,'split':case['split'],'frame':frame,'force_type':case['force_type'],'boundary_type':case['boundary_type'],**metrics})
            panel.append(np.concatenate(line,1))
        Image.fromarray(np.clip(np.concatenate(panel,0)*255,0,255).astype(np.uint8)).save(output/f'{eid}.png')
        print(json.dumps({'episode':eid,'seconds':time.perf_counter()-t0}),flush=True)
    result={'purpose':'Oracle geometry diagnostic only; no geometry truth supplied to model',
            'frames':rows,'seconds':time.perf_counter()-t0,'resolution':list(size),
            'plan_sha256':plan.get('plan_sha256'),
            'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['diagnose_gaussian_observation.py','gaussian_observation.py']},
            'versions':{'numpy':np.__version__,'torch':torch.__version__},
            'roi':roi,'crop_rule':'Known rest projection plus 30 native pixels' if roi else None,'appearance':appearance,
            'qualification':'Training and validation groups only; shader mismatch and antialiasing remain'}
    for name in [k for k in rows[0] if k.startswith(('static_','oracle_'))]:
        result[name]={k:float(np.mean([r[name][k] for r in rows])) for k in rows[0][name]}
    if device.startswith('cuda'):result['peak_cuda_memory_MB']=torch.cuda.max_memory_allocated(torch.device(device))/1e6
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='runs/isaac-boundary-orbit-v01')
    parser.add_argument('--output',default='runs/gaussian-observation-roi');parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--episodes',nargs='*');parser.add_argument('--full-frame',action='store_true')
    parser.add_argument('--fit-appearance',action='store_true');args=parser.parse_args()
    size=(128,96) if args.full_frame else (192,160);appearance=None
    if args.fit_appearance:appearance=fit_train_appearance(args.root,Path(args.output)/'train-appearance.json',args.device,size)
    result=diagnose(args.root,args.output,args.device,size,episodes=args.episodes,roi=not args.full_frame,appearance=appearance)
    print(json.dumps({k:v for k,v in result.items() if k not in ['frames','appearance']},indent=2))
