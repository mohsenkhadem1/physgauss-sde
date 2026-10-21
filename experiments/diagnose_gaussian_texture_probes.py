"""Validation-only probe RGB observability; this is not active-policy scoring.

All four candidate outcomes are deliberately exposed for this method diagnostic.
Withheld evaluation-control outcomes, depth and hidden trajectories are unread.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from gaussian_texture_features import frame_texture_features


def diagnose(root,passive_root,calibration_path,output):
    root=Path(root);passive_root=Path(passive_root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a fresh immutable diagnostic path')
    output.mkdir(parents=True);calibration=json.loads(Path(calibration_path).read_text());rows=[]
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    for public in sorted((root/'public').glob('*/action-menu.json')):
        episode=public.parent.name;menu=json.loads(public.read_text())
        with np.load(passive_root/'episodes'/episode/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
        actions=['reset_replay']+[r['action_id'] for r in menu['candidates']]
        records={}
        for action in actions:
            source=root/'outcomes'/episode/action/'observations'
            with np.load(source/'cameras.npz') as raw:c={k:raw[k] for k in raw.files}
            features=[];rgbs=[];masks=[]
            for index,frame in enumerate(c['frame_ids']):
                rgb=np.asarray(Image.open(source/'rgb'/f'{frame:05d}.png').convert('RGB'))/255.
                mask=np.asarray(Image.open(source/'masks'/f'{frame:05d}.png').convert('L'))>0
                feature,_=frame_texture_features(reference,c['camera_to_world_usd_row'][index],c['K'],rgb,mask,calibration)
                features.append(feature);rgbs.append(rgb);masks.append(mask)
            records[action]={'features':np.asarray(features),'rgb':np.asarray(rgbs),'mask':np.asarray(masks),'times':c['times_s']}
        baseline=records['reset_replay']
        np.savez_compressed(output/f'{episode}-features.npz',**{k:v['features'] for k,v in records.items()})
        for action in actions[1:]:
            record=records[action];delta=(record['features']-baseline['features']).reshape(-1,6,13)
            shifts=np.linalg.norm(delta[...,1:3]*10,axis=-1) # feature/.01m -> mm
            valid=(record['features'].reshape(-1,6,13)[...,0]>0)&(baseline['features'].reshape(-1,6,13)[...,0]>0)
            rgb_delta=abs(record['rgb']-baseline['rgb']);mask=record['mask']&baseline['mask']
            rows.append({'episode':episode,'action':action,
                         'max_apparent_shift_difference_mm':float(shifts[valid].max(initial=0)),
                         'reference_apparent_shift_difference_mm':float(shifts[0][valid[0]].max(initial=0)),
                         'later_median_apparent_shift_difference_mm':float(np.median(shifts[1:][valid[1:]])),
                         'maximum_rgb_difference_255':float(rgb_delta[mask].max(initial=0)*255),
                         'foreground_rgb_difference_rmse_255':float(np.sqrt(np.mean(rgb_delta[mask]**2))*255),
                         'reference_rgb_difference_rmse_255':float(np.sqrt(np.mean(rgb_delta[0][mask[0]]**2))*255),
                         'later_rgb_difference_rmse_255':float(np.sqrt(np.mean(rgb_delta[1:][mask[1:]]**2))*255),
                         'per_face_max_shift_mm':[float(shifts[:,f][valid[:,f]].max(initial=0)) for f in range(6)]})
    result={'rows':rows,'calibration_sha256':hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
            'qualification':'VALIDATION candidate exposure for feature observability only. Values compare reset replay images, not ground-truth motion; they cannot score or select an active policy. Withheld evaluation control, depth and trajectories not read.'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--passive-root',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(diagnose(a.root,a.passive_root,a.calibration,a.output),indent=2))
