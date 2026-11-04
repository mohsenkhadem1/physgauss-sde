"""Virtual-camera projections of existing manual 3D landmarks, not RGB accuracy."""
import json,pickle
import numpy as np
from scipy.spatial import cKDTree
from .campaign import ROOT,LABEL,CASES,write
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction


def project(points,pose,K):
    camera=(points-pose[:3,3])@pose[:3,:3]
    if not (camera[:,2]>0).all():raise ValueError('Landmark behind virtual camera')
    p=camera@K.T
    return p[:,:2]/p[:,2,None]


def main():
    base=ROOT/'runs/real-transfer-v22/novel-views';rows=[]
    keys=CASES+['single_lift_sloth--single_push_sloth','single_lift_zebra--double_lift_zebra']
    for key in keys:
        manifest=json.loads((base/key/'manifest.json').read_text());case=manifest['case'];source=manifest['source']
        data=phystwin(source or case,240);flip=data['evaluation']['transform'];initial=data['support_geometry']@flip
        if source:
            reg=next(r for r in json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())['pairs'] if r['source']==source and r['target']==case)
            initial=initial@(flip@np.asarray(reg['rotation'])@flip)+np.asarray(reg['translation_m'])@flip
            with np.load(ROOT/'runs/real-transfer-v20/frozen-transfer/v22-corrected-frozen'/key/'neural/prediction.npz') as f:ours=f['predicted']@flip
            with np.load(ROOT/'runs/real-transfer-v20/native-transfer'/key/'prediction.npz') as f:native=f['predicted']@flip
        else:
            with np.load(ROOT/'runs/real-transfer-v19/fits'/LABEL/case/'neural/prediction.npz') as f:ours=expand_prediction(data,f['predicted'])@flip
            with (ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl').open('rb') as f:native=np.asarray(pickle.load(f))[:,:len(initial)]
        with (ROOT/'runs/third_party/PhysTwin/data/different_types'/case/'gt_track_3d.pkl').open('rb') as f:manual=np.asarray(pickle.load(f))
        frame=manifest['frame'];valid=np.isfinite(manual[0]).all(1)&np.isfinite(manual[frame]).all(1)
        if not valid.any():
            rows.append(dict(key=key,source=source,case=case,frame=frame,landmarks=0,status='no_annotation',scores=None));continue
        ids=cKDTree(initial).query(manual[0,valid])[1];truth=manual[frame,valid]
        K=np.asarray(manifest['camera_intrinsics']);cameras=np.asarray(manifest['camera_to_world']);scores={}
        for method,pred in [('neural',ours),('phystwin',native)]:
            errors=[float(np.linalg.norm(project(pred[frame,ids],C,K)-project(truth,C,K),axis=1).mean()) for C in cameras]
            scores[method]=dict(mean_pixels=float(np.mean(errors)),min_pixels=min(errors),max_pixels=max(errors),per_angle_pixels=errors)
        rows.append(dict(key=key,source=source,case=case,frame=frame,landmarks=int(valid.sum()),scores=scores))
    write(ROOT/'reports/real-transfer-v22/projected-landmarks.json',dict(rows=rows,
        scope='Final-state manual 3D landmarks projected into 120 virtual 512x512 cameras. Same canonical initial correspondence for both methods. No occlusion filtering: hidden landmarks still contribute. Reuses existing 3D evidence, not an independent novel-view RGB test.',
        summary={group:dict(cases=sum(bool(r['source'])==transfer and r['scores'] is not None for r in rows),**{m:float(np.mean([r['scores'][m]['mean_pixels'] for r in rows if bool(r['source'])==transfer and r['scores'] is not None])) for m in ('neural','phystwin')}) for group,transfer in [('standard',False),('frozen_transfer',True)]}))
    for r in rows:print(r['key'],None if r['scores'] is None else {m:r['scores'][m]['mean_pixels'] for m in r['scores']})

if __name__=='__main__':main()
