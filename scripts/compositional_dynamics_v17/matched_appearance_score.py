"""Reference images are opened only after the fixed complete image prediction census."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import matched_appearance_render as r
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.compositional_dynamics_v17.render_metrics import compare_renderings
TRUTH=r.ROOT/'runs/compositional-dynamics-v17/calibration-development/forecast'

def prediction_gate():
    p=r.guard();commit=r.read(r.BASE/'prediction-commit.json')
    if commit['plan_sha256']!=r.sha(r.PLAN):raise ValueError('Image plan commitment changed')
    expected={(method,i,c['name'])for method in p['methods']for i in range(26)for c in p['cameras']}
    keys=[(x['method'],x['frame'],x['camera'])for x in commit['records']]
    if len(keys)!=len(expected)or set(keys)!=expected:raise ValueError('Exact260 unique predictions required before reference access')
    for row in commit['records']:
        if row['time_s']!=p['times_s'][row['frame']]or r.sha(r.ROOT/row['path'])!=row['sha256']:raise ValueError('Changed image prediction')
    return p,commit

def score():
    p,commit=prediction_gate()
    output=r.REPORT/'dynamic-results.json'
    if output.exists():raise ValueError('Preserve completed image scores')
    # Private reference access begins only below the complete prediction gate.
    status=r.read(TRUTH/'complete.json');truth_path=TRUTH/'trajectory.npz'
    if not status['completed']or status['trajectory_sha256']!=r.sha(truth_path):raise ValueError('Complete sealed reference required')
    with np.load(truth_path)as f:
        indices=np.arange(0,len(f['times_s']),10);times=f['times_s'][indices];faces=f['surface_frames'][indices]
    if len(times)!=26 or not np.allclose(times,p['times_s'],atol=1e-12,rtol=0):raise ValueError('Reference clock mismatch')
    bykey={(x['method'],x['frame'],x['camera']):x for x in commit['records']};rows=[];refs={}
    for i,t in enumerate(p['times_s']):
        for camera in p['cameras']:
            actual=render_frame(faces[i],r.fit.helper.camera(camera),time_s=t)
            actual['rgb']=np.round(actual['rgb']*255)/255
            path=r.BASE/'references'/f"{i:03d}-{camera['name']}.npz";path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**{k:v for k,v in actual.items()if k!='camera'})
            refs[str(path.relative_to(r.ROOT))]=r.sha(path)
            for method in p['methods']:
                record=bykey[method,i,camera['name']]
                with np.load(r.ROOT/record['path'])as f:prediction={k:f[k]for k in ['rgb','body_visible','body_visible_depth']}
                prediction.update(camera=actual['camera'],time_s=t)
                metrics=compare_renderings(prediction,actual)
                rows.append(dict(method=method,frame=i,time_s=t,camera=camera['name'],role=camera['role'],metrics=metrics,prediction_sha256=record['sha256'],reference_sha256=r.sha(path)))
    summary={}
    for method in p['methods']:
        summary[method]={}
        for role in ['train','held']:
            selected=[x for x in rows if x['method']==method and x['role']==role];metrics={}
            for key in ['psnr_global_db','psnr_foreground_union_db','ssim_rgb_global','ssim_rgb_foreground_union','silhouette_iou','depth_joint_rmse_m']:
                values=[x['metrics'][key]for x in selected if x['metrics'][key]is not None]
                metrics[key]=dict(mean=float(np.mean(values))if values else None,available=len(values),expected=len(selected))
            summary[method][role]=dict(rows=len(selected),metrics=metrics)
    prediction_gate()
    r.write(output,dict(scope=p['scope'],qualification='Development numerical target, not spatially qualified benchmark truth; one object and correlated views/times, no confidence-interval or population claim.',plan_sha256=r.sha(r.PLAN),prediction_commit_sha256=r.sha(r.BASE/'prediction-commit.json'),reference_trajectory_sha256=r.sha(truth_path),reference_complete_sha256=r.sha(TRUTH/'complete.json'),reference_images_sha256=refs,rows=rows,summary=summary))

if __name__=='__main__':score()
