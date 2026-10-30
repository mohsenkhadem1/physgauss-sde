"""Descriptive full-reference motion audit; no selection or predictor changes."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/coupled-contact-v13';REPORT=ROOT/'reports/coupled-contact-v13'

def motion(frames):
    x=np.asarray(frames,float)
    if x.ndim!=3 or x.shape[-1]!=3 or len(x)<2 or not np.isfinite(x).all():return dict(complete=False,reason='Invalid/nonfinite full reference trajectory')
    rest=x[0];a=rest-rest.mean(0);residual=[];angles=[]
    for frame in x[1:]:
        b=frame-frame.mean(0);u,_,v=np.linalg.svd(a.T@b)
        correction=np.diag([1.,1.,np.linalg.det(u@v)]);rotation=u@correction@v
        residual.append(float(np.mean((a@rotation-b)**2)))
        angles.append(float(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
    world=float(np.sqrt(np.mean((x[1:]-rest)**2)));shape=float(np.sqrt(np.mean(residual)))
    return dict(complete=True,frames_excluding_initial=len(x)-1,nodes=x.shape[1],world_coordinate_motion_rms_m=world,
        nonrigid_coordinate_motion_rms_m=shape,nonrigid_to_world_rms_ratio=shape/world if world>0 else None,
        max_frame_nonrigid_coordinate_rms_m=float(np.sqrt(max(residual))),
        com_coordinate_motion_rms_m=float(np.sqrt(np.mean((x[1:].mean(1)-rest.mean(0))**2))),
        max_rigid_rotation_degrees=float(np.degrees(max(angles))))

def summarize(rows):
    valid=[r for r in rows if r['motion']['complete']];out=dict(declared=len(rows),complete=len(valid),failed=[dict(object=r['object'],scenario=r['scenario'],reason=r['motion']['reason']) for r in rows if not r['motion']['complete']])
    for key in ['world_coordinate_motion_rms_m','nonrigid_coordinate_motion_rms_m','nonrigid_to_world_rms_ratio','max_frame_nonrigid_coordinate_rms_m','com_coordinate_motion_rms_m','max_rigid_rotation_degrees']:
        values=[r['motion'][key] for r in valid if r['motion'][key] is not None]
        out[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),minimum=float(min(values)),maximum=float(max(values))) if values else None
    return out

def main():
    p=json.loads((REPORT/'protocol.json').read_text());rows=[]
    for obj in p['objects']:
        for scenario in ['squeeze']+p['forecast_scenarios']:
            path=BASE/'reference'/obj['id']/scenario;status=json.loads((path/'complete.json').read_text())
            row=dict(object=obj['id'],split=obj['split'],scenario=scenario,role='calibration' if scenario=='squeeze' else 'forecast')
            if status['success']:
                with np.load(path/'trajectory.npz') as a:row['motion']=motion(a['frames'])
                row['trajectory_sha256']=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest()
            else:row['motion']=dict(complete=False,reason=status.get('error','Reference generation failed'))
            rows.append(row)
    output=dict(scope='Descriptive audit of original vertex-contact references only; no new surface outcomes read; no model selection/tuning.',
        weighting='Uniform fine-mesh node and coordinate weighting; every post-initial frame included. Best proper rigid transform fitted independently each frame from initial coordinates to current frame. Ratio is RMS nonrigid residual / RMS total displacement, not a variance fraction. Summary means/medians weight cases equally; subgroup ratios are not pooled frame ratios.',
        expected_cases=70,expected_calibration=14,expected_forecast=56,cases=rows,all_cases=summarize(rows),
        by_scenario={s:summarize([r for r in rows if r['scenario']==s]) for s in ['squeeze']+p['forecast_scenarios']},
        by_split={s:summarize([r for r in rows if r['split']==s]) for s in ['development','validation','reserved']},
        by_split_scenario={s:{scene:summarize([r for r in rows if r['split']==s and r['scenario']==scene]) for scene in ['squeeze']+p['forecast_scenarios']} for s in ['development','validation','reserved']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'motion-population-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['ORIGINAL v13 POPULATION MOTION AUDIT',output['weighting'],'','All70declared reference trajectories retained (14calibration,56forecast). No new surface outcomes inspected.','Scenario | complete | median world RMS mm | median nonrigid RMS mm | median nonrigid/world % | ratio range % | median maximum shape RMS mm']
    for scene,s in output['by_scenario'].items():
        w,n,r,m=(s[k] for k in ['world_coordinate_motion_rms_m','nonrigid_coordinate_motion_rms_m','nonrigid_to_world_rms_ratio','max_frame_nonrigid_coordinate_rms_m'])
        lines.append(f"{scene} | {s['complete']}/{s['declared']} | {w['median']*1000:.3f} | {n['median']*1000:.3f} | {r['median']*100:.2f} | {r['minimum']*100:.2f}–{r['maximum']*100:.2f} | {m['median']*1000:.3f}")
    lines.extend(['','This measures deformation present in the discrete reference trajectories. It does not establish material identifiability, reference spatial convergence, real-world validity or a predictor advantage. The edge-reference spatial sampling limitation remains. Full per-case and per-split values, failures and source hashes are in the companion JSON.'])
    (REPORT/'motion-population-audit.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
