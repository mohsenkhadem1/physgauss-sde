"""Secondary cohort image scores; private references only after image seals."""
import numpy as np
from scripts.compositional_dynamics_v17 import cohort_appearance as r
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.compositional_dynamics_v17.render_metrics import compare_renderings

def score():
    p,commit=r.prediction_gate()
    output=r.ROOT/'reports/compositional-dynamics-v17/fresh-cohort-appearance-results.json'
    if output.exists():raise ValueError('Preserve previous scores')
    generation_path=r.ROOT/'reports/compositional-dynamics-v17/fresh-cohort-generation.json'
    generation=r.read(generation_path)
    if not generation['terminal']or generation['protocol_sha256']!=p['bindings']['protocol_sha']or generation['method_freeze_sha256']!=p['bindings']['freeze_sha']:raise ValueError('Terminal reference generation required')
    evidence={str(generation_path.relative_to(r.ROOT)):r.sha(generation_path)};rows=[]
    for q in p['queries']:
        subset=[x for x in commit['rows']if x['world_id']==q['world_id']and x['query']==q['role']]
        folder=r.ROOT/'runs/compositional-dynamics-v17/fresh-cohort/references'/q['world_id']/q['role']/'n4-dt0.00002500'
        status_path=folder/'complete.json';available=False;reason='Missing reference at terminal census'
        if status_path.exists():
            status=r.read(status_path);evidence[str(status_path.relative_to(r.ROOT))]=r.sha(status_path)
            if status['world_id']!=q['world_id']or status['role']!=q['role']or status['protocol_sha256']!=p['bindings']['protocol_sha']or status['method_freeze_sha256']!=p['bindings']['freeze_sha']or status['n']!=4 or status['dt_s']!=25e-6:raise ValueError('Reference identity changed')
            available=status['completed'];reason=status.get('error','Reference incomplete')
        if not available:
            rows.extend(dict(**x,reference_complete=False,scored=False,reference_error=reason)for x in subset);continue
        path=folder/'trajectory.npz'
        if r.sha(path)!=status['trajectory_sha256']:raise ValueError('Reference array changed')
        evidence[str(path.relative_to(r.ROOT))]=r.sha(path)
        with np.load(path)as f:
            ids=np.arange(0,len(f['times_s']),10);times=f['times_s'][ids];faces=f['surface_frames'][ids]
        if len(times)!=31 or not np.allclose(times,q['times_s'],atol=1e-12,rtol=0):raise ValueError('Reference clock mismatch')
        bykey={(x['method'],x['frame'],x['camera']):x for x in subset}
        for i,t in enumerate(times):
            for camera in q['cameras']:
                actual=render_frame(faces[i],r.fit.helper.camera(camera),time_s=t);actual['rgb']=np.round(actual['rgb']*255)/255
                dest=r.BASE/'references'/q['world_id']/q['role']/f"{i:03d}-{camera['name']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**{k:v for k,v in actual.items()if k!='camera'});evidence[str(dest.relative_to(r.ROOT))]=r.sha(dest)
                for method in ['native','shared_q2']:
                    row=dict(bykey[method,i,camera['name']],reference_complete=True,scored=False,reference_sha256=r.sha(dest))
                    if row['completed']:
                        with np.load(r.ROOT/row['prediction']['path'])as f:pred={k:f[k]for k in ['rgb','body_visible','body_visible_depth']}
                        pred.update(camera=actual['camera'],time_s=t);row.update(scored=True,metrics=compare_renderings(pred,actual))
                    rows.append(row)
    fields=['psnr_global_db','psnr_foreground_union_db','ssim_rgb_global','ssim_rgb_foreground_union','silhouette_iou','depth_joint_rmse_m']
    # First average correlated frames/views within each world; worlds remain explicit.
    summaries=[]
    for world in sorted({q['world_id']for q in p['queries']}):
        for method in ['native','shared_q2']:
            for role in ['train','held']:
                selected=[x for x in rows if x['world_id']==world and x['method']==method and x['role']==role]
                metrics={}
                for key in fields:
                    values=[x['metrics'][key]for x in selected if x['scored']and x['metrics'][key]is not None]
                    metrics[key]=dict(conditional_mean=float(np.mean(values))if values else None,available=len(values),expected=len(selected),full_coverage_mean=float(np.mean(values))if len(values)==len(selected)else None)
                summaries.append(dict(world_id=world,method=method,camera_role=role,scored=sum(x['scored']for x in selected),expected=len(selected),metrics=metrics))
    r.prediction_gate()
    r.write(output,dict(scope=p['scope'],plan_sha256=r.sha(r.PLAN),prediction_commit_sha256=r.sha(r.BASE/'prediction-commit.json'),expected_rows=2480,rows=rows,per_world_summary=summaries,evidence_sha256=evidence,qualification='Numerical n4 targets; image differences include mechanics AND different motion binding/covariance transport. Same initial learned GS. No automatic rendering parity claim; no independent frame/view sample claim.'))

if __name__=='__main__':score()
