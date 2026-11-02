"""Independent final image arithmetic audit; no fitting or target generation."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17_reporting import cohort_appearance_inference as inference
ROOT=inference.ROOT;REPORT=ROOT/'reports/compositional-dynamics-v17'

def main():
    started=time.time();renderer=inference.bound();p,commit=renderer.prediction_gate()
    source=REPORT/'fresh-cohort-appearance-results.json'
    if not source.exists():raise RuntimeError('Scores pending; no audit claimed')
    results=renderer.read(source);evidence={str(source.relative_to(ROOT)):renderer.sha(source),str((renderer.BASE/'prediction-commit.json').relative_to(ROOT)):renderer.sha(renderer.BASE/'prediction-commit.json')};errors=[];checks=0;ssim_checks=0
    def check(ok,msg):
        nonlocal checks
        checks+=1
        if not ok and len(errors)<200:errors.append(msg)
    def equal(actual,expected,label):
        check((actual is None and expected is None)or(actual is not None and expected is not None and bool(np.isclose(actual,expected,rtol=2e-7,atol=2e-10))),label)
    check(results['plan_sha256']==renderer.sha(renderer.PLAN)and results['prediction_commit_sha256']==renderer.sha(renderer.BASE/'prediction-commit.json'),'Score seals')
    for path,h in results['evidence_sha256'].items():check(renderer.sha(ROOT/path)==h,'Evidence '+path);evidence[path]=h
    keys=lambda x:(x['world_id'],x['query'],x['method'],x['frame'],x['camera'])
    sealed={keys(x):x for x in commit['rows']};rows={keys(x):x for x in results['rows']}
    check(len(results['rows'])==2480 and len(rows)==2480 and set(rows)==set(sealed),'Exact2480census')
    initial={};calculated={};reference_cache={};counts=dict(scored=0,failed_prediction=0,failed_reference=0)
    def frame(path):
        with np.load(path)as f:data={k:f[k].copy()for k in ['rgb','body_visible','body_visible_depth','time_s']}
        rgb=data['rgb'].astype(np.float64);mask=data['body_visible'];depth=data['body_visible_depth'].astype(np.float64)
        check(rgb.shape==(480,848,3)and np.isfinite(rgb).all()and np.all((rgb>=0)&(rgb<=1)),'RGB validity '+str(path))
        check(mask.dtype==bool and mask.shape==(480,848)and depth.shape==mask.shape,'Mask/depth shape')
        check(np.array_equal(np.isfinite(depth),mask)and not np.isnan(depth).any()and not np.isneginf(depth).any()and np.all(depth[mask]>0),'Depth convention')
        return rgb,mask,depth,float(data['time_s'])
    from skimage.metrics import structural_similarity
    for key,row in rows.items():
        label='/'.join(map(str,key));old=sealed[key]
        check(all(row.get(k)==old.get(k)for k in ['completed','time_s','role','prediction']),label+' prediction binding')
        if not row['completed']:counts['failed_prediction']+=1
        if not row['reference_complete']:counts['failed_reference']+=1
        check(row['scored']==bool(row['completed']and row['reference_complete']),label+' coverage')
        if row['completed']:
            a,ma,da,ta=frame(ROOT/row['prediction']['path']);equal(ta,row['time_s'],label+' prediction time')
            evidence[row['prediction']['path']]=row['prediction']['sha256']
            if row['frame']==0:initial[key]=a.copy()
        if not row['scored']:continue
        counts['scored']+=1
        ref=renderer.BASE/'references'/row['world_id']/row['query']/f"{row['frame']:03d}-{row['camera']}.npz"
        path=str(ref.relative_to(ROOT));check(results['evidence_sha256'].get(path)==row['reference_sha256'],label+' reference binding')
        if path not in reference_cache:reference_cache={path:frame(ref)}
        b,mb,db,tb=reference_cache[path];equal(ta,tb,label+' matched time')
        union=ma|mb;joint=ma&mb;delta=(a-b)**2
        mse=float(delta.mean());umse=float(delta[union].mean())if union.any()else None
        values=dict(rgb_global_mse=mse,rgb_foreground_union_mse=umse,psnr_global_db=-10*np.log10(max(mse,1e-12)),psnr_foreground_union_db=-10*np.log10(max(umse,1e-12))if umse is not None else None,silhouette_iou=float(joint.sum()/union.sum())if union.any()else 1.,reference_body_pixels=int(mb.sum()),predicted_body_pixels=int(ma.sum()),union_body_pixels=int(union.sum()),joint_depth_pixels=int(joint.sum()),missing_predicted_pixels=int((mb&~ma).sum()),extra_predicted_pixels=int((ma&~mb).sum()),reference_depth_coverage=float(joint.sum()/mb.sum())if mb.any()else None,depth_joint_rmse_m=float(np.sqrt(np.mean((da[joint]-db[joint])**2)))if joint.any()else None)
        for name,v in values.items():equal(row['metrics'][name],v,label+' '+name)
        if row['frame']in [0,10,20,30]and row['camera']in ['held-0','train-0']:
            score,maps=structural_similarity(a,b,data_range=1.,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
            inner=union[5:-5,5:-5];foreground=float(maps[5:-5,5:-5][inner].mean())if inner.any()else None
            equal(row['metrics']['ssim_rgb_global'],score,label+' officialSSIM');equal(row['metrics']['ssim_rgb_foreground_union'],foreground,label+' foregroundSSIM');ssim_checks+=1
        calculated[key]=values
    parity=[]
    for q in p['queries']:
        observed=[]
        for camera in q['cameras']:
            a=(q['world_id'],q['role'],'native',0,camera['name']);b=(q['world_id'],q['role'],'shared_q2',0,camera['name'])
            if a in initial and b in initial:observed.append(float(np.max(abs(initial[a]-initial[b]))))
        recorded=[x for x in commit['query_evidence']if x['world_id']==q['world_id']and x['query']==q['role']]
        check(len(recorded)==1,'Parity query census')
        if len(observed)==5:
            equal(recorded[0]['initial_rgb_max_difference'],max(observed),'Initial parity recorded');check(max(observed)<=1e-4,'Initial appearance parity')
        else:check(recorded[0]['initial_rgb_max_difference']is None,'No false parity on unavailable native')
        parity.append(dict(world_id=q['world_id'],query=q['role'],pairs=len(observed),max_rgb_difference=max(observed)if observed else None))
    for summary in results['per_world_summary']:
        selected=[r for r in rows.values()if r['world_id']==summary['world_id']and r['method']==summary['method']and r['role']==summary['camera_role']]
        check(summary['expected']==len(selected)and summary['scored']==sum(r['scored']for r in selected),'Summary coverage')
        for metric,record in summary['metrics'].items():
            vals=[r['metrics'][metric]for r in selected if r['scored']and r['metrics'][metric]is not None]
            check(record['available']==len(vals)and record['expected']==len(selected),'Metric denominator')
            equal(record['conditional_mean'],float(np.mean(vals))if vals else None,'Conditional aggregate '+metric)
            equal(record['full_coverage_mean'],float(np.mean(vals))if len(vals)==len(selected)else None,'Full coverage aggregate '+metric)
    renderer.prediction_gate();check(renderer.sha(source)==evidence[str(source.relative_to(ROOT))],'Results stable during audit')
    report=dict(passed=not errors,checks=checks,errors=errors,counts=counts,official_ssim_sample_count=ssim_checks,official_ssim_sample='Everyworld/bothloads/frames0,10,20,30/camerasheld-0,train-0/bothmethods whenavailable. Failures retained.',initial_parity=parity,evidence_sha256=evidence,auditor_sha256=renderer.sha(__file__),seconds=time.time()-started,scope='Independent image arithmetic on all successful slots; official SSIM fixed sample and all summary arithmetic. No target generation, fitting, tuning or duplicate production scoring. Integrity is not spatial reference qualification.')
    renderer.write(REPORT/'fresh-cohort-appearance-final-independent-audit.json',report);print(json.dumps({k:report[k]for k in ['passed','checks','errors','counts','official_ssim_sample_count','seconds']}))
if __name__=='__main__':main()
