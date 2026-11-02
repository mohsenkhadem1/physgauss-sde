"""Read-only native prediction/provenance audit; no reference target API."""
from pathlib import Path
import json,hashlib,pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
REPORT=ROOT/'reports/compositional-dynamics-v17'

def main():
    evidence={};errors=[];checks=0
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(p):
        evidence[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
    def check(condition,message):
        nonlocal checks
        checks+=1
        if not condition:errors.append(message)
    def pin(p,digest):check(sha(p)==digest,'Hash '+str(p));evidence[str(p.relative_to(ROOT))]=sha(p)
    planpath=REPORT/'native-cohort-recovery-plan.json';p=read(planpath);manifest=read(BASE/'native-recovery/recovery-manifest.json')
    for group in ['source_sha256','original_evidence_sha256']:
        for name,h in p[group].items():pin(ROOT/name,h)
    check(manifest['terminal']and manifest['expected']==8 and manifest['plan_sha256']==sha(planpath),'Manifest identity')
    expected={(f'world-{i}',role)for i in range(4)for role in ['forecast-40mm','forecast-80mm']}
    check(len(manifest['rows'])==8 and {(r['world_id'],r['role'])for r in manifest['rows']}==expected,'Exact8census')
    import torch
    summaries=[]
    for world in p['worlds']:
        oldpath=ROOT/world['original_plan'];newpath=ROOT/world['recovery_plan'];old=read(oldpath);new=read(newpath);pin(newpath,world['recovery_plan_sha256'])
        check({k:v for k,v in old.items()if k!='output_directory'}=={k:v for k,v in new.items()if k!='output_directory'},'Routing-only plan '+world['world_id'])
        for group in ['source_sha256','input_sha256']:
            for name,h in new[group].items():pin(ROOT/name,h)
        original=ROOT/world['original_output'];check(not(original/'fit').exists()and not list(original.rglob('*.pth')),'No original optimizer '+world['world_id'])
        base=ROOT/new['output_directory'];fit=read(base/'fit/fit-complete.json')
        check(fit['plan_sha256']==sha(newpath),'Fit plan binding')
        if not fit['completed']:
            for row in [r for r in manifest['rows']if r['world_id']==world['world_id']]:
                oldcommit=ROOT/row['original_commit'];newcommit=ROOT/row['recovery_commit'];pin(oldcommit,row['original_commit_sha256']);pin(newcommit,row['recovery_commit_sha256']);oc=read(oldcommit);nc=read(newcommit)
                check(not oc['completed']and not nc['completed']and row['completed']is False,'Failed fit retains both failures')
                check(nc['world_id']==row['world_id']and nc['role']==row['role']and nc['native_world_plan_sha256']==sha(newpath)and nc['recovery_plan_sha256']==sha(planpath)and nc['original_prediction_commit_sha256']==sha(oldcommit),'Failed replay bindings')
                check(nc['protocol_sha256']==oc['protocol_sha256']and nc['method_freeze_sha256']==oc['method_freeze_sha256'],'Failed original cohort bindings')
                summaries.append(dict(world_id=row['world_id'],role=row['role'],completed=False,frames=0,error=fit.get('error')))
            continue
        check(fit['epochs']==1000 and fit['cma_generations_requested']==100,'Exact fit recipe')
        for file,key in [('prefix.pkl','prefix_sha256'),('optimal_params.pkl','optimal_params_sha256'),('fitted-spring-endpoints.npy','spring_endpoints_sha256')]:pin(base/'fit'/file,fit[key])
        pin(ROOT/fit['checkpoint'],fit['checkpoint_sha256']);checkpoint=torch.load(ROOT/fit['checkpoint'],map_location='cpu',weights_only=False)
        edges=np.load(base/'fit/fitted-spring-endpoints.npy');check(edges.ndim==2 and edges.shape[1]==2 and np.min(edges)>=0 and np.max(edges)<125,'Graph endpoints');check(len(checkpoint['spring_Y'])==len(edges),'Graph checkpoint edgecount')
        check(all(torch.isfinite(v).all().item()for v in checkpoint.values()if isinstance(v,torch.Tensor)),'Finite fitted checkpoint')
        public=ROOT/new['public_directory'];meta=read(public/'metadata.json')
        with np.load(public/'observations.npz')as f:rest=np.concatenate([f['object_points'][0],f['surface_points'],f['interior_points']])
        with (base/'fit/prefix.pkl').open('rb')as f:prefix=pickle.load(f)
        check(np.allclose(prefix['object_points'][0][:,2]+meta['public_floor_z_m'],rest[:98,2],rtol=0,atol=1e-8),'Prefix public frame')
        for row in [r for r in manifest['rows']if r['world_id']==world['world_id']]:
            oldcommit=ROOT/row['original_commit'];newcommit=ROOT/row['recovery_commit'];pin(oldcommit,row['original_commit_sha256']);pin(newcommit,row['recovery_commit_sha256']);oc=read(oldcommit);nc=read(newcommit)
            check(not oc['completed']and nc['completed']==row['completed'],'Original/recovery statuses')
            check(nc['world_id']==row['world_id']and nc['role']==row['role']and nc['original_prediction_commit_sha256']==sha(oldcommit)and nc['native_world_plan_sha256']==sha(newpath)and nc['recovery_plan_sha256']==sha(planpath),'Outer bindings')
            check(nc['protocol_sha256']==oc['protocol_sha256']and nc['method_freeze_sha256']==oc['method_freeze_sha256'],'Original cohort binding')
            status=read(newcommit.parent/'complete.json');pin(newcommit.parent/'complete.json',nc['native_complete_sha256'])
            pin(newcommit.parent/'prediction.npz',nc['prediction_sha256']);check(status['plan_sha256']==sha(newpath)and status['fit_commit_sha256']==sha(base/'fit/fit-complete.json')and status['checkpoint_sha256']==fit['checkpoint_sha256'],'Replay fit/graph/checkpoint binding')
            check(status['prediction_sha256']==nc['prediction_sha256']and status['completed']and status['frames']==31,'Replay completion')
            setup=next(q['setup']for q in new['forecast_setups']if q['name']==row['role']);check(status['setup']==setup,'Public reset setup')
            with np.load(newcommit.parent/'prediction.npz')as f:
                x=f['positions_m'];t=f['times_s'];tracked=f['tracked_positions_m'];check(x.shape==(31,125,3)and tracked.shape==(31,98,3),'Array dimensions');check(np.isfinite(x).all()and np.isfinite(t).all(),'Finite forecasts');check(np.array_equal(tracked,x[:,:98]),'Tracked correspondence');check(np.allclose(t,np.arange(31)*.01,atol=1e-12,rtol=0),'31frame clock');check(np.allclose(x[0],rest,atol=1e-8,rtol=0),'Same public initial geometry')
            summaries.append(dict(world_id=row['world_id'],role=row['role'],completed=nc['completed'],frames=31))
    output=dict(passed=not errors,checks=checks,errors=errors,rows=summaries,scope='Read-only audit of operationally recovered native fit/replay bindings and finite predictions. No fresh reference arrays or scores read. Runtime exact graph equality is enforced by pinned replay source; this audit checks its saved fitted graph/checkpoint provenance, not a fresh native rollout.',evidence_sha256=evidence,auditor_sha256=sha(__file__))
    (REPORT/'native-cohort-recovery-prediction-audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'passed':not errors,'checks':checks,'errors':errors}))
if __name__=='__main__':main()
