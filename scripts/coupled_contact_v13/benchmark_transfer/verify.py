"""Independent per-case score/accounting audit plus frozen original pose replay."""
from pathlib import Path
import concurrent.futures,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/coupled-surface-v13/benchmark-transfer';BASE=ROOT/'runs/coupled-surface-v13/benchmark-transfer';OLD=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def case_name(spec):return spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"

def primary_score(data,nodes):
    # Independently reconstruct expansion and primary statistic; no score helper.
    from scripts.external_benchmark.data import interpolate_matrix
    initial=data.get('support_geometry',data['evaluation']['points'][0]);A=interpolate_matrix(data['nodes'],initial)
    pred=initial[None]+np.einsum('pn,tnc->tpc',A,nodes-data['nodes'][None]);start=data['train_end'];values=[]
    if data['dataset']=='PhysTwin':
        manual=data['evaluation']['manual'];first=np.isfinite(manual[0]).all(1)
        index=np.argmin(np.sum((manual[0,first,None,:]-pred[0,None,:,:])**2,axis=2),axis=1)
        for t in range(start,len(pred)):
            truth=manual[t,first];valid=np.isfinite(truth).all(1)&np.isfinite(pred[t,index]).all(1)
            if valid.any():values.append(float(np.mean(np.linalg.norm(pred[t,index[valid]]-truth[valid],axis=1))*1000))
    else:
        target=data['evaluation']['points'];mask=data['evaluation']['mask']
        for t in range(start,len(pred)):
            valid=mask[t]&np.isfinite(target[t]).all(1)
            if np.isfinite(pred[t]).all():values.append(float(np.sqrt(np.mean((pred[t,:len(target[t])][valid]-target[t,valid])**2))*1000))
    return float(np.mean(values)) if values else None

def audit_case(spec):
    from scripts.contact_benchmark_v12.data import load,predictor_payload
    from scripts.contact_benchmark_v12.scene import load_scene
    from scripts.contact_benchmark_v12.contact_rollout import rollout
    case=case_name(spec);data=load(spec);errors=[];row=read(BASE/case/'metrics.json')
    with np.load(BASE/case/'prediction.npz') as a:old=a['previous'].copy();new=a['predicted'].copy()
    with np.load(OLD/case/'predictions.npz') as a:archive=a['model_selection_selected'].copy()
    if not np.array_equal(old,archive,equal_nan=True):errors.append('Previous array differs from archived selected array')
    unchanged=np.ones(len(old),bool)
    if data['dataset']=='SoftVTBench':
        payload=predictor_payload(data);scene=load_scene(payload);replayed,meta=rollout(payload,archive,scene)
        if not np.array_equal(replayed,archive,equal_nan=True):errors.append('Original frozen replay not bitwise exact')
        free=[x['frame'] for x in meta['diagnostics']];unchanged[free]=False
        savedmeta=read(BASE/case/'prediction-metadata.json')
        if free!=[x['frame'] for x in savedmeta['diagnostics']]:errors.append('Changed-frame policy differs from original free-frame policy')
        if row['integrated_free_frames']!=len(free):errors.append('Wrong free-frame accounting')
    if not np.array_equal(new[unchanged],archive[unchanged],equal_nan=True):errors.append('Prefix/held or PhysTwin-copy changed')
    before=primary_score(data,old);after=primary_score(data,new)
    if not np.isfinite(new).all() or not row['complete']:errors.append('Incomplete forecast')
    for key,value in [('previous',before),('volume_pose',after),('gain_mm',before-after)]:
        if not np.isclose(row[key],value,rtol=1e-10,atol=1e-10):errors.append('Primary score mismatch '+key)
    if row['prediction_sha256']!=digest(BASE/case/'prediction.npz'):errors.append('Prediction hash mismatch')
    if row['archive_sha256']!=digest(OLD/case/'predictions.npz'):errors.append('Archive hash mismatch')
    if row['protocol_sha256']!=digest(REPORT/'protocol.json'):errors.append('Protocol hash mismatch')
    return dict(case=case,dataset=data['dataset'],previous_mm=before,new_mm=after,gain_mm=before-after,errors=errors,unchanged_frames=int(unchanged.sum()),all_frames_unchanged=bool(np.array_equal(new,old,equal_nan=True)))

def main():
    protocol=read(REPORT/'protocol.json');result=read(REPORT/'results.json');errors=[]
    for p,h in {**protocol['sources'],**protocol['archived_inputs']}.items():
        if digest(ROOT/p)!=h:errors.append('Frozen hash changed '+p)
    declared=[case_name(s) for s in protocol['cases']];present=[r['case'] for r in result['cases']]
    if len(declared)!=21 or len(present)!=21 or len(set(present))!=21 or set(present)!=set(declared):errors.append('Wrong21-case accounting')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(audit_case,protocol['cases']))
    summary={}
    for dataset in ['SoftVTBench','PhysTwin']:
        subset=[r for r in rows if r['dataset']==dataset];gain=np.array([r['gain_mm'] for r in subset]);summary[dataset]=dict(cases=len(subset),previous_mm=float(np.mean([r['previous_mm'] for r in subset])),new_mm=float(np.mean([r['new_mm'] for r in subset])),wins=int((gain>1e-9).sum()),losses=int((gain< -1e-9).sum()),ties=int((abs(gain)<=1e-9).sum()))
        saved=result['summary'][dataset];s=summary[dataset]
        if any(saved[k]!=s[k] for k in ['wins','losses','ties']):errors.append(dataset+' aggregate win/loss mismatch')
        if not np.isclose(saved['mean_previous_mm'],s['previous_mm'],atol=1e-10) or not np.isclose(saved['mean_volume_pose_mm'],s['new_mm'],atol=1e-10):errors.append(dataset+' aggregate mean mismatch')
    for row in rows:errors.extend(row['case']+': '+e for e in row['errors'])
    audit=dict(passed=not errors,errors=errors,cases=rows,summary=summary,source_sha256=digest(Path(__file__)),scope='Primary scores independently recomputed; original frozen rollout replayed exactly; allprefix/heldstates and archives checked. Development-only mixed effect, not two-dataset improvement.')
    (REPORT/'independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(passed=audit['passed'],errors=errors,summary=summary),indent=2));return int(not audit['passed'])
if __name__=='__main__':sys.exit(main())
