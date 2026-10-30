"""Additive post-score independent verifier. Never run before explicit scoring.

Independent ridge/selection arithmetic and evaluator metrics; reused frozen
geometry/contact operators are disclosed. Original sources are never modified.
"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time,datetime,collections
import numpy as np
from scipy.spatial import cKDTree
REPORT=ROOT/'reports/prefix-ridge-switch-v13';BASE=ROOT/'runs/prefix-ridge-switch-v13';OLD=ROOT/'runs/contact-benchmark-v12'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def guard_amendment():
    amendment=read(REPORT/'independent-verifier-amendment.json')
    if sha(REPORT/'method-freeze.json')!=amendment['method_freeze_sha256'] or sha(REPORT/'protocol.json')!=amendment['protocol_sha256']:raise RuntimeError('Independent verifier amendment parent changed')
    for name,h in amendment['additive_source_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Independent verifier source changed: '+name)
    return amendment

def remaining(p):
    end=datetime.datetime.fromisoformat(p['compute_bound']['original_absolute_deadline_utc'].replace('Z','+00:00')).timestamp()
    return end-time.time()

def array(p,key):
    with np.load(p,allow_pickle=False) as a:return a[key].copy()
class Audit:
    def __init__(self):self.checks=0;self.errors=[]
    def check(self,ok,label):
        self.checks+=1
        if not ok:self.errors.append(label)
    def close(self,x,y,label):
        self.check(x is None and y is None or x is not None and y is not None and bool(np.isclose(x,y,rtol=1e-8,atol=1e-10,equal_nan=True)),label)

def ridge_independent(d):
    n=d['nodes'];target=d['controller_targets'];span=max(float(np.ptp(n,axis=0).max()),1e-6);disp=(target-n[None,None])/span
    means=disp.mean(axis=2);moments=np.einsum('tgni,nj->tgij',disp,(n-n.mean(0))/span)/len(n)
    features=np.concatenate([means.reshape(len(target),-1),moments.reshape(len(target),-1),d['attachment_active']],axis=1)
    end=int(d['train_end']);center=features[:end].mean(0);scale=features[:end].std(0);scale[scale<1e-5]=1
    design=np.column_stack([np.ones(len(target)),(features-center)/scale]);tracks=[]
    for j in range(len(d['observed_initial'])):
        valid=d['observed_mask'][:,j];X=design[:end][valid];Y=(d['observed'][:,j]-d['observed_initial'][j])[valid]
        coeff=np.linalg.solve(X.T@X+.1*np.eye(X.shape[1]),X.T@Y);tracks.append(design@coeff)
    tracks=np.stack(tracks,axis=1);distance,ids=cKDTree(d['observed_initial']).query(n,k=min(6,len(d['observed_initial'])))
    if distance.ndim==1:distance=distance[:,None];ids=ids[:,None]
    weight=np.exp(-3*(distance/np.maximum(distance[:,-1,None],1e-9))**2)/np.maximum(distance,1e-6);weight/=weight.sum(1,keepdims=True)
    return n[None]+np.einsum('nk,tnkc->tnc',weight,tracks[:,ids])

def guard_prediction(d,anchor,scene,elements):
    from scripts.contact_benchmark_v12.feasibility import repair,jacobians
    from scripts.contact_benchmark_v12.contact_rollout import rollout
    result,_=repair(d,anchor,ridge_independent(d),elements=elements)
    if d['dataset']=='SoftVTBench':result,_=rollout(d,result,scene)
    for t in range(len(result)):
        if np.isfinite(result[t]).all() and not (np.isfinite(jacobians(d['nodes'],result[t],elements))&(jacobians(d['nodes'],result[t],elements)>.03000001)).all():result[t]=np.nan
    return result

def validation_error(full,short,pred,cut,end):
    from scripts.contact_benchmark_v12.feasibility import jacobians,inherited_elements
    if not np.isfinite(pred[cut:end]).all():return float('inf')
    target=full['observed'][cut:end];mask=full['observed_mask'][cut:end]&np.isfinite(target).all(2)
    if not mask.any():return float('inf')
    projected=short['observed_initial'][None]+np.einsum('pn,tnc->tpc',short['observation_matrix'],pred[cut:end]-short['nodes'][None])
    return float(np.sqrt(np.mean((projected[mask]-target[mask])**2)))

def selection_case(case,attempt,a):
    folder=BASE/'forecasts'/case['cohort']/case['name'];meta=read(folder/'metadata.json');pred=array(folder/'prediction.npz','predicted');parent=OLD/'model-selection-evaluation'/case['cohort']/case['name'];old=array(parent/'predictions.npz','model_selection_selected')
    a.check(meta['case']==case,'metadata case');a.check(pred.shape==old.shape,'forecastshape')
    if not meta.get('switched'):a.check(pred.dtype==old.dtype and pred.tobytes()==old.tobytes(),'exact archived fallback')
    for name,h in meta['input_sha256'].items():a.check(sha(ROOT/name)==h,'forecast input '+name)
    if attempt['state']!='complete':
        a.check(not meta.get('switched') and bool(meta.get('attempt_error')),'explicit unavailable fallback');return
    from scripts.prefix_ridge_switch_v13.inputs import load_payload,load_scene_payload
    from scripts.contact_benchmark_v12.experiment import base_path
    from scripts.contact_benchmark_v12.model_selection import _predict,rigid_nodal,_topology_failures
    from scripts.contact_benchmark_v12.feasibility import inherited_elements
    inputdir=BASE/'inputs'/case['name'];full=load_payload(inputdir/'full.npz');short=load_payload(inputdir/'short.npz');end=full['train_end'];cut=short['train_end']
    for name,h in read(inputdir/'export-provenance.json')['payload_sha256'].items():a.check(sha(ROOT/name)==h,'safe exported payload '+name)
    d=dict(short)
    for key in ('times','controller_targets','attachment_active','controls','gripper_width','gripper_command'):
        if key in d:d[key]=d[key][:end].copy()
    scene=load_scene_payload(inputdir/'scene.npz') if full['dataset']=='SoftVTBench' else None
    chosen=read(parent/'prediction-metadata.json')['selected_configuration'];cfg=(chosen['family'],chosen['method'],chosen['strength'])
    try:
        if not 2<=cut<end:raise ValueError('invalid cut')
        elements=inherited_elements(d);current=array(base_path(full,case['cohort'],True),'predicted')[:end]
        anchor=current if cfg[0]=='current' else array(OLD/'model-selection-base/validation'/case['name']/'prediction.npz','predicted')[:end] if cfg[0]=='field_free' else rigid_nodal(d)
        oldvalidation,_=_predict(d,anchor,cfg,scene,elements);oldscore=validation_error(full,d,oldvalidation,cut,end)
        if _topology_failures(d,oldvalidation,cut,end,elements):oldscore=float('inf')
        expected=chosen['validation_rmse_m'] if chosen['validation_finite'] else float('inf')
        if not np.isclose(oldscore,expected,rtol=1e-10,atol=1e-12):raise ValueError('archived validation mismatch')
        newscore=validation_error(full,d,guard_prediction(d,current,scene,elements),cut,end)
    except Exception as error:
        a.check(not meta.get('switched') and 'validation_error' in meta,'validation error accounted');return
    a.check('validation_error' not in meta,'independent validation succeeds')
    a.close(meta.get('old_validation_rmse_m'),oldscore if np.isfinite(oldscore) else None,'old SHORT loss')
    a.close(meta.get('ridge_validation_rmse_m'),newscore if np.isfinite(newscore) else None,'ridge SHORT loss')
    switch=bool(np.isfinite(newscore) and newscore<.98*oldscore);a.check(meta['switched']==switch,'strict2percent switch')
    if switch:
        try:rebuilt=guard_prediction(full,array(base_path(full,case['cohort']),'predicted'),scene,elements)
        except Exception:rebuilt=np.full_like(pred,np.nan)
        a.check(np.array_equal(np.isfinite(pred),np.isfinite(rebuilt)),'selected failure mask')
        a.check(np.allclose(pred,rebuilt,rtol=1e-8,atol=1e-10,equal_nan=True),'selected full forecast reconstruction')

def metric_summary(data,nodal):
    from scripts.external_benchmark.data import interpolate_matrix
    initial=data.get('support_geometry',data['evaluation']['points'][0]);weights=interpolate_matrix(data['nodes'],initial)
    pred=initial[None]+np.einsum('pn,tnc->tpc',weights,nodal-data['nodes'][None]);ev=data['evaluation'];start=data['train_end'];rows=[];failed=0
    for t in range(start,len(data['times'])):
        if not np.isfinite(pred[t]).all():failed+=1;continue
        mask=ev['mask'][t]&np.isfinite(ev['points'][t]).all(1);x=pred[t,:len(ev['points'][t])][mask];y=ev['points'][t,mask];delta=x-y
        X=x-x.mean(0);Y=y-y.mean(0);u,_,v=np.linalg.svd(X.T@Y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@v);aligned=X@u@fix@v-Y
        rows.append({'mean_l2_mm':float(np.linalg.norm(delta,axis=1).mean()*1000),'coordinate_rmse_mm':float(np.sqrt(np.mean(delta**2))*1000),'aligned_shape_rmse_mm':float(np.sqrt(np.mean(aligned**2))*1000),'observed_to_predicted_chamfer_l1_mm':float(cKDTree(pred[t,:ev.get('surface_count',pred.shape[1])]).query(y,p=1)[0].mean()*1000)})
    result={k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
    result.update(total_frames=len(data['times'])-start,valid_frames=len(rows),failed_frames=failed)
    if data['dataset']=='PhysTwin':
        values=[]
        if np.isfinite(pred[0]).all():
            manual=ev['manual'];v0=np.isfinite(manual[0]).all(1);ids=cKDTree(pred[0]).query(manual[0,v0])[1]
            for t in range(start,len(pred)):
                truth=manual[t,v0];good=np.isfinite(truth).all(1)&np.isfinite(pred[t,ids]).all(1)
                if good.any():values.append(float(np.linalg.norm(pred[t,ids[good]]-truth[good],axis=1).mean()*1000))
        result['manual_track_mean_mm']=float(np.mean(values)) if values else None
    return result

def check_summary(a,saved,actual,label):
    for key,value in actual.items():a.close(saved.get(key),value,label+' '+key)
    for key in ('total_frames','valid_frames','failed_frames'):a.check(type(saved.get(key)) is int and saved[key]>=0,label+' nonnegative count '+key)
    a.check(saved['valid_frames']+saved['failed_frames']==saved['total_frames'],label+' frame accounting')
    for key,value in saved.items():
        if key.endswith('_mm') and value is not None:a.check(np.isfinite(value) and value>=0,label+' finite nonnegative '+key)

def aggregation(a,p,rows,summary):
    method='ridge_switch_selected';comparators=('model_selection_selected','model_selection_rigid','rigid_scene_state','prefix_calibrated_rigid','position_ridge','official_phystwin')
    for cohort in ['__all__',*dict.fromkeys(c['cohort'] for c in p['cases'])]:
        for dataset,metric in [('SoftVTBench','coordinate_rmse_mm'),('PhysTwin','manual_track_mean_mm')]:
            subset=[r for r in rows if r['dataset']==dataset and (cohort=='__all__' or r['cohort']==cohort)];saved=summary['overall_datasets'][dataset] if cohort=='__all__' else summary['cohorts'][cohort][dataset]
            a.check(saved['expected']==len(subset),'fixed cohort count')
            a.check(saved['attempt_states']==dict(collections.Counter(r['attempt_state'] for r in subset)),'attempt state counts')
            a.check(saved['switched']==sum(r['switched'] for r in subset),'switch counts')
            def complete(s):return s.get(metric) is not None and np.isfinite(s[metric]) and s[metric]>=0 and s.get('total_frames',0)>0 and s.get('valid_frames')==s.get('total_frames') and s.get('failed_frames')==0
            for name in (method,*comparators):
                applicable=not(name=='official_phystwin' and dataset!='PhysTwin' or name=='rigid_scene_state' and dataset!='SoftVTBench');items=[r['results'][name]['summary'] for r in subset if name in r['results']] if applicable else [];valid=[s[metric] for s in items if complete(s)];finite=[s[metric] for s in items if s.get(metric) is not None and np.isfinite(s[metric])];s=saved['methods'][name]
                a.check(s['present']==len(items) and s['complete']==len(valid) and s['expected']==(len(subset) if applicable else 0),'coverage '+name)
                a.check(s['failed_frames']==sum(item.get('failed_frames',0) for item in items),'failed frames '+name)
                missing=[row['name'] for row in subset if applicable and not complete(row['results'].get(name,{}).get('summary',{}))]
                a.check(s['incomplete_names']==missing,'incomplete case identities '+name)
                a.close(s['complete_only_mean_mm'],float(np.mean(valid)) if valid else None,'mean '+name);a.close(s['finite_case_mean_mm'],float(np.mean(finite)) if finite else None,'partialmean '+name)
                if name==method or not applicable:continue
                gains=[]
                for row in subset:
                    x=row['results'].get(method,{}).get('summary',{});y=row['results'].get(name,{}).get('summary',{})
                    if complete(x) and complete(y):gains.append(y[metric]-x[metric])
                s=saved['paired_comparisons'][name];a.check(s['matched']==len(gains) and s['expected']==len(subset),'paired coverage '+name);a.close(s['mean_gain_mm'],float(np.mean(gains)) if gains else None,'pairedgain '+name)
                a.check((s['wins'],s['losses'],s['ties'])==(sum(g>1e-9 for g in gains),sum(g< -1e-9 for g in gains),sum(abs(g)<=1e-9 for g in gains)),'winloss '+name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True);args=parser.parse_args()
    # No evaluator-capable data loader before completed scored artifacts + gate.
    guard_amendment()
    if not (REPORT/'scores.json').exists() or not (REPORT/'summary.json').exists():raise RuntimeError('Explicit scoring/report must finish first')
    from scripts.prefix_ridge_switch_v13.score import terminal_gate
    p,attempts,attemptsha=terminal_gate();scores=read(REPORT/'scores.json');summary=read(REPORT/'summary.json');a=Audit()
    a.check(scores['attempts_sha256']==attemptsha,'scored manifest');a.check(scores['protocol_sha256']==sha(REPORT/'protocol.json'),'scored protocol');a.check(summary['scores_sha256']==sha(REPORT/'scores.json'),'summary scores')
    a.check(scores['method_freeze_sha256']==sha(REPORT/'method-freeze.json') and scores['scorer_sha256']==sha(ROOT/'scripts/prefix_ridge_switch_v13/score.py'),'scorer/freeze binding')
    a.check(summary['protocol_sha256']==sha(REPORT/'protocol.json') and summary['method_freeze_sha256']==sha(REPORT/'method-freeze.json') and summary['reporter_sha256']==sha(ROOT/'scripts/prefix_ridge_switch_v13/report.py'),'report source binding')
    for name,h in scores['evidence_sha256'].items():a.check(sha(ROOT/name)==h,'score evidence '+name)
    rows=scores['cases'];a.check(len(rows)==148 and {r['name'] for r in rows}=={c['name'] for c in p['cases']},'all148 scored');lookup={r['name']:r for r in rows}
    from scripts.contact_benchmark_v12.data import load
    initial_scores=sha(REPORT/'scores.json');initial_summary=sha(REPORT/'summary.json');token=initial_scores+sha(Path(__file__));cache=REPORT/'independent-case-audits';case_results=[]
    for case in p['cases']:
        if remaining(p)<120:
            save(REPORT/'independent-audit-partial.json',{'passed':False,'errors':a.errors,'pending':['Original deadline leaves less than120seconds; resume checkpoints only under existing campaign authorization'],'checks':a.checks,'completed_cases':len(case_results),'scores_sha256':initial_scores,'verifier_sha256':sha(Path(__file__))})
            return 2
        out=cache/(case['name']+'.json')
        for name,h in read(BASE/'forecasts'/case['cohort']/case['name']/'metadata.json')['input_sha256'].items():a.check(sha(ROOT/name)==h,'current case input '+name)
        if out.exists() and read(out).get('token')==token:result=read(out)
        else:
            local=Audit();row=lookup[case['name']];attempt=attempts[case['name']]
            local.check(row['case']==case and row['attempt_state']==attempt['state'],'case/attempt identity')
            local.check(row['cohort']==case['cohort'] and row['dataset']==('PhysTwin' if 'case' in case['spec'] else 'SoftVTBench'),'dataset/cohort identity')
            try:
                selection_case(case,attempt,local)
                data=load(case['spec']);pred=array(BASE/'forecasts'/case['cohort']/case['name']/'prediction.npz','predicted');actual=metric_summary(data,pred)
                check_summary(local,row['results']['ridge_switch_selected']['summary'],actual,'newforecast')
                parent=read(OLD/'model-selection-evaluation'/case['cohort']/case['name']/'metrics.json')
                for method in row['results']:
                    check_summary(local,row['results'][method]['summary'],{},'metric validity '+method)
                    if method!='ridge_switch_selected':local.check(row['results'][method]==parent['results'][method],'exact archived comparator '+method)
            except Exception as error:local.check(False,'case verification exception '+repr(error))
            result={'token':token,'case':case,'checks':local.checks,'errors':local.errors};save(out,result)
        a.checks+=result['checks'];a.errors.extend(case['name']+': '+x for x in result['errors']);case_results.append(result)
        print(case['name'],len(result['errors']),'errors',flush=True)
    aggregation(a,p,rows,summary);terminal_gate();guard_amendment()
    a.check(sha(REPORT/'scores.json')==initial_scores and sha(REPORT/'summary.json')==initial_summary,'scored/report inputs stable during audit')
    output={'passed':not a.errors,'errors':a.errors,'checks':a.checks,'cases':case_results,'scores_sha256':sha(REPORT/'scores.json'),'historical_summary_sha256':initial_summary,'verified_summary_metrics_sha256':hashlib.sha256(json.dumps({k:v for k,v in summary.items() if k!='independent_audit'},sort_keys=True,separators=(',',':')).encode()).hexdigest(),'attempts_sha256':attemptsha,'verifier_sha256':sha(Path(__file__)),'scope':'Independent ridge features/coefficient/loss/decision reconstruction and suffix metric arithmetic. Reuses frozen contact/geometry/adaptation operators and dataset loading; no predictor selector or scorer invoked. Archived baseline score payloads verified unchanged, not retrained.'}
    save(REPORT/'independent-audit.json',output);return int(bool(a.errors))
if __name__=='__main__':raise SystemExit(main())
