"""Independent fixed-eight Gaussian arithmetic audit; only after scoring release."""
import sys,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.benchmark_improvement_v14.common import REPORT,read,write,sha,remaining,check_preservation
from scripts.benchmark_improvement_v14.phystwin_gaussian_production import guard,validate,folder
from scripts.benchmark_improvement_v14.phystwin_independent_audit import manual_score
from scripts.benchmark_improvement_v14.independent_audit_v14 import paired

def main():
 remaining();frozen,digest=guard();cases=frozen['cases'];errors=[];checks=0;evidence={}
 def check(ok,msg):
  nonlocal checks
  checks+=1
  if not ok:errors.append(msg)
 def bind(p):evidence[str(Path(p).relative_to(ROOT))]=sha(p)
 release=read(REPORT/'phystwin-gaussian-score-release.json')
 if release.get('authorized') is not True or release.get('freeze_sha256')!=digest:raise ValueError('Missing root scoring release')
 for c in cases:
  record=validate(c)
  for file in ('complete.json','predictions.npz','metadata.json'):bind(folder(c)/file)
  for p,h in record['inputs'].items():evidence[str(Path(p).relative_to(ROOT))]=h
 scores=read(REPORT/'phystwin-gaussian-scores.json');summary=read(REPORT/'phystwin-gaussian-summary.json');primary=read(REPORT/'phystwin-scores.json')
 expected={c['name']:c for c in cases}
 for dataset in (scores,primary):
  if len(dataset['cases'])!=8 or {r['case']['name']:r['case'] for r in dataset['cases']}!=expected:raise ValueError('Exact eight score identities required')
 if scores['freeze_sha256']!=digest or primary['freeze_sha256']!=frozen['parent_width_freeze_sha256']:raise ValueError('Score freeze mismatch')
 check(summary['freeze_sha256']==digest,'summary freeze');check(summary['gaussian_scores_sha256']==sha(REPORT/'phystwin-gaussian-scores.json'),'Gaussian summary input');check(summary['primary_scores_sha256']==sha(REPORT/'phystwin-scores.json'),'primary summary input')
 lookup={r['case']['name']:r for r in scores['cases']};old={r['case']['name']:r for r in primary['cases']};combined=[]
 for c in cases:
  r=lookup[c['name']];parent=old[c['name']]
  if r['prediction_sha256']!=sha(folder(c)/'predictions.npz') or r['freeze_sha256']!=digest:raise ValueError('Score artifact mismatch')
  check(r['metrics']['object_frame']==parent['metrics']['object_frame'],'original metrics identical '+c['name'])
  methods=dict(parent['metrics']);methods['attachment_width_primary']=methods['benchmark_v14'];methods.update(r['metrics']);methods['benchmark_v14']=r['metrics']['gaussian_v14'];combined.append(dict(case=c,metrics=methods,archived_controls=parent.get('archived_controls',{})))
 for label,result in dict(summary['comparisons'],coordinate_vs_original=summary['coordinate_vs_original'],shape_vs_original=summary['shape_vs_original']).items():
  metric='aligned_shape_rmse_mm' if label=='shape_vs_original' else 'coordinate_rmse_mm' if label=='coordinate_vs_original' else 'manual_track_mean_mm'
  values=paired(combined,result['baseline'],metric);check(result['available_pairs']==len(values) and result['declared_cases']==8,'pair coverage '+label)
  coverage=[];complete=[]
  for row in combined:
   methods=dict(row.get('archived_controls',{}),**row['metrics']);a=methods['benchmark_v14']['summary'];b=methods.get(result['baseline'],{}).get('summary',{})
   coverage.append(dict(name=row['case']['name'],new_failed=a.get('failed_frames'),old_failed=b.get('failed_frames'),new_valid=a.get('valid_frames'),old_valid=b.get('valid_frames'),total=a.get('total_frames'),old_total=b.get('total_frames')))
   if a.get(metric) is not None and b.get(metric) is not None and a['failed_frames']==b.get('failed_frames')==0 and a['total_frames']==b.get('total_frames'):complete.append((a[metric],b[metric]))
  check(result['coverage']==coverage,'frame coverage '+label)
  check(result['all_cases_and_frames_complete']==(len(complete)==8),'complete flag '+label)
  if not values:continue
  check(result['complete_pairs']['count']==len(complete),'complete pair count '+label)
  if complete:
   cv=np.array(complete);check(np.isclose(result['complete_pairs']['proposal_mean_mm'],cv[:,0].mean()) and np.isclose(result['complete_pairs']['baseline_mean_mm'],cv[:,1].mean()),'complete means '+label)
  a=np.array([v[1] for v in values]);b=np.array([v[2] for v in values]);gain=b-a
  for key,value in [('proposal_mean_mm',a.mean()),('baseline_mean_mm',b.mean()),('mean_improvement_mm',gain.mean())]:check(np.isclose(result[key],value,rtol=1e-10,atol=1e-8),'paired '+label+key)
  check((result['wins'],result['losses'],result['ties'])==(int((gain>1e-8).sum()),int((gain< -1e-8).sum()),int((abs(gain)<=1e-8).sum())),'wins '+label)
  boot=gain[np.random.default_rng(140031).integers(0,len(gain),(10000,len(gain)))].mean(1);check(np.allclose(np.quantile(boot,[.025,.975]),result['paired_episode_bootstrap_95_interval_mm']),'bootstrap '+label)
 # No target loader is imported until every frozen forecast and scored case passes identity gates.
 from scripts.external_benchmark.data import phystwin,interpolate_matrix
 raw=[]
 for c in cases:
  remaining();name=c['name'];data=phystwin(name,300);target=data['evaluation']['points'];mask=data['evaluation']['mask'];manual=data['evaluation']['manual'];initial=data['support_geometry'];A=interpolate_matrix(data['nodes'],initial);end=data['train_end'];meta=read(folder(c)/'metadata.json');results={}
  with np.load(folder(c)/'predictions.npz') as saved:
   if meta['fallback_original']:check(saved['gaussian_v14'].tobytes()==saved['object_frame'].tobytes(),'byte-exact fallback '+name)
   check(lookup[name]['fallback_original']==meta['fallback_original'],'fallback score '+name)
   for method in saved.files:
    expanded=initial[None]+np.einsum('pn,tnc->tpc',A,saved[method]-data['nodes'][None]);coordinate=[];shape=[];failed=0
    for t in range(end,len(expanded)):
     if not np.isfinite(expanded[t]).all():failed+=1;continue
     valid=mask[t]&np.isfinite(target[t]).all(1);a=expanded[t,:len(target[t])][valid];b=target[t,valid];coordinate.append(float(np.sqrt(np.mean((a-b)**2))*1000));ac=a-a.mean(0);bc=b-b.mean(0);u,_,vt=np.linalg.svd(ac.T@bc);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt);shape.append(float(np.sqrt(np.mean((ac@(u@fix@vt)-bc)**2))*1000))
    actual=dict(coordinate_rmse_mm=float(np.mean(coordinate)) if coordinate else None,aligned_shape_rmse_mm=float(np.mean(shape)) if shape else None,manual_track_mean_mm=manual_score(expanded,manual,end),failed_frames=failed,valid_frames=len(coordinate),total_frames=len(expanded)-end);wanted=lookup[name]['metrics'][method]['summary']
    for key,value in actual.items():check(wanted.get(key) is None if value is None else wanted.get(key) is not None and np.isclose(wanted[key],value,rtol=1e-8,atol=1e-7),'raw '+name+method+key)
    results[method]=actual
  raw.append(dict(name=name,methods=results,fallback=meta['fallback_original']))
 check(summary['fallback_cases']==[c['name'] for c in cases if lookup[c['name']]['fallback_original']],'fallback list')
 preserved=check_preservation()
 for p in [REPORT/x for x in ('phystwin-gaussian-scores.json','phystwin-gaussian-summary.json','phystwin-scores.json','phystwin-gaussian-freeze.json','phystwin-gaussian-score-release.json')]+[Path(__file__)]:bind(p)
 write(REPORT/'phystwin-gaussian-final-independent-audit.json',dict(passed=not errors,checks=checks,errors=errors,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Independent all-eight saved-frame coordinate/proper-shape/manual score and paired/bootstrap arithmetic; frozen geometry interpolation reused. No optimizer rerun. Historical exploratory cohort.',raw_cases=raw,prior_files_preserved=preserved,evidence_sha256=evidence))
 if errors:raise RuntimeError(str(errors[:10]))
 print('Gaussian independent audit passed',checks)
if __name__=='__main__':main()
