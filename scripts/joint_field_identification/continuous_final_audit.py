"""Independent terminal audit and descriptive paired continuous-posterior results.
No scientific sources, observations, jobs, or exclusion criteria are modified.
"""
from pathlib import Path
import datetime, hashlib, json, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
METRICS=('rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90','interval_width_mm')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def name(j):return f"{j['shape']}-{j['population']}-case{j['case']:03d}-{j['stage']}-n{j['particles']}-s{j['seed']}"
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def audit(root):
 report=ROOT/'reports'/root;manifest=read(report/'continuous/manifest.json');summary=read(report/'continuous-summary.json');records={};issues=[]
 for f,h in manifest['source_sha256'].items():
  if sha(ROOT/f)!=h:issues.append('frozen source mismatch: '+f)
 for j in manifest['allowed_jobs']:
  p=report/'continuous'/f'{name(j)}.json'
  if p.exists():
   d=read(p)
   if d.get('complete'):records[name(j)]=d
 if len(records)!=288:return dict(complete=False,completed=len(records))
 if not summary['complete']:return dict(complete=False,completed=len(records),pending='summary refresh')
 dispatch=report/'continuous/dynamic-dispatch.json'
 if dispatch.exists() and read(dispatch).get('status')!='complete':return dict(complete=False,completed=len(records),pending='dispatcher terminal state')
 main={};finite={};groups=[]
 for d in records.values():
  j=d['job']
  if d['temperature']!=1 or not all(np.isfinite(v) for v in d['metrics'].values()):issues.append('invalid completed record '+name(j))
  if j['population']=='continuous-coarse':
   k=(j['shape'],j['case'],j['stage']);main[k]=d['metrics']
   raw=read(report/j['shape']/j['population']/f"case-{j['case']:03d}.json")
   finite[k]=next(r for r in raw['rows'] if r['policy']==('active' if j['stage']=='initial' else j['stage']) and r['step']==(0 if j['stage']=='initial' else 4))
 for s in summary['summary']:
  ds=[d for d in records.values() if d['job']['population']==s['population'] and d['job']['stage']==s['stage'] and (s['population']=='continuous-coarse' or (d['job']['particles']==1024 and d['job']['seed']==(171001 if 'surface' in root else 71001)))]
  vals=[]
  for d in ds:
   if s['model']=='continuous-smc':vals.append(d['metrics'])
   else:
    j=d['job'];raw=read(report/j['shape']/j['population']/f"case-{j['case']:03d}.json")
    vals.append(next(r for r in raw['rows'] if r['policy']==('active' if j['stage']=='initial' else j['stage']) and r['step']==(0 if j['stage']=='initial' else 4)))
  if len(vals)!=s['n']:issues.append('summary count mismatch')
  errors={m:abs(float(np.mean([v[m] for v in vals]))-s[m]) for m in METRICS}
  if max(errors.values())>1e-12:issues.append('summary arithmetic mismatch')
  groups.append(dict(population=s['population'],stage=s['stage'],model=s['model'],n=len(vals),max_absolute_mean_discrepancy=max(errors.values())))
 rng=np.random.default_rng(632001);comparisons=[]
 # Descriptive posthoc intervals, stratified by geometry and paired by case.
 pairs=[]
 for stage in ('initial','active','fixed'):
  pairs.append((f'continuous-minus-finite:{stage}',[(k,main[k],finite[k]) for k in sorted(main) if k[2]==stage]))
 for other in ('initial','fixed'):
  pairs.append((f'continuous-active-minus-{other}',[(k,main[k],main[(k[0],k[1],other)]) for k in sorted(main) if k[2]=='active']))
 for label,rr in pairs:
  idxs=[np.array([i for i,(k,_,_) in enumerate(rr) if k[0]==shape]) for shape in ('block','step')]
  boot=np.concatenate([rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in idxs],axis=1)
  for metric in METRICS:
   delta=np.array([a[metric]-b[metric] for _,a,b in rr]);ci=np.quantile(delta[boot].mean(1),[.025,.975]);comparisons.append(dict(comparison=label,metric=metric,n=len(rr),mean_difference=float(delta.mean()),descriptive_paired95=list(ci)))
 numerical=read(report/'continuous-numerical-comparisons.json')['comparisons']
 evidence=read(report/'evidence-audit.json')
 if not evidence['integrity_pass']:issues.append('main evidence integrity failure')
 result=dict(complete=True,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(report/'continuous/manifest.json'),summary_sha256=sha(report/'continuous-summary.json'),completed=len(records),main_count=len(main),diagnostic_count=len(records)-len(main),integrity_pass=not issues,issues=issues,summary_checks=groups,paired_comparisons=comparisons,numerical=dict(pairs=len(numerical),flagged=sum(r['flag_cdf'] or r['flag_prediction'] for r in numerical),max_cdf_difference=max(r['max_cdf_difference'] for r in numerical),max_prediction_mean_rms_difference_mm=max(r['target_mean_rms_difference_mm'] for r in numerical)),invalid_likelihood_evaluations=sum(d['invalid_evaluation_count'] for d in records.values()),limits=['Intervals are posthoc descriptive paired bootstrap intervals, not new confirmatory tests.','Actions were chosen by finite-dictionary planning; continuous inference reuses identical acquired observations.','Continuous and finite priors differ; differences are not exclusively quadrature error.','No numerical sensitivity flag is proof of convergence; no cases excluded.','Finite-particle SMC approximates the continuous posterior, while conditional force integration is analytic.'])
 save(report/'continuous-independent-final-audit.json',result);return result
if __name__=='__main__':
 deadline=datetime.datetime.fromisoformat('2026-11-07T07:37:29+00:00');done=set()
 while datetime.datetime.now(datetime.timezone.utc)<deadline:
  for root in ('joint-field-identification-v01','joint-field-surface-v04'):
   if root in done:continue
   result=audit(root);print(root,{k:result.get(k) for k in ('complete','completed','pending','integrity_pass')},flush=True)
   if result['complete']:done.add(root)
  if len(done)==2:break
  time.sleep(45)
