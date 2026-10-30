"""Independent reserved-only forecast recomputation; no fitting or simulation."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/coupled-friction-v13';BASE=ROOT/'runs/coupled-friction-v13'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def metric(pred,target):
 if pred is None or pred.shape!=target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():return dict(complete=False,coordinate_rmse_m=None,shape_rmse_m=None)
 a=pred[1:]-pred[1:].mean(1,keepdims=True);b=target[1:]-target[1:].mean(1,keepdims=True)
 u,_,vt=np.linalg.svd(np.einsum('tni,tnj->tij',a,b));sign=np.linalg.det(u@vt);u[:,:,-1]*=sign[:,None];aligned=a@(u@vt)
 return dict(complete=True,coordinate_rmse_m=float(np.sqrt(np.mean((pred[1:]-target[1:])**2))),shape_rmse_m=float(np.sqrt(np.mean((aligned-b)**2))))
def main():
 p=read(REPORT/'protocol.json');evidence={};results={};checks=0
 def bind(path):evidence[str(path.relative_to(ROOT))]=sha(path)
 for grid,name in [('fine','coupled-friction-v13'),('coarse','coupled-surface-v13')]:
  folder=ROOT/'reports'/name;path=folder/'reserved-results.json';doc=read(path);bind(path);lookup={}
  expected={(o['id'],s) for o in p['objects'] if o['split']=='reserved' for s in p['forecast_scenarios']}
  if {(r['object'],r['scenario']) for r in doc['cases']}!=expected or len(doc['cases'])!=32:raise RuntimeError('Reserved matrix differs')
  for row in doc['cases']:
   oid,scene=row['object'],row['scenario'];base=ROOT/'runs'/name;predfile=base/'evaluation'/oid/scene/'predictions.npz';reference=ROOT/'runs/coupled-surface-v13/reference'/oid/scene/'trajectory.npz';bind(predfile);bind(reference)
   observation=base/'inference'/oid/'observations.npz';bind(observation)
   with np.load(observation,allow_pickle=False) as a:rest=a['rest'].copy()
   # All scenario rest grids differ only by their prescribed initial translation.
   # Use saved coarse oracle rest for the exact scenario, not squeeze geometry.
   coarsefile=base/'prediction'/oid/scene/'oracle/trajectory.npz';bind(coarsefile)
   with np.load(coarsefile,allow_pickle=False) as a:rest=a['rest'].copy()
   with np.load(reference,allow_pickle=False) as a:
    distances,ids=cKDTree(a['rest']).query(rest)
    if distances.max()>1e-10:raise RuntimeError('Shared node mismatch')
    target=a['frames'][:,ids].copy()
   computed={}
   with np.load(predfile,allow_pickle=False) as a:
    for family,stored in row['results'].items():
     m=metric(a[family] if family in a.files else None,target);computed[family]=m
     if m['complete']!=stored['complete']:raise RuntimeError('Completion mismatch')
     for key in ('coordinate_rmse_m','shape_rmse_m'):
      if m['complete']:np.testing.assert_allclose(m[key],stored[key],rtol=1e-10,atol=1e-12)
      elif stored[key] is not None:raise RuntimeError('Incomplete metric not null')
      checks+=1
   lookup[(oid,scene)]=computed
  results[grid]=lookup
 cohorts={}
 for cohort,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios'])]:
  keys=[k for k in results['fine'] if k[1] in scenes];summary={'expected':len(keys),'methods':{},'paired':{}}
  for label,grid,family in [('fine_fem','fine','fem'),('coarse_fem','coarse','fem'),('fine_rigid','fine','rigid'),('oracle','fine','oracle'),('fine_native','fine','native')]:
   rows=[results[grid][k][family] for k in keys];valid=[r for r in rows if r['complete']]
   summary['methods'][label]=dict(complete=len(valid),expected=len(keys),world_mm=float(np.mean([r['coordinate_rmse_m'] for r in valid])*1000) if valid else None,shape_mm=float(np.mean([r['shape_rmse_m'] for r in valid])*1000) if valid else None)
   if label!='fine_fem':
    pairs=[dict(object=k[0],scenario=k[1],fine_mm=results['fine'][k]['fem']['coordinate_rmse_m']*1000,other_mm=results[grid][k][family]['coordinate_rmse_m']*1000,gain_mm=(results[grid][k][family]['coordinate_rmse_m']-results['fine'][k]['fem']['coordinate_rmse_m'])*1000) for k in keys if results['fine'][k]['fem']['complete'] and results[grid][k][family]['complete']]
    summary['paired'][label]=dict(matched=len(pairs),expected=len(keys),mean_gain_mm=float(np.mean([x['gain_mm'] for x in pairs])) if pairs else None,cases=pairs)
  cohorts[cohort]=summary
 bind(REPORT/'protocol.json');out=dict(passed=True,scope='Thirty-two reserved forecasts, both fine and original coarse grid; no new simulation or fitting; descriptive metrics only',metric_checks=checks,cohorts=cohorts,input_sha256=evidence,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
 summary=read(REPORT/'summary.json');bind(REPORT/'summary.json')
 for cohort,v in cohorts.items():
  for ours,theirs in [('fine_fem','fine_fem'),('coarse_fem','coarse_fem'),('fine_rigid','fine_rigid'),('oracle','fine_oracle'),('fine_native','fine_native')]:
   a=v['methods'][ours];b=summary['cohorts']['reserved'][cohort]['methods'][theirs]
   if a['complete']!=b['complete'] or a['expected']!=b['expected']:raise RuntimeError('Summary completion differs')
   for key,other in [('world_mm','coordinate_mean_mm'),('shape_mm','shape_mean_mm')]:
    if a[key] is not None:np.testing.assert_allclose(a[key],b[other],rtol=1e-10,atol=1e-10)
    elif b[other] is not None:raise RuntimeError('Failed summary not null')
  for other,paired in v['paired'].items():
   cases=paired['cases'];paired.update(wins=sum(r['gain_mm']>1e-9 for r in cases),losses=sum(r['gain_mm']< -1e-9 for r in cases),ties=sum(abs(r['gain_mm'])<=1e-9 for r in cases))
  v['regressions_vs_coarse']=[r for r in v['paired']['coarse_fem']['cases'] if r['gain_mm']< -1e-9]
 parameters={}
 for grid,dirname in [('coarse','coupled-surface-v13'),('fine','coupled-friction-v13')]:
  items=[]
  for o in p['objects']:
   if o['split']!='reserved':continue
   path=ROOT/'runs'/dirname/'inference'/o['id']/'selection.json';bind(path);fit=read(path)['fem']
   items.append(dict(object=o['id'],young_pa=fit['scale'],friction=fit['friction'],relative_E_abs_error=abs(fit['scale']/o['young']-1),friction_abs_error=abs(fit['friction']-o['friction']),calibration_rmse_mm=1000*fit['calibration_rmse_m']))
  r=dict(cases=items,expected=8,complete=len(items),mean_relative_E_abs_error=float(np.mean([x['relative_E_abs_error'] for x in items])),mean_friction_abs_error=float(np.mean([x['friction_abs_error'] for x in items])),mean_calibration_rmse_mm=float(np.mean([x['calibration_rmse_mm'] for x in items])))
  stored=summary['parameter_diagnostics']['splits']['reserved']['summary'][grid+'_fem']
  for key,other in [('mean_relative_E_abs_error','mean_young_relative_abs_error'),('mean_friction_abs_error','mean_friction_abs_error'),('mean_calibration_rmse_mm','mean_calibration_rmse_mm')]:np.testing.assert_allclose(r[key],stored[other],rtol=1e-10,atol=1e-12)
  parameters[grid]=r
 coverage={}
 for family,n in [('fem',119),('native',119),('rigid',17)]:
  statuses=list((BASE/'library'/family).glob('*/complete.json'))
  if len(statuses)!=n:raise RuntimeError('Library inventory differs')
  count=0
  for path in statuses:bind(path);count+=bool(read(path)['success'])
  if summary['library_coverage'][family]['complete']!=count:raise RuntimeError('Library coverage differs')
  coverage[family]=dict(recorded=n,complete=count,failed=n-count)
 out.update(parameters=parameters,library_coverage=coverage,interpretation=['Separate fixed uniform friction-grid sensitivity study, not best-of coarse/fine future selection','All24 reserved primary and8 stress remain; two primary body-spatial failures and all stress qualification caveats remain unchanged','Material/friction point errors are post-fit descriptive diagnostics, not calibrated uncertainty or identifiability','Fine native remains an adapted material core with16/24 primary and0/8 stress complete; no full PhysTwin superiority claim','No matched fine-grid vertex comparison; increased calibration search budget explicitly255 vs75 candidates'])
 (REPORT/'final-interpretation-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'primary':cohorts['primary']['methods'],'stress':cohorts['stress']['methods'],'parameters':{k:{x:y for x,y in v.items() if x!='cases'} for k,v in parameters.items()},'library':coverage},indent=2))
if __name__=='__main__':main()
