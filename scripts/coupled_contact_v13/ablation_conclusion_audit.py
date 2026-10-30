"""Read-only saved-array audit of matched vertex/nominal supplements."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from scripts.coupled_contact_v13.fine_development_forecast_audit import metric
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=read(REPORT/'protocol.json');a=read(REPORT/'ablation-results.json');summary=read(REPORT/'summary.json');evidence={};mainrows={};checks=0
 def bind(path):evidence[str(path.relative_to(ROOT))]=sha(path)
 for split in ('development','reserved'):
  path=REPORT/f'{split}-results.json';bind(path);mainrows.update({(r['object'],r['scenario']):r for r in read(path)['cases']})
 if len(a['cases'])!=40 or {(r['object'],r['scenario']) for r in a['cases']}!=set(mainrows):raise RuntimeError('Full40 matrix mismatch')
 computed=[]
 for row in a['cases']:
  oid,scene=row['object'],row['scenario'];old=mainrows[(oid,scene)]
  if row['joint_reference_qualified']!=old['joint_reference_qualified']:raise RuntimeError('Qualification mismatch')
  reference=BASE/'reference'/oid/scene/'trajectory.npz';oracle=BASE/'prediction'/oid/scene/'oracle/trajectory.npz';bind(reference);bind(oracle)
  with np.load(oracle,allow_pickle=False) as d:rest=d['rest'].copy()
  with np.load(reference,allow_pickle=False) as d:
   distances,ids=cKDTree(d['rest']).query(rest)
   if distances.max()>1e-10:raise RuntimeError('Shared nodes mismatch')
   target=d['frames'][:,ids].copy();centers=d['centers'].copy()
  results={}
  for family,path in [('vertex',BASE/'ablation/prediction'/oid/scene),('nominal',BASE/'ablation/nominal'/scene)]:
   bind(path/'complete.json');status=read(path/'complete.json');pred=None
   if status['success']:
    bind(path/'trajectory.npz')
    with np.load(path/'trajectory.npz',allow_pickle=False) as d:pred=d['frames'].copy();body=d['centers'].copy()
   got=metric(pred,target);stored=row['results'][family]
   if got['complete']!=stored['complete']:raise RuntimeError('Completion mismatch')
   for key in ('coordinate_rmse_m','shape_rmse_m'):
    if got['complete']:np.testing.assert_allclose(got[key],stored[key],rtol=1e-10,atol=1e-12)
    elif stored[key] is not None:raise RuntimeError('Failed metric not null')
    checks+=1
   if got['complete']:
    gain=(got['coordinate_rmse_m']-old['results']['fem']['coordinate_rmse_m'])*1000
    np.testing.assert_allclose(gain,row['surface_fem_gain_mm'][family],rtol=1e-10,atol=1e-10)
    if scene=='moving_sphere':np.testing.assert_allclose(np.sqrt(np.mean(np.sum((body[1:,-1]-centers[1:,-1])**2,axis=1))),stored['dynamic_body_position_rmse_m'],rtol=1e-10,atol=1e-12)
   results[family]=got
  computed.append(dict(object=oid,scenario=scene,split=row['split'],results=results,fem=old['results']['fem'],joint_reference_qualified=row['joint_reference_qualified']))
 cohorts={}
 for split in ('development','reserved'):
  cohorts[split]={}
  for cohort,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios']),('full',p['forecast_scenarios'])]:
   rows=[r for r in computed if r['split']==split and r['scenario'] in scenes];record={'expected':len(rows),'joint_reference_qualified':sum(r['joint_reference_qualified'] for r in rows),'methods':{}}
   for family in ('vertex','nominal'):
    valid=[r for r in rows if r['results'][family]['complete']];gain=[(r['results'][family]['coordinate_rmse_m']-r['fem']['coordinate_rmse_m'])*1000 for r in valid if r['fem']['complete']]
    v=dict(complete=len(valid),expected=len(rows),world_mm=float(np.mean([r['results'][family]['coordinate_rmse_m'] for r in valid])*1000) if valid else None,shape_mm=float(np.mean([r['results'][family]['shape_rmse_m'] for r in valid])*1000) if valid else None,paired=len(gain),surface_fem_gain_mm=float(np.mean(gain)) if gain else None,fem_wins=sum(x>1e-9 for x in gain),fem_losses=sum(x< -1e-9 for x in gain),ties=sum(abs(x)<=1e-9 for x in gain));record['methods'][family]=v
    saved=a['forecast_cohorts'][split][cohort]['methods'][family]
    for key,val in [('complete',v['complete']),('mean_coordinate_mm',v['world_mm']),('surface_fem_gain_mm',v['surface_fem_gain_mm'])]:np.testing.assert_allclose(saved[key],val,rtol=1e-10,atol=1e-10)
   cohorts[split][cohort]=record
 for split in cohorts:
  for family,v in cohorts[split]['full']['methods'].items():
   s=summary['matched_ablation']['splits'][split][family]
   for key,value in [('mean_coordinate_mm',v['world_mm']),('mean_shape_mm',v['shape_mm']),('mean_fem_gain_mm',v['surface_fem_gain_mm']),('fem_wins',v['fem_wins']),('fem_losses',v['fem_losses'])]:np.testing.assert_allclose(s[key],value,rtol=1e-10,atol=1e-10)
 for name in ('protocol.json','ablation-results.json','summary.json','RESULTS.txt','supplementary-protocol.json'):bind(REPORT/name)
 caveat='The matched vertex comparator shares scenes, coarse mesh, stiffness, observations and fine surface-contact reference, but changes both contact sampling and effective damping mass (nodal versus surface-area distributed). The surface-formulation reference also gives no independent solver ground truth. Results cannot isolate pure quadrature improvement.'
 sources=[Path(__file__),ROOT/'scripts/coupled_contact_v13/fine_development_forecast_audit.py',ROOT/'scripts/coupled_contact_v13/surface_ablation.py',ROOT/'scripts/coupled_contact_v13/surface_report.py']
 out=dict(passed=True,metric_checks=checks,scope='All40 matched supplement forecasts independently recomputed; no simulations or tuning',cohorts=cohorts,input_sha256=evidence,source_sha256={str(s.relative_to(ROOT)):sha(s) for s in sources},interpretation=caveat,limitations=['Nominal E6000/mu.35 fixed across objects; fitted-vs-nominal evaluates calibration utility, not same parameter condition','All24 primary cases retained despite2 reference body-spatial failures','All8 reserved sharp-edge references remain development/per-case unqualified; descriptive stress only','No comparison with finer-friction search inferred here'])
 (REPORT/'ablation-conclusion-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(cohorts['reserved'],indent=2))
if __name__=='__main__':main()
