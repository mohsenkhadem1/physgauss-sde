"""Read-only original reserved forecast audit; fixed24 primary and8 stress."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from scripts.coupled_contact_v13.fine_development_forecast_audit import metric
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=read(REPORT/'protocol.json');saved=read(REPORT/'reserved-results.json');qualification=read(REPORT/'reference-inventory-independent-audit.json');evidence={};checks=0
 def bind(path):evidence[str(path.relative_to(ROOT))]=sha(path)
 expected={(o['id'],s) for o in p['objects'] if o['split']=='reserved' for s in p['forecast_scenarios']}
 if len(saved['cases'])!=32 or {(r['object'],r['scenario']) for r in saved['cases']}!=expected:raise RuntimeError('Fixed reserved case matrix differs')
 if not qualification['audit_passed']:raise RuntimeError('Reference audit not passed')
 for rel,h in qualification['input_sha256'].items():
  if sha(ROOT/rel)!=h:raise RuntimeError('Reference audit stale')
 q={(r['object'],r['scenario']):r['qualification'] for r in qualification['cases']};rows=[]
 for stored in saved['cases']:
  oid,scene=stored['object'],stored['scenario'];reference=BASE/'reference'/oid/scene/'trajectory.npz';oracle=BASE/'prediction'/oid/scene/'oracle/trajectory.npz';predfile=BASE/'evaluation'/oid/scene/'predictions.npz'
  for path in (reference,oracle,predfile):bind(path)
  with np.load(oracle,allow_pickle=False) as a:rest=a['rest'].copy()
  with np.load(reference,allow_pickle=False) as a:
   distance,ids=cKDTree(a['rest']).query(rest)
   if distance.max()>1e-10:raise RuntimeError('Shared nodes mismatch')
   target=a['frames'][:,ids].copy();body=a['centers'].copy()
  result={}
  with np.load(predfile,allow_pickle=False) as a:
   for name,value in stored['results'].items():
    result[name]=metric(a[name] if name in a.files else None,target)
    if result[name]['complete']!=value['complete']:raise RuntimeError('Completion mismatch')
    for key in ('coordinate_rmse_m','shape_rmse_m'):
     if result[name]['complete']:np.testing.assert_allclose(result[name][key],value[key],rtol=1e-10,atol=1e-12)
     elif value[key] is not None:raise RuntimeError('Incomplete metric not null')
     checks+=1
    if 'dynamic_body_position_rmse_m' in value:
     path=BASE/'prediction'/oid/scene/name/'trajectory.npz';bind(path)
     with np.load(path,allow_pickle=False) as b:predbody=b['centers'].copy()
     bodyerror=float(np.sqrt(np.mean(np.sum((predbody[1:,-1]-body[1:,-1])**2,axis=1))))
     np.testing.assert_allclose(bodyerror,value['dynamic_body_position_rmse_m'],rtol=1e-10,atol=1e-12);result[name]['dynamic_body_position_rmse_m']=bodyerror;checks+=1
  fq,cq=q[(oid,scene)]['qualified'],q[(oid,'squeeze')]['qualified']
  if stored['reference_qualified']!=fq or stored['calibration_reference_qualified']!=cq or stored['joint_reference_qualified']!=(fq and cq):raise RuntimeError('Reference qualification flags differ')
  rows.append(dict(object=oid,scenario=scene,reference_qualified=fq,calibration_reference_qualified=cq,joint_reference_qualified=fq and cq,results=result))
 cohorts={}
 for cohort,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios'])]:
  group=[r for r in rows if r['scenario'] in scenes];out=dict(expected=len(group),reference_qualified=sum(r['reference_qualified'] for r in group),joint_qualified=sum(r['joint_reference_qualified'] for r in group),methods={},paired={})
  for name in saved['cases'][0]['results']:
   values=[r['results'][name] for r in group if r['results'][name]['complete']]
   out['methods'][name]=dict(complete=len(values),expected=len(group),world_mm=float(np.mean([v['coordinate_rmse_m'] for v in values])*1000) if values else None,shape_mm=float(np.mean([v['shape_rmse_m'] for v in values])*1000) if values else None,conditional_on_completion=len(values)!=len(group))
   if name=='fem':continue
   pair={}
   for key in ('coordinate_rmse_m','shape_rmse_m'):
    cases=[dict(object=r['object'],scenario=r['scenario'],reference_qualified=r['reference_qualified'],fem_mm=1000*r['results']['fem'][key],comparator_mm=1000*r['results'][name][key],gain_mm=1000*(r['results'][name][key]-r['results']['fem'][key])) for r in group if r['results']['fem']['complete'] and r['results'][name]['complete']]
    pair[key]=dict(matched=len(cases),expected=len(group),gain_mm=float(np.mean([r['gain_mm'] for r in cases])) if cases else None,wins=sum(r['gain_mm']>1e-9 for r in cases),losses=sum(r['gain_mm']< -1e-9 for r in cases),ties=sum(abs(r['gain_mm'])<=1e-9 for r in cases),cases=cases)
   out['paired'][name]=pair
  cohorts[cohort]=out
 for name in ('protocol.json','reserved-results.json','reference-inventory-independent-audit.json'):bind(REPORT/name)
 sources=[Path(__file__),ROOT/'scripts/coupled_contact_v13/fine_development_forecast_audit.py']
 out=dict(passed=True,scope='Original reserved32 cases; independent saved-array world/shape/body metric reconstruction and reference flags; no fitting or additional simulation; fixed24primary/8stress, no qualification filtering',metric_checks=checks,cohorts=cohorts,cases=rows,input_sha256=evidence,source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},interpretation='World error gains need not imply improved deformation: compare rigid-aligned shape separately. Incomplete baselines are conditional and paired denominators explicit. Stress references remain unqualified.')
 (REPORT/'reserved-forecast-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({co:dict(expected=v['expected'],reference_qualified=v['reference_qualified'],methods=v['methods'],fem_vs_rigid={k:{x:y for x,y in z.items() if x!='cases'} for k,z in v['paired']['rigid'].items()}) for co,v in cohorts.items()},indent=2))
if __name__=='__main__':main()
