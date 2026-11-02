"""Read-only independent saved-field qualification and transfer audit."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/contact-transfer'
def main():
 h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rd=lambda p:json.loads(Path(p).read_text());p=rd(R/'contact-transfer-plan.json');d=rd(R/'contact-transfer-fine.json');errors=[];checks=0;evidence={}
 def ck(v,msg):
  nonlocal checks
  checks+=1
  if not v:errors.append(msg)
 def eq(a,b,msg):ck(np.allclose(a,b,rtol=1e-10,atol=1e-13),msg)
 for table in ('source_sha256','artifact_sha256'):
  for path,v in p[table].items():ck(h(ROOT/path)==v,'pin '+path)
 for path in (R/'contact-transfer-plan.json',R/'contact-transfer-fine.json'):evidence[str(path.relative_to(ROOT))]=h(path)
 ck(d['plan_sha256']==h(R/'contact-transfer-plan.json'),'plan seal')
 expected={(f,m,n,q)for f in ('mr_saturated','yeoh_polynomial')for m in ('analytic_teacher','shared_laws')for n in (6,8)for q in (16,32)}
 ck({(r['family'],r['method'],r['n'],r['q'])for r in d['rows']}==expected and len(d['rows'])==16,'exact census')
 ck(d['complete']and d['terminal']and d['completed_jobs']==16 and d['terminal_jobs']==16 and d['expected_jobs']==16 and not d['missing_keys'],'all statuses')
 arrays={};rows={}
 for row in d['rows']:
  key=row['key'];rows[key]=row;path=B/'states'/(key+'.npz');meta=path.with_suffix('.json');ck(rd(meta)==row,'row identity');ck(row['plan_sha256']==d['plan_sha256']and row['completed'],'row status');ck(h(path)==row['response_sha256'],'response seal');evidence[str(path.relative_to(ROOT))]=h(path);evidence[str(meta.relative_to(ROOT))]=h(meta)
  with np.load(path)as a:arrays[key]={k:a[k].copy()for k in a.files}
  a=arrays[key];ck(all(np.isfinite(v).all()for v in a.values()),'finite arrays');ck(a['field'].shape==(729,3)and a['forces'].shape==(4,3)and a['torques'].shape==(4,3),'array shapes')
  eq(a['frames'][-1],a['u'],'final frame')
  depth=max(x['max_penetration_m']for x in row['search129']);change=max(abs(x['max_penetration_m']-y['max_penetration_m'])for x,y in zip(row['search65'],row['search129']));under=max(x['dense_underestimate_m']for x in row['search129']);eq([depth,change,under],[row['depth_m'],row['gap_search_change_m'],row['dense_underestimate_m']],'gap aggregation')
  gate=row['residual_N']<1e-8 and row['minimum_J']>.1 and row['force_balance_N']<1e-7 and row['torque_balance_Nm']<1e-7 and change<=2e-5 and under<=2e-5 and not row['missed_contact'];ck(bool(gate)==row['numerical_gate'],'numerical gate')
 def comparison(c,gate=True):
  a,b=arrays[c['a']],arrays[c['b']];ck(np.array_equal(a['query'],b['query']),'common queries');field=np.linalg.norm(a['field']-b['field'])/np.sqrt(a['field'].size);force=np.linalg.norm(a['forces']-b['forces'],axis=1);tol=.005+.01*np.linalg.norm(b['forces'],axis=1);torque=np.linalg.norm(a['torques']-b['torques'],axis=1);depth=abs(rows[c['a']]['depth_m']-rows[c['b']]['depth_m'])
  for k,v in [('field_rms_m',field),('per_body_force_difference_N',force),('per_body_force_tolerance_N',tol),('per_body_torque_difference_Nm',torque),('depth_difference_m',depth)]:eq(v,c[k],k)
  passed=rows[c['a']]['numerical_gate']and rows[c['b']]['numerical_gate']and field<=2e-5 and np.all(force<=tol)and depth<=2e-5
  if gate:ck(bool(passed)==c['passed'],'comparison gate')
  return dict(field_rms_um=field*1e6,sphere_force_error_N=float(force[-1]),sphere_force_relative_error=float(force[-1]/np.linalg.norm(b['forces'][-1])))
 for key,v in d['qualification'].items():
  ck(len(v['checks'])==3,'three qualification checks')
  for c in v['checks']:comparison(c)
  ck(v['passed']==all(c['passed']for c in v['checks']),'qualification conjunction')
 transfer={k:comparison(v,False)for k,v in d['transfer_comparisons'].items()};ck(d['elapsed_seconds']<=p['fine']['seconds'],'stage budget')
 out=dict(passed=not errors,checks=checks,errors=errors,created_unix=time.time(),evidence_sha256=evidence,qualified={k:v['passed']for k,v in d['qualification'].items()},transfer=transfer,scope='All16 saved fields and14 comparisons independently recomputed. Oracle known descriptors, two synthetic constitutive/interface combinations with analytic normal contact; no friction/dynamic rigid/contact inference claim. Fixed-penalty two-resolution consistency, not continuum convergence. Gap searches/equilibrium diagnostics source-reviewed and hash-bound, not re-solved. Transfer errors descriptive, not qualification thresholds.')
 (R/'contact-transfer-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('passed','checks','errors','transfer')}))
if __name__=='__main__':main()
