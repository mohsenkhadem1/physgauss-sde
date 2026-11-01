"""Read-only recomputation of development qualification from saved arrays."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/physical-refinement-v15'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 evidence={};rows=[]
 def read(p):
  evidence[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
 def arrays(folder):
  c=read(folder/'complete.json');out=[]
  for file,key in [('private-state.npz','field'),('surface.npz','vertices'),('reaction-wrench.npz','forces_N'),('reaction-wrench.npz','torques_Nm')]:
   p=folder/file;assert sha(p)==c['artifact_sha256'][file];evidence[str(p.relative_to(ROOT))]=sha(p)
   with np.load(p) as z:a=z[key].copy()
   assert np.isfinite(a).all();out.append(a)
  assert np.isfinite([c['residual_N'],c['minimum_J'],c['reaction_balance_N']]).all()
  assert c['residual_N']<=1e-9 and c['minimum_J']>.1 and c['reaction_balance_N']<=1e-7
  return out
 sets=[('development-bank-initial-qualification.json','development/mechanics-jobs.json'),('development-worlds-initial-qualification.json','development/worlds-queue/manifest.json'),('response-prediction-qualification.json','response-predictions/manifest.json'),('development-quadrature-qualification.json','development/quadrature-queue/manifest.json')]
 for report_name,manifest_name in sets:
  report=read(REPORT/report_name);path=ROOT/'runs/physical-refinement-v15'/manifest_name;m=read(path);assert sha(path)==report['manifest_sha256'];g=m['gates'];byname={(j['name'],j['n']):j for j in m['jobs']}
  for group in report['comparisons']:
   name=group['name'];assert len(group['rows'])==len(m['action_names'])
   if 'quadrature' in report_name:
    j=next(j for j in m['jobs'] if j['name']==name);paths=[ROOT/j['baseline'],ROOT/j['out']]
   else:paths=[ROOT/byname[(name,n)]['out'] for n in (4,6)]
   for i,record in enumerate(group['rows']):
    assert record['action']==m['action_names'][i]
    a,b=[arrays(p/f'pose-{i:03d}') for p in paths];diff=[x-y for x,y in zip(a,b)];f=np.linalg.norm(diff[2],axis=1);t=np.linalg.norm(diff[3],axis=1)
    metrics=dict(field_rms_m=float(np.sqrt(np.mean(diff[0]**2))),field_max_m=float(abs(diff[0]).max()),surface_rms_m=float(np.sqrt(np.mean(diff[1]**2))),surface_max_m=float(abs(diff[1]).max()),force_difference_N=f.tolist(),torque_difference_Nm=t.tolist())
    for key,value in metrics.items():assert np.allclose(value,record[key],rtol=1e-9,atol=1e-12),(name,key)
    passed=all(metrics[k]<=g[k] for k in ('field_rms_m','field_max_m','surface_rms_m','surface_max_m')) and bool(np.all(f<=g['force_floor_N']+g['force_relative']*np.linalg.norm(b[2],axis=1)) and np.all(t<=g['torque_floor_Nm']+g['torque_relative']*np.linalg.norm(b[3],axis=1)))
    assert passed==record['passed'];rows.append(dict(report=report_name,name=name,action=record['action'],passed=passed))
 evidence[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 result=dict(passed=all(r['passed'] for r in rows),checks=len(rows),rows=rows,evidence_sha256=evidence,scope='Independent saved-array metrics/gates and finite solver diagnostics for all declared development spatial/quadrature comparisons. No simulations or reserved arrays read. Does not qualify renderer or continuous posterior.')
 (REPORT/'independent-mechanics-gate-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result['passed'],result['checks'])
if __name__=='__main__':main()
