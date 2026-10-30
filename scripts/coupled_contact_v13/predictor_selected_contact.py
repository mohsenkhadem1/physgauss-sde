"""Exact selected12.5us predictor contact refinement, using hashed baselines."""
from pathlib import Path
import sys,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.consistent_reference_checks_v2 import task,physics
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare
from scripts.coupled_contact_v13.campaign import REPORT,write

def main():
 p=REPORT/'consistent-development-qualification.json';base=json.loads(p.read_text());hashes=dict(base['plan']['source_sha256'])
 for name in ['predictor_selected_contact.py','consistent_reference_checks_v2.py']:
  f=ROOT/'scripts/coupled_contact_v13'/name;hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 if not all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items()):raise RuntimeError('Historical source mismatch')
 version='selected-contact-'+hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
 cases=[(n,E,mu) for n in ['squeeze','edge_drop','curved','moving_sphere'] for E,mu in [(6000.,.35),(3000.,.1)]]
 specs=[(n,E,mu,4,1.25e-5,.0015 if n in ['squeeze','edge_drop'] else .00075,2 if n in ['squeeze','edge_drop'] else 1) for n,E,mu in cases]
 rows=[]
 for n,E,mu in cases:
  matches=[r for r in base['runs'] if r['spec']==[n,E,mu,1.25e-5,.0015,1]]
  if len(matches)!=1:raise RuntimeError('Require unique identical selected baseline')
  rows.extend(matches)
 plan=dict(scope='Exact selected predictor contact checks; old25us quadrature evidence preserved separately',source_sha256=hashes,base_report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cells=4,dt=1.25e-5,baseline_spacing=.0015,baseline_friction_level=1,cases=cases,specs=specs,gates=dict(world_rms_m=.001,shape_rms_m=.0005,body_center_rms_m=.001,penetration_m=.001,min_J=.02))
 write(REPORT/'predictor-selected-contact-plan.json',plan)
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:new=list(ex.map(task,[(s,version) for s in specs]))
 checks=[]
 for (n,E,mu),a,b in zip(cases,rows,new):
  c=compare(a,b);c.update(scenario=n,young=E,friction=mu,type='friction' if n in ['squeeze','edge_drop'] else 'sphere')
  c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and physics(a) and physics(b);checks.append(c)
 unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
 write(REPORT/'predictor-selected-contact.json',dict(plan=plan,runs=rows+new,checks=checks,source_unchanged=unchanged,all_gates_pass=unchanged and all(c['passed'] for c in checks)))
if __name__=='__main__':main()
