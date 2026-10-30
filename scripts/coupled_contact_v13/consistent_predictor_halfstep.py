"""Additional predictor12.5 versus6.25 microsecond development checks."""
from pathlib import Path
import sys,json,hashlib,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.consistent_qualification import task,key,compare,physics
from scripts.coupled_contact_v13.campaign import REPORT,write,remaining
CASES=[(n,E,mu) for n in ['squeeze','slide','moving_sphere','curved','edge_drop'] for E,mu in [(6000.,.35),(3000.,.1)]]
def waitfor(p):
 while not p.exists():
  if remaining()<=0:raise RuntimeError('Deadline')
  time.sleep(5)

def main():
 names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','consistent_mass.py','engineered_scenarios.py','consistent_qualification.py','surface_qualification.py','consistent_predictor_halfstep.py']
 sources={str((Path('scripts/coupled_contact_v13')/n)):hashlib.sha256((ROOT/'scripts/coupled_contact_v13'/n).read_bytes()).hexdigest() for n in names};version='predictor-half-'+hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
 plan=dict(scope='Prospective selected predictor4 step12.5us verification; no reserved data',source_sha256=sources,cases=CASES,timesteps=[.000025,.0000125,.00000625],cells=4,friction_level=1,sphere_spacing_m=.0015,gates=dict(soft_coordinate_rms_m=.00025,per_body_coordinate_rms_m=.00025,refinement_ratio=.8,penetration_m=.001,min_J=.02))
 write(REPORT/f'consistent-predictor-halfstep-plan-{version}.json',plan)
 waitfor(REPORT/'engineered-mesh-refinement.json') # frees the eight archival workers
 specs=[(n,E,mu,6.25e-6,.0015,1) for n,E,mu in CASES]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:rows=list(ex.map(task,[(s,version) for s in specs]))
 bp=REPORT/'consistent-development-qualification.json';waitfor(bp);base=json.loads(bp.read_text());lut={r['key']:r for r in base['runs']+rows};checks=[]
 for n,E,mu in CASES:
  kk=[key((n,E,mu,dt,.0015,1)) for dt in [2.5e-5,1.25e-5,6.25e-6]]
  if any(k not in lut for k in kk):checks.append(dict(scenario=n,young=E,friction=mu,complete=False,passed=False,reason='Base qualification did not produce required steps'));continue
  aa,bb,cc=[lut[k] for k in kk];c=compare(bb,cc);coarse=compare(aa,cc);ratio=c.get('world_rms_m',float('inf'))/max(coarse.get('world_rms_m',0),1e-20)
  c.update(scenario=n,young=E,friction=mu,ratio=ratio);c['passed']=c['complete'] and c['world_rms_m']<=.00025 and c['body_positions_finite'] and c['body_center_rms_m']<=.00025 and ratio<=.8 and all(physics(r) for r in [aa,bb,cc]);checks.append(c)
 unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in sources.items())
 write(REPORT/'consistent-predictor-halfstep.json',dict(plan=plan,base_report_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),runs=rows,checks=checks,source_unchanged=unchanged,all_gates_pass=unchanged and all(c['passed'] for c in checks)))
if __name__=='__main__':main()
