"""Diagnose capsule12 timestep independently of its failed spatial gate."""
from pathlib import Path
import sys,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.capsule_reference_checks import task,physics
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare
from scripts.coupled_contact_v13.campaign import REPORT,write

def main():
 p=REPORT/'capsule-development-diagnostic.json';initial=json.loads(p.read_text());h=initial['accepted_spacing_m']
 if h is None:raise RuntimeError('Quadrature must pass')
 hashes=dict(initial['plan']['source_sha256'])
 for name in ['capsule_reference_checks.py','capsule_time_diagnostic.py']:
  path=ROOT/'scripts/coupled_contact_v13'/name;hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 version='time-diagnostic-'+hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
 plan=dict(scope='Temporal diagnosis only; does not override failed spatial evidence',source_sha256=hashes,prior_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cells=12,dt_pair=[1.25e-5,6.25e-6],spacing=h,materials=initial['plan']['materials'],gates=dict(world_rms_m=.00025,body_center_rms_m=.00025,penetration_m=.001,min_J=.02))
 write(REPORT/'capsule-time-diagnostic-plan.json',plan)
 rows=[r for r in initial['runs'] if r['spec'][2]==12 and r['spec'][3]==h]
 specs=[('edge_drop',E,mu,12,6.25e-6,h,1) for E,mu in plan['materials']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:new=list(ex.map(task,[(s,version) for s in specs]))
 checks=[]
 for (E,mu),b in zip(plan['materials'],new):
  a=next(r for r in rows if r['spec'][:2]==[E,mu]);c=compare(a,b);c.update(young=E,friction=mu,cells=12)
  c['passed']=c['complete'] and c['world_rms_m']<=.00025 and c['body_positions_finite'] and c['body_center_rms_m']<=.00025 and physics(a) and physics(b);checks.append(c)
 unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==v for r,v in hashes.items())
 write(REPORT/'capsule-time-diagnostic.json',dict(plan=plan,runs=rows+new,checks=checks,source_unchanged=unchanged,temporal_gates_pass=unchanged and all(c['passed'] for c in checks),all_gates_pass=False))
if __name__=='__main__':main()
