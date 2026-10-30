"""Single prospectively authorized capsule12/16 fallback; no further ladder."""
from pathlib import Path
import sys,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.capsule_diagnostic import task,physics,key
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare,dense_compare
from scripts.coupled_contact_v13.campaign import REPORT,write

def main():
 initial_path=REPORT/'capsule-development-diagnostic.json';initial=json.loads(initial_path.read_text())
 amendment_path=REPORT/'capsule-spatial-fallback-amendment.json';amendment=json.loads(amendment_path.read_text())
 if initial['accepted_spacing_m'] is None:raise RuntimeError('Quadrature must qualify before spatial fallback')
 if initial['diagnostic_gates_pass']:raise RuntimeError('Initial spatial pair passed; fallback not authorized')
 hashes=dict(initial['plan']['source_sha256']);rel=str(Path(__file__).relative_to(ROOT));hashes[rel]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 if not all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in hashes.items()):raise RuntimeError('Source changed')
 version='fallback-'+hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12];h=initial['accepted_spacing_m'];materials=initial['plan']['materials']
 plan=dict(initial_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),amendment_sha256=hashlib.sha256(amendment_path.read_bytes()).hexdigest(),source_sha256=hashes,cells=[12,16],dt=1.25e-5,spacing=h,materials=materials,gates=amendment['gates'])
 write(REPORT/'capsule-fallback-plan.json',plan)
 rows=[r for r in initial['runs'] if r['spec'][2]==12 and r['spec'][3]==h]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  rows.extend(ex.map(task,[((E,mu,16,h),version) for E,mu in materials]))
 indexed={key(r['spec']):r for r in rows};checks=[]
 for E,mu in materials:
  a,b=[indexed[key((E,mu,n,h))] for n in [12,16]];c=compare(a,b,True);d=dense_compare(a,b);c.update(young=E,friction=mu,cells=[12,16],dense_surface=d)
  c['passed']=c['complete'] and d['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and d['world_rms_m']<=.001 and d['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and physics(a) and physics(b);checks.append(c)
 unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in hashes.items())
 write(REPORT/'capsule-spatial-fallback.json',dict(plan=plan,runs=rows,checks=checks,source_unchanged=unchanged,diagnostic_gates_pass=unchanged and all(c['passed'] for c in checks),all_gates_pass=False,remaining_gates=initial['remaining_gates']))
if __name__=='__main__':main()
