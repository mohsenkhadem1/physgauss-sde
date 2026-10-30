"""Development-only consistent mass spatial diagnostic; not a reference freeze."""
from pathlib import Path
import sys,json,hashlib,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.consistent_mass import ConsistentClippedFEMSystem
from scripts.coupled_contact_v13.engineered_scenarios import run,build
from scripts.coupled_contact_v13.engineered_mesh_refinement import dense_compare,compare
from scripts.coupled_contact_v13.mesh_diagnostic import interpolation,shape
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write,remaining
CASES=[(n,E,mu) for n in ['edge_drop','curved'] for E,mu in [(6000.,.35),(3000.,.1)]]
class Contact(ConsistentClippedFEMSystem):surface_spacing=.0015;friction_quadrature_level=1
def key(s):return f'{s[0]}-E{s[1]:g}-mu{s[2]:g}-n{s[3]}'
def task(payload):
 spec,version=payload;n,E,mu,cells=spec;start=time.time();path=BASE/'consistent-mass-diagnostic'/version/key(spec);path.mkdir(parents=True,exist_ok=True)
 try:
  if remaining()<=0:raise RuntimeError('Deadline')
  r=run(n,E,mu,(cells,cells,cells),2.5e-5,system_class=Contact)
  query=build(n,E,mu,(2,2,2)).rest;W=interpolation(r['rest'],r['tets'],query);sampled=np.einsum('qn,tnc->tqc',W,r['frames'])
  np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},sampled=sampled)
  row={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))};row.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()),max_kabsch_deformation_m=max(shape(query,x) for x in sampled))
 except Exception as e:row=dict(success=False,error=repr(e))
 row.update(spec=spec,key=key(spec),path=str(path.relative_to(ROOT)),seconds=time.time()-start);write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def main():
 names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','consistent_mass.py','engineered_scenarios.py','engineered_mesh_refinement.py','surface_qualification.py','mesh_diagnostic.py','consistent_mass_diagnostic.py']
 sources=[ROOT/'scripts/coupled_contact_v13'/n for n in names];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
 specs=[(n,E,mu,c) for n,E,mu in CASES for c in [4,6,8]]
 plan=dict(scope='Development diagnostic only; no reference selection or held-out access',source_sha256=hashes,specs=specs,dt=.000025,friction_level=1,sphere_spacing_m=.0015,comparison_gates=dict(world_rms_m=.001,shape_rms_m=.0005,body_center_rms_m=.001,penetration_m=.001,min_J=.02))
 write(REPORT/f'consistent-mass-diagnostic-plan-{version}.json',plan)
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:rows=list(ex.map(task,[(s,version) for s in specs]))
 lut={r['key']:r for r in rows};checks=[]
 for n,E,mu in CASES:
  for low,high in [(4,6),(6,8)]:
   a,b=[lut[key((n,E,mu,c))] for c in [low,high]];c=compare(a,b,True);c.update(scenario=n,young=E,friction=mu,cells=[low,high]);d=dense_compare(a,b);c['dense_surface']=d
   c['passed']=c['complete'] and d['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and d['world_rms_m']<=.001 and d['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and all(r['min_j']>.02 and r['max_penetration_m']<=.001 for r in [a,b]);checks.append(c)
 unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
 write(REPORT/'consistent-mass-diagnostic.json',dict(plan=plan,runs=rows,checks=checks,source_unchanged=unchanged))
if __name__=='__main__':main()
