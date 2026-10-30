"""Prospective endpoint feasibility screen, not held-out evaluation."""
from pathlib import Path
import sys,json,hashlib,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.engineered_scenarios import run
from scripts.coupled_contact_v13.consistent_mass import ConsistentClippedFEMSystem as ClippedSurfaceFEMSystem
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write,remaining
class Contact(ClippedSurfaceFEMSystem):surface_spacing=.0015;friction_quadrature_level=1
SPECS=[(name,E,mu,n) for name in ['squeeze','edge_drop'] for E in [2000.,18000.] for mu in [0.,.8] for n in [4]]
def task(payload):
 spec,version=payload;name,E,mu,n=spec;start=time.time();key=f'{name}-E{E:g}-mu{mu:g}-n{n}';path=BASE/'consistent-predictor-endpoint-screen'/version/key;path.mkdir(parents=True,exist_ok=True)
 try:
  if remaining()<=0:raise RuntimeError('Deadline')
  r=run(name,E,mu,(n,n,n),1.25e-5,system_class=Contact)
  np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
  row={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))};row.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()))
  row['passed']=row['min_j']>.02 and row['max_penetration_m']<=.001 and row['finite_colliders']
 except Exception as e:row=dict(success=False,passed=False,error=repr(e))
 row.update(spec=spec,path=str(path.relative_to(ROOT)),seconds=time.time()-start);write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def main():
 sources=[ROOT/'scripts/coupled_contact_v13'/n for n in ['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','engineered_scenarios.py','consistent_endpoint_screen.py','consistent_mass.py']]
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
 plan=dict(scope='Prospective numerical endpoint screen; no reserved objects',source_sha256=hashes,specs=SPECS,edge_center_x_m=.03,stiffness_N_per_m_reference_patch=76800.,reference_patch_area_m2=.0004,dt=.0000125,sphere_spacing_m=.0015,friction_level=1,gates=dict(penetration_m=.001,min_J=.02,finite_colliders=True))
 write(REPORT/f'consistent-predictor-endpoint-screen-plan-{version}.json',plan)
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:rows=list(ex.map(task,[(s,version) for s in SPECS]))
 unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
 write(REPORT/'consistent-predictor-endpoint-screen.json',dict(plan=plan,runs=rows,source_unchanged=unchanged,all_gates_pass=unchanged and all(r['passed'] for r in rows)))
if __name__=='__main__':main()
