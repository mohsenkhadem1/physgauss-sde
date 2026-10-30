"""Prospective rounded-rail development control; sharp-edge outputs preserved."""
from pathlib import Path
import sys,json,hashlib,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.capsule_development import run,build
from scripts.coupled_contact_v13.consistent_mass import ConsistentClippedFEMSystem
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare,dense_compare
from scripts.coupled_contact_v13.mesh_diagnostic import interpolation,shape
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write,remaining
MATERIALS=[(6000.,.35),(3000.,.1)]
def key(s):return '-'.join(str(x) for x in s)
def task(payload):
 spec,version=payload;E,mu,n,h=spec;path=BASE/'capsule-diagnostic'/version/key(spec)
 if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
 path.mkdir(parents=True,exist_ok=True);start=time.time()
 try:
  if remaining()<=0:raise RuntimeError('Deadline')
  class Contact(ConsistentClippedFEMSystem):surface_spacing=h;friction_quadrature_level=1
  r=run('edge_drop',E,mu,(n,n,n),1.25e-5,system_class=Contact)
  query=build('edge_drop',E,mu,(2,2,2)).rest;W=interpolation(r['rest'],r['tets'],query);sampled=np.einsum('qn,tnc->tqc',W,r['frames'])
  np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},sampled=sampled)
  row={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))};row.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()),max_kabsch_deformation_m=max(shape(query,x) for x in sampled))
 except Exception as e:row=dict(success=False,error=repr(e))
 row.update(spec=spec,key=key(spec),path=str(path.relative_to(ROOT)),seconds=time.time()-start);write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def physics(r):return r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders']
def main():
 names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','consistent_mass.py','engineered_scenarios.py','capsule_development.py','engineered_mesh_refinement.py','surface_qualification.py','mesh_diagnostic.py','capsule_diagnostic.py']
 hashes={str(Path('scripts/coupled_contact_v13')/n):hashlib.sha256((ROOT/'scripts/coupled_contact_v13'/n).read_bytes()).hexdigest() for n in names};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
 plan=dict(scope='Prospective engineering-control diagnostic, not repair/relabeling of sharp-edge results or a production switch',source_sha256=hashes,geometry=dict(radius_m=.008,segment_length_m=.124,center_m=[.01,0,.042],unchanged_block_center_m=[.03,0,.09]),materials=MATERIALS,cells=[8,12],dt=.0000125,friction_level=1,quadrature_ladder_m=[[.0015,.00075],[.00075,.000375]],gates=dict(world_rms_m=.001,shape_rms_m=.0005,per_body_rms_m=.001,penetration_m=.001,min_J=.02),decision='Interpret spatial8/12 comparison only after both materials pass capsule quadrature at cells12. All failed rungs retained. Further time/endpoints/contact gates and independent audit required before production.')
 write(REPORT/f'capsule-diagnostic-plan-{version}.json',plan);rows={};quadrature=[];spacing=None
 def obtain(specs):
  jobs=[s for s in specs if key(s) not in rows]
  with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
   for r in ex.map(task,[(s,version) for s in jobs]):rows[r['key']]=r
 for low,high in plan['quadrature_ladder_m']:
  specs=[(E,mu,12,h) for E,mu in MATERIALS for h in [low,high]]+[(E,mu,8,low) for E,mu in MATERIALS];obtain(specs);stage=[]
  for E,mu in MATERIALS:
   a,b=[rows[key((E,mu,12,h))] for h in [low,high]];c=compare(a,b);c.update(young=E,friction=mu,spacings=[low,high]);c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and physics(a) and physics(b);stage.append(c)
  quadrature.extend(stage);write(REPORT/f'capsule-diagnostic-progress-{version}.json',dict(plan=plan,runs=list(rows.values()),quadrature_checks=quadrature));print(json.dumps(stage),flush=True)
  if all(c['passed'] for c in stage):spacing=low;break
 spatial=[]
 if spacing is not None:
  for E,mu in MATERIALS:
   a,b=[rows[key((E,mu,n,spacing))] for n in [8,12]];c=compare(a,b,True);d=dense_compare(a,b);c.update(young=E,friction=mu,cells=[8,12],dense_surface=d);c['passed']=c['complete'] and d['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and d['world_rms_m']<=.001 and d['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and physics(a) and physics(b);spatial.append(c)
 unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in hashes.items())
 write(REPORT/'capsule-development-diagnostic.json',dict(plan=plan,runs=list(rows.values()),quadrature_checks=quadrature,spatial_checks=spatial,accepted_spacing_m=spacing,source_unchanged=unchanged,diagnostic_gates_pass=unchanged and spacing is not None and all(c['passed'] for c in spatial),all_gates_pass=False,remaining_gates=['Chosen meshes timestep','Material-range endpoints','Floor friction integration','Independent audit','Predictor oracle bias']))
if __name__=='__main__':main()
