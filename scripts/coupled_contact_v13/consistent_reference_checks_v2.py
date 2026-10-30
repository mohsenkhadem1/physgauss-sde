"""Explicit selected-reference temporal, contact and endpoint qualification."""
from pathlib import Path
import sys,json,hashlib,argparse,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.consistent_mass import ConsistentClippedFEMSystem
from scripts.coupled_contact_v13.engineered_scenarios import run,build
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare
from scripts.coupled_contact_v13.mesh_diagnostic import interpolation
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write,remaining
MATERIALS=[(6000.,.35),(3000.,.1)]
def key(s):return '-'.join(str(x) for x in s)
def task(payload):
 spec,version=payload;n,E,mu,cells,dt,h,level=spec;path=BASE/'consistent-reference-checks'/version/key(spec)
 if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
 path.mkdir(parents=True,exist_ok=True);start=time.time()
 try:
  if remaining()<=0:raise RuntimeError('Deadline')
  class Contact(ConsistentClippedFEMSystem):surface_spacing=h;friction_quadrature_level=level
  r=run(n,E,mu,(cells,cells,cells),dt,system_class=Contact)
  query=build(n,E,mu,(2,2,2)).rest;W=interpolation(r['rest'],r['tets'],query);sampled=np.einsum('qn,tnc->tqc',W,r['frames'])
  np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},sampled=sampled)
  row={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))};row.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()))
 except Exception as e:row=dict(success=False,error=repr(e))
 row.update(spec=spec,key=key(spec),path=str(path.relative_to(ROOT)),seconds=time.time()-start);write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def physics(r):return r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders']
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-map',required=True);p.add_argument('--mode',choices=['time','contact','endpoints'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--dt',type=float,default=2.5e-5);a=p.parse_args();mapping=json.loads(a.reference_map)
 names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','consistent_mass.py','engineered_scenarios.py','engineered_mesh_refinement.py','surface_qualification.py','mesh_diagnostic.py','consistent_reference_checks_v2.py']
 hashes={str(Path('scripts/coupled_contact_v13')/n):hashlib.sha256((ROOT/'scripts/coupled_contact_v13'/n).read_bytes()).hexdigest() for n in names};simulation_version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
 version=hashlib.sha256((simulation_version+json.dumps(mapping,sort_keys=True)+a.mode+str(a.dt)).encode()).hexdigest()[:12]
 pairs=[];specs=[]
 for n,cells in mapping.items():
  if a.mode=='endpoints':specs.extend((n,E,mu,cells,a.dt,.0015,1) for E in [2000.,18000.] for mu in [0.,.8]);continue
  for E,mu in MATERIALS:
   base=(n,E,mu,cells,a.dt,.0015,1)
   if a.mode=='time':refined=(n,E,mu,cells,a.dt/2,.0015,1)
   elif n in ['squeeze','edge_drop']:refined=(n,E,mu,cells,a.dt,.0015,2)
   elif n in ['curved','moving_sphere','slide']:refined=(n,E,mu,cells,a.dt,.00075,1)
   else:raise ValueError(n)
   pairs.append((base,refined));specs.extend([base,refined])
 plan=dict(scope='Selected consistent reference qualification; development only',source_sha256=hashes,reference_map=mapping,mode=a.mode,dt=a.dt,specs=specs,gates=dict(world_rms_m=.00025 if a.mode=='time' else .001,shape_rms_m=None if a.mode=='time' else .0005,body_center_rms_m=.00025 if a.mode=='time' else .001,penetration_m=.001,min_J=.02))
 write(REPORT/f'consistent-reference-{a.mode}-plan-{version}.json',plan)
 rows={};reused=[]
 # Reuse only declared consistent25us spatial trajectories with matching physical sources.
 manifests=[REPORT/'consistent-mass-diagnostic.json',*REPORT.glob('consistent-higher-refinement-progress-*.json'),*REPORT.glob('consistent-other-spatial-progress-*.json')]
 for manifest in manifests:
  if not manifest.exists():continue
  manifest_bytes=manifest.read_bytes();manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
  snapshot=REPORT/'manifest-snapshots'/(manifest_hash+'.json');snapshot.parent.mkdir(parents=True,exist_ok=True)
  if not snapshot.exists():snapshot.write_bytes(manifest_bytes)
  if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=manifest_hash:raise RuntimeError('Immutable manifest snapshot mismatch')
  d=json.loads(manifest_bytes);oldhash=d['plan']['source_sha256']
  common=[r for r in hashes if r in oldhash]
  if any(hashes[r]!=oldhash[r] for r in common):continue
  for r in d['runs']:
   s=r['spec']
   if len(s)!=4:continue
   candidate=tuple(s)+(2.5e-5,.0015,1)
   if candidate in specs:
    rr=dict(r);rr['spec']=list(candidate);rr['key']=key(candidate);rows[key(candidate)]=rr
  reused.append(dict(path=str(snapshot),original_path=str(manifest),sha256=manifest_hash))
 missing=[s for s in specs if key(s) not in rows]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  for r in ex.map(task,[(s,simulation_version) for s in missing]):rows[r['key']]=r
 checks=[]
 if a.mode=='endpoints':checks=[dict(spec=r['spec'],passed=physics(r)) for r in rows.values()]
 else:
  for sa,sb in pairs:
   aa,bb=rows[key(sa)],rows[key(sb)];c=compare(aa,bb);c.update(scenario=sa[0],young=sa[1],friction=sa[2],cells=sa[3]);g=plan['gates']
   c['passed']=c['complete'] and c['world_rms_m']<=g['world_rms_m'] and c['body_positions_finite'] and c['body_center_rms_m']<=g['body_center_rms_m'] and (g['shape_rms_m'] is None or c['shape_rms_m']<=g['shape_rms_m']) and physics(aa) and physics(bb);checks.append(c)
 unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in hashes.items())
 result=dict(plan=plan,reused_manifests=reused,runs=list(rows.values()),checks=checks,source_unchanged=unchanged,all_gates_pass=unchanged and all(c['passed'] for c in checks))
 write(REPORT/f'consistent-reference-{a.mode}-{version}.json',result)
if __name__=='__main__':main()
