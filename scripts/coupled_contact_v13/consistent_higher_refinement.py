"""Prospective higher consistent-mass refinement; scene-specific acceptance."""
from pathlib import Path
import sys,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.consistent_mass_diagnostic import task,key
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare,dense_compare
from scripts.coupled_contact_v13.campaign import REPORT,write
MATERIALS=[(6000.,.35),(3000.,.1)]
def main():
 previous_path=REPORT/'consistent-mass-diagnostic.json';previous=json.loads(previous_path.read_text());sources=dict(previous['plan']['source_sha256'])
 sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for rel,h in sources.items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed '+rel)
 version='higher-'+hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
 plan=dict(scope='Prospective development refinement, scene-specific reference candidates; temporal/full-scene gates still required',source_sha256=sources,prior_report_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),ladder=[[8,10],[10,12],[12,16]],scenarios=['edge_drop','curved'],materials=MATERIALS,dt=.000025,gates=dict(sparse_and_dense_world_rms_m=.001,sparse_and_dense_shape_rms_m=.0005,per_body_center_rms_m=.001,penetration_m=.001,min_J=.02),acceptance='Each scenario accepts the lower mesh of its first passing pair, requiring BOTH materials and all gates. Failed stages stay recorded.')
 write(REPORT/f'consistent-higher-refinement-plan-{version}.json',plan)
 rows={r['key']:r for r in previous['runs']};accepted={};checks=[]
 for low,high in plan['ladder']:
  active=[n for n in plan['scenarios'] if n not in accepted]
  if not active:break
  specs=[(n,E,mu,c) for n in active for E,mu in MATERIALS for c in [low,high] if key((n,E,mu,c)) not in rows]
  with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
   for r in ex.map(task,[(s,version) for s in specs]):rows[r['key']]=r
  stage=[]
  for n in active:
   case=[]
   for E,mu in MATERIALS:
    a,b=[rows[key((n,E,mu,c))] for c in [low,high]];c=compare(a,b,True);d=dense_compare(a,b);c.update(scenario=n,young=E,friction=mu,cells=[low,high],dense_surface=d)
    c['passed']=bool(c['complete'] and d['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and d['world_rms_m']<=.001 and d['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and all(r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders'] for r in [a,b]));case.append(c)
   stage.extend(case)
   if all(c['passed'] for c in case):accepted[n]=low
  checks.extend(stage);write(REPORT/f'consistent-higher-refinement-progress-{version}.json',dict(plan=plan,runs=list(rows.values()),checks=checks,accepted_reference_cells=accepted));print(json.dumps(stage),flush=True)
 unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in sources.items())
 write(REPORT/'consistent-higher-refinement.json',dict(plan=plan,runs=list(rows.values()),checks=checks,accepted_reference_cells=accepted,source_unchanged=unchanged,spatial_gates_pass=len(accepted)==len(plan['scenarios']) and unchanged,all_gates_pass=False,remaining_gates=['Selected candidate contact/friction/sphere integration','Selected references timestep refinement','Other scene spatial and body checks','Endpoint stress and predictor feasibility']))
if __name__=='__main__':main()
