"""Prospective deterministic 40-state static catalog; NO physical protocol yet.

No random draws, sensors, observation noise, object roles or posterior data.
Importing this module runs neither mechanics nor GPU initialization.
"""
from pathlib import Path,PurePosixPath
import argparse,json,os,time,subprocess,sys
import numpy as np
from experiments import run_campaign_c38_graded_family_warp as immutable
from experiments.run_campaign_c38_static_prequalification import cases as original_cases
from experiments.campaign_owned_gpu_budget_v02 import run_owned_command
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/campaign_c38_nonlinear_table_catalog.py'
TEST='tests/test_campaign_c38_nonlinear_table_catalog.py'
GPUS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b','GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','GPU-571378aa-c44e-ffb8-b346-1e409f9821f9')
TABLE_ROOT='runs/gaussian-campaign-v01/c38-nonlinear-table-v01'
METADATA_ROOT='runs/operator-dev/c38-nonlinear-table-v01'
FORCES=((-2.,'down2'),(0.,'gravity'),(2.,'up2'),(1.,'target_up1'))

def repo_path(name):
 if not isinstance(name,str) or not name or '\\' in name:raise ValueError('Canonical repository-relative path required')
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('Canonical repository-relative path required')
 result=(ROOT/name).resolve()
 if not result.is_relative_to(ROOT):raise ValueError('Repository escape forbidden')
 return result

def catalog():
 original={c['id']:c for c in original_cases()};rows=[];lo,hi=75000.,150000.
 materials=[('Elo',lo,0.,'TABLE_NODE'),('ElogQ1',float(lo*(hi/lo)**.25),.25,'WITHHELD_INTERPOLATION_VALIDATION'),('Emid',float(np.sqrt(lo*hi)),.5,'TABLE_NODE'),('ElogQ3',float(lo*(hi/lo)**.75),.75,'WITHHELD_INTERPOLATION_VALIDATION'),('Ehi',hi,1.,'TABLE_NODE')]
 for material_id,E,fraction,use in materials:
  for support in immutable.SUPPORTS:
   for force,force_id in FORCES:
    case_id=('interior-gravity__'+support if material_id=='Emid' and force==0 else material_id+'__'+support+'__'+force_id)
    old=original.get(case_id)
    row=dict(id=case_id,young_Pa=E,support=support,Fz_N=force,kind=old['kind'] if old else 'nonlinear_table_new',material_id=material_id,log_material_fraction=fraction,material_use=use,force_id=force_id,reused_original14_case=old is not None)
    if old and any(row[k]!=old[k] for k in ['id','young_Pa','support','Fz_N','kind']):raise ValueError('Existing fourteen physical cases must match exactly')
    rows.append(row)
 if len(rows)!=40 or len({r['id'] for r in rows})!=40 or sum(r['reused_original14_case'] for r in rows)!=14:raise AssertionError('Exact 40state/14reuse/26new design required')
 return rows

def phase_plan(allowed_cores):
 cores=sorted(set(allowed_cores))
 if len(cores)<6:raise ValueError('Three future disjoint CPU2 assignments require six permitted cores')
 new=[r for r in catalog() if not r['reused_original14_case']];phases=[]
 for index,gpu in enumerate(GPUS):
  selected=new[index::3];name='gpu'+str(index)
  phases.append(dict(id=name,gpu_uuid=gpu,CPU_affinity_cores=cores[2*index:2*index+2],case_ids=[r['id'] for r in selected],cases=selected,output=TABLE_ROOT+'/'+name,owned_metadata_directory=METADATA_ROOT+'/'+name,new_case_count=len(selected),new_resolution_count=2*len(selected),status='PROSPECTIVE_ASSIGNMENT_NOT_GPU_LEASE_OR_LAUNCH_AUTHORIZATION'))
 return phases

def source_pins():return {name:immutable.sha(ROOT/name) for name in sorted(set(immutable.source_closure())|{SOURCE,TEST})}
def checked_record(r):
 p=repo_path(r['path'])
 if immutable.sha(p)!=r['sha256']:raise ValueError('Changed hashed input '+r['path'])
 return p

def check_completed_reuse(case,source_result,external_cached_result,certified_original_Torch_results=()):
 """Normalize one completed case's32/40fields, using metadata/hash checks only."""
 a=checked_record(source_result);b=checked_record(external_cached_result)
 if source_result['sha256']!=external_cached_result['sha256'] or a.read_bytes()!=b.read_bytes():raise ValueError('Independent completed case-byte cache required')
 result=json.loads(a.read_text());old=result.get('case',{})
 if any(old.get(k)!=case[k] for k in ['id','young_Pa','support','Fz_N']):raise ValueError('Wrong physical reuse case')
 if not result.get('completed') or not result.get('passed') or not result.get('physics_passed'):raise ValueError('Completed spatial and physical gate required for reuse')
 meshes=[r for r in result.get('meshes',[]) if r.get('cells') in [list(x) for x in immutable.MESHES]]
 if len(meshes)!=2 or {tuple(r['cells']) for r in meshes}!=set(immutable.MESHES):raise ValueError('Exact unique32/40 pair required')
 for m in meshes:
  if not m.get('completed') or not m.get('physics_passed'):raise ValueError('Original physical checks required')
  if not m.get('all_independent_original_Torch_linear_checks_passed') and source_result not in certified_original_Torch_results:raise ValueError('IndependentWarpcheck or exactcertifiedoldTorchbackend required')
  checked_record(m['capture'])
 values=result.get('conditional_spatial_indicators_m',[])
 if not values or values[-1]>immutable.GATES['conditional_actual_h_spatial_indicator_m']:raise ValueError('Unchanged20um conditional spatial gate required')
 return dict(original_Torch_backend_reuse=source_result in certified_original_Torch_results,case_id=case['id'],source_result=source_result,external_cached_result=external_cached_result,fields=[dict(cells=m['cells'],capture=m['capture']) for m in meshes],reused_without_new_solves=True)

def write_catalog_draft(destination,allowed_cores):
 path=repo_path(destination)
 if path.exists():raise FileExistsError('Preserve draft artifact')
 doc=dict(schema='C38-static-nonlinear-table-CATALOG-DRAFT-v0.1',status='DESIGN_SOURCE_ONLY_NOT_PHYSICAL_PROTOCOL_NOT_LAUNCH_AUTHORIZATION',catalog=catalog(),phase_plan=phase_plan(allowed_cores),mesh_cells=[list(x) for x in immutable.MESHES],immutable_worker=immutable.record(ROOT/immutable.SOURCE),source_sha256=source_pins(),unchanged_physical_gates=immutable.GATES,counts=dict(states=40,reused_states=14,new_states=26,reused_fine_fields=28,new_fine_fields=52,total_fine_fields=80),physics='SamegradedP2stableNH/fullbodygravity/topmiddleknownband/deadforce/bothendsorfinitebottompatches; eachNEW32/40fromREST',interpolation_design=dict(coordinate='inverseYoungsmodulus',table_nodes=3,validation_nodes=2,prospective_tolerance_m=1e-6,not_yet_frozen_for_execution=True),unmet_execution_conditions=['All14 static cases complete and pass; exact32/40fields/cachedcasebytes checked','Supplied-physics baseline controls complete/pass with source/evidence pins','Rootreviews/freezes actualperphase/percase caps from measuredcost','Rootpublishes/reviews exactphaseprotocols and grantsGPUleases'],no_observations_roles_noise_or_posterior_drawn=True,no_new_physical_calls=True,nonclaims=['Deterministicdevelopmentandinterpolationvalidationgrid NOT40independentobjects','NotcontinuousE/forcecertificate','Notdynamic/contact/image/inferencequalification','Validationmaterialsneverpromotedtotableafterseeingerrors'])
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(immutable.safe(doc),f,indent=2,allow_nan=False);f.write('\n')
 return immutable.record(path)

def validate_future_phase(protocol,digest):
 """Reject draft before any output creation; future root-created seal only."""
 path=repo_path(protocol)
 if immutable.sha(path)!=digest:raise ValueError('Changed prospective phase protocol')
 p=json.loads(path.read_text())
 if p.get('schema')!='C38-static-nonlinear-table-PHASE-v0.1' or p.get('status')!='ROOT_REVIEWED_FROZEN_BEFORE_NEW_PHYSICS':raise ValueError('Reviewed physical phase required; catalogDRAFT cannot launch')
 plans=phase_plan(p['all_reserved_CPU_cores']);plan=next((x for x in plans if x['id']==p.get('phase_id')),None)
 if plan is None or p.get('cases')!=plan['cases'] or p.get('output')!=plan['output'] or p.get('owned_metadata_directory')!=plan['owned_metadata_directory'] or p.get('CPU_affinity_cores')!=plan['CPU_affinity_cores']:raise ValueError('Exact disjoint prospective phase/cases/output required')
 if p.get('gates')!=immutable.GATES or p.get('meshes')!=[list(x) for x in immutable.MESHES] or p.get('source_sha256')!=source_pins():raise ValueError('Immutable original gates/mesh/source closure required')
 if p.get('mapping')!=immutable.MAPPING:raise ValueError('Unchangedphysicalgrading required')
 if p.get('solver')!=dict(Newton_cap=25,CG_cap=4000,quadrature_chunk_size=8,force_tolerance_N=1e-10):raise ValueError('UnchangedoriginalNewton/CG/tolerances required')
 accepted=json.loads(checked_record(p['accepted_immutable_worker_protocol']).read_text())
 for name,h in accepted['source_sha256'].items():
  if immutable.sha(ROOT/name)!=h:raise ValueError('Changedacceptedimmutableworker source')
 if p.get('installed_warp_API_sha256')!=accepted['installed_warp_API_sha256'] or not p['installed_warp_API_sha256']:raise ValueError('ExactacceptednonemptyWarpAPIpins required')
 resources=p['resources']
 if resources['physical_GPU_UUID']!=plan['gpu_uuid'] or resources['CPU_threads']!=2 or resources['GPU_allocated_cap_bytes']!=8*1024**3 or resources['RSS_cap_bytes']!=12*1024**3 or resources['disk_free_reserve_bytes']!=30*1024**3 or resources['max_concurrent_cases']!=1:raise ValueError('Original physical-resource limits required')
 for k in ['global_wall_cap_s','per_case_wall_cap_s']:
  if type(resources[k]) is not int or resources[k]<=0:raise ValueError('Root-reviewed positive phase/case budget required')
 prereq=json.loads(checked_record(p['all14_and_baseline_prerequisite']).read_text())
 if prereq.get('all14_static_passed') is not True or prereq.get('supplied_physics_baseline_controls_passed') is not True:raise ValueError('All14/baseline prerequisites required')
 reuse=prereq['reused_case_records'];expected={r['id']:r for r in catalog() if r['reused_original14_case']}
 if len(reuse)!=14 or {r['case_id'] for r in reuse}!=set(expected):raise ValueError('All14 exact reused case records required')
 _,_,certified_old_Torch=immutable.references()
 for r in reuse:check_completed_reuse(expected[r['case_id']],r['source_result'],r['external_cached_result'],certified_old_Torch)
 for r in prereq['baseline_evidence']:checked_record(r)
 if not prereq['baseline_evidence']:raise ValueError('Actual baseline evidence required')
 for name,h in p['installed_warp_API_sha256'].items():
  if immutable.sha(immutable.ENV_SITE/name)!=h:raise ValueError('Changed installed accepted Warp API')
 if immutable.sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
  if os.environ.get(k)!='2':raise ValueError('ExplicitCPU2 required')
 return p,path

def dispatch_worker(protocol,digest,case_id):
 p,path=validate_future_phase(protocol,digest)
 if sum(c['id']==case_id for c in p['cases'])!=1:raise ValueError('OnlydeclaredNEWcase inthisphase mayrun')
 return immutable.worker(p,path,case_id)

def run_phase(protocol,digest):
 p,path=validate_future_phase(protocol,digest)
 if set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('ExactlydeclaredCPU2 required')
 if p['resources']['physical_GPU_UUID'] in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):raise RuntimeError('FuturephaseGPU notidle/leased')
 out=repo_path(p['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ledger=dict(completed=False,passed=False,protocol=immutable.record(path),records=[],queued_case_ids=[c['id'] for c in p['cases']],retries=0,object_roles_or_observations_drawn=False);immutable.write(out/'execution.json',ledger)
 try:
  for case in p['cases']:
   remaining=p['resources']['global_wall_cap_s']-(time.monotonic()-start)
   if remaining<=0:raise TimeoutError('Frozenphasecap; preservequeue')
   immutable.guard_resources(p);row=dict(case=case,terminal=False);ledger['records'].append(row);ledger['queued_case_ids'].remove(case['id']);immutable.write(out/'execution.json',ledger)
   def on_child(pid):row['pid']=pid;immutable.write(out/'execution.json',ledger)
   with (out/(case['id']+'.log')).open('x') as log:
    row.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),sys.executable,'-u','-m','experiments.campaign_c38_nonlinear_table_catalog','worker','--protocol',protocol,'--sha',digest,'--case-id',case['id']],environment={**os.environ,'CUDA_VISIBLE_DEVICES':p['resources']['physical_GPU_UUID']},log=log,timeout=min(remaining,p['resources']['per_case_wall_cap_s']),on_child=on_child))
   row['terminal']=True;f=out/case['id']/'result.json';result=json.loads(f.read_text()) if f.exists() else {};row.update(completed=result.get('completed',False),passed=result.get('passed',False),result=immutable.record(f) if f.exists() else None);immutable.write(out/'execution.json',ledger)
   if row.get('exit_code')!=0 or not result.get('completed') or not result.get('physics_passed'):raise RuntimeError('Stopnewlaunchesonphysical/solver/resourcefailure; preservequeue')
   # Completed spatial-indicator failure is retained and allotherdeclaredcasescontinue.
  ledger.update(completed=True,passed=all(r['passed'] for r in ledger['records']))
 except BaseException as error:ledger['failure']=dict(type=type(error).__name__,message=str(error))
 finally:ledger.update(terminal=True,seconds=time.monotonic()-start);immutable.write(out/'execution.json',ledger)
 return ledger

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['catalog-draft','worker','phase']);parser.add_argument('--output');parser.add_argument('--protocol');parser.add_argument('--sha');parser.add_argument('--case-id');a=parser.parse_args()
 if a.action=='catalog-draft':print(json.dumps(write_catalog_draft(a.output,os.sched_getaffinity(0))))
 else:
  result=dispatch_worker(a.protocol,a.sha,a.case_id) if a.action=='worker' else run_phase(a.protocol,a.sha);print(json.dumps(dict(completed=result.get('completed'),passed=result.get('passed'))));raise SystemExit(0 if result.get('completed') else 1)
