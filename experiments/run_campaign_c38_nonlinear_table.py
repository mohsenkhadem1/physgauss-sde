"""Prospective three-phase execution of the immutable forty-state catalog.

Preparation and validation read metadata/hashes only. No observations, roles or
noise are sampled. Actual workers reuse the unchanged graded-family mechanics.
"""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
from experiments import campaign_c38_nonlinear_table_catalog as catalog
from experiments import run_campaign_c38_graded_family_warp as worker
from experiments.campaign_owned_gpu_budget_v02 import run_owned_command
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c38_nonlinear_table.py'
TEST='tests/test_run_campaign_c38_nonlinear_table.py'
META='runs/operator-dev/c38-nonlinear-table-v01'
FAMILY='runs/operator-dev/c38-graded-family-warp-v01'
BASELINE='runs/operator-dev/c38-static-linear-predictions-v01'
PHASE_CAPS=(8100,8100,7200)
GiB=1024**3

def record(path):return worker.record(ROOT/path)
def read(r):return json.loads(catalog.checked_record(r).read_text())
def pins():return {**catalog.source_pins(),SOURCE:worker.sha(ROOT/SOURCE),TEST:worker.sha(ROOT/TEST)}
def fresh_write(path,data):
 path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(worker.safe(data),f,indent=2,allow_nan=False);f.write('\n')
 return worker.record(path)

def baseline_numeric_check(summary,terminal,seal):
 """J and spatial accuracy are evidence, never numerical-completion filters."""
 if not (summary.get('completed') is True and summary.get('all8_basis_original_residual_checks_passed') is True and summary.get('all8_independent_direction_checks_passed') is True and summary.get('predictions_completed')==28):raise ValueError('All eight numerical basis solves and28 predictions required')
 if terminal.get('completed') is not True or not all(terminal.get('owned_PID_absence',{}).values()) or not terminal.get('owned_PID_absence'):raise ValueError('Baseline terminal/reaped proof required')
 if seal.get('completed') is not True or seal.get('all28_predictions_sealed_before_nonlinear_target_access') is not True:raise ValueError('Complete prospective baseline prediction seal required')
 return True

def prepare_prerequisite():
 family_record=record(FAMILY+'/completed-summary.json');family=read(family_record)
 if family.get('completed') is not True or family.get('all14_finite_case_q63_displacement_indicators_passed') is not True or len(family['rows'])!=14:raise ValueError('All14 completed static cases required')
 certified=worker.references()[2];expected={c['id']:c for c in catalog.catalog() if c['reused_original14_case']};reused=[]
 for row in family['rows']:
  original=row['case_result'];cache=dict(path='runs/operator-dev/completed-metadata-cache-v01/'+original['sha256']+'.json',sha256=original['sha256'])
  reused.append(catalog.check_completed_reuse(expected[row['case']['id']],original,cache,certified))
 summary_rec=record(BASELINE+'/completed-summary.json');terminal_rec=record(BASELINE+'/terminal-proof.json');summary=read(summary_rec);terminal=read(terminal_rec);seal_rec=summary['predictions_seal'];seal=read(seal_rec)
 baseline_numeric_check(summary,terminal,seal)
 # Verify the frozen baseline sources and capture hashes without opening arrays.
 protocol_rec=summary['protocol'];bp=read(protocol_rec)
 for name,h in bp['source_sha256'].items():
  if worker.sha(ROOT/name)!=h:raise ValueError('Baseline source changed '+name)
 for capture in terminal['captures']:catalog.checked_record(capture)
 return dict(schema='C38-table-completed-prerequisite-v01',all14_static_passed=True,baseline_numerical_CG_completed=True,reused_case_records=reused,family_evidence=[family_record,record(FAMILY+'/terminal-proof.json'),record(FAMILY+'/completed-case-cache-receipt.json')],baseline_evidence=[summary_rec,terminal_rec,seal_rec,protocol_rec,record(BASELINE+'/completed-publication-inventory.json')],baseline_J_and_spatial_are_preserved_evidence_not_launch_filters=True,baseline_outcomes=dict(J=summary['all_prediction_J_passed'],spatial=summary['all_linear_q63_spatial_indicators_passed']),no_arrays_loaded=True)

def prepare():
 prerequisite=fresh_write(META+'/prerequisite.json',prepare_prerequisite());accepted_rec=record(FAMILY+'/protocol-v01.json');accepted=read(accepted_rec);result=[]
 for index,plan in enumerate(catalog.phase_plan(range(6))):
  p=copy.deepcopy(accepted)
  p.update(schema='C38-static-nonlinear-table-execution-PHASE-v01',status='PROSPECTIVE_FROZEN_PENDING_ROOT_REVIEW_NO_GPU_LEASE',phase_id=plan['id'],all_reserved_CPU_cores=list(range(6)),cases=plan['cases'],all_original_cases=catalog.catalog(),output=plan['output'],owned_metadata_directory=plan['owned_metadata_directory'],CPU_affinity_cores=plan['CPU_affinity_cores'],source_sha256=pins(),accepted_immutable_worker_protocol=accepted_rec,all14_and_baseline_prerequisite=prerequisite,all14_reuse_count=14,new_case_count=len(plan['cases']),new_fine_field_count=2*len(plan['cases']),object_roles_or_observations_drawn=False,interpolation_or_dataset_admitted=False)
  p['resources'].update(physical_GPU_UUID=plan['gpu_uuid'],physical_GPU_index=index,global_wall_cap_s=PHASE_CAPS[index],per_case_wall_cap_s=900,absolute_phase_bound_including_cleanup_s=PHASE_CAPS[index]+3,projected_remaining_writes_reserve_bytes=9*GiB)
  p['resource_allocation']='Prospective only: GPU0 root review+baseline completion; GPU1 additionally waits contact release; GPU2 additionally waits baseline release. No simultaneous GPU lease inferred.'
  p['combined_storage_reserve']='Each phase subtracts ALL three phases projected9GiB writes from global free disk, in addition to30GiB free reserve; deliberately conservative as outputs accumulate.'
  p['first_case_gate']='Every case obeys original physical/solver guards; completed spatial failure retained, continue declared cases; stop new launches on technical/physical failure.'
  result.append(fresh_write(META+'/'+plan['id']+'/protocol-v01.json',p))
 return dict(prerequisite=prerequisite,phases=result)

def validate(protocol,digest):
 path=catalog.repo_path(protocol)
 if worker.sha(path)!=digest:raise ValueError('Changed phase protocol')
 p=json.loads(path.read_text());plans=catalog.phase_plan(range(6));index=next((i for i,x in enumerate(plans) if x['id']==p.get('phase_id')),None)
 if p.get('schema')!='C38-static-nonlinear-table-execution-PHASE-v01' or p.get('status')!='PROSPECTIVE_FROZEN_PENDING_ROOT_REVIEW_NO_GPU_LEASE' or index is None:raise ValueError('Exact frozen prospective phase required')
 plan=plans[index]
 for key in ['cases','output','owned_metadata_directory','CPU_affinity_cores']:
  if p[key]!=plan[key]:raise ValueError('Changed exact phase '+key)
 if p['source_sha256']!=pins() or p['gates']!=worker.GATES or p['meshes']!=[list(x) for x in worker.MESHES] or p['mapping']!=worker.MAPPING:raise ValueError('Changed sources/original physical gates')
 if p['solver']!=dict(Newton_cap=25,CG_cap=4000,quadrature_chunk_size=8,force_tolerance_N=1e-10):raise ValueError('Changed solver')
 r=p['resources'];limits=dict(physical_GPU_UUID=plan['gpu_uuid'],physical_GPU_index=index,CPU_threads=2,global_wall_cap_s=PHASE_CAPS[index],per_case_wall_cap_s=900,GPU_allocated_cap_bytes=8*GiB,RSS_cap_bytes=12*GiB,disk_free_reserve_bytes=30*GiB,max_concurrent_cases=1,projected_remaining_writes_reserve_bytes=9*GiB)
 if any(r.get(k)!=v for k,v in limits.items()):raise ValueError('Changed approved budgets or combined write reserve')
 accepted=read(p['accepted_immutable_worker_protocol'])
 for name,h in accepted['source_sha256'].items():
  if worker.sha(ROOT/name)!=h:raise ValueError('Immutable mechanics source changed')
 if p['installed_warp_API_sha256']!=accepted['installed_warp_API_sha256']:raise ValueError('Changed runtime API pins')
 for name,h in p['installed_warp_API_sha256'].items():
  if worker.sha(worker.ENV_SITE/name)!=h:raise ValueError('Changed installed Warp')
 prereq=read(p['all14_and_baseline_prerequisite'])
 if prereq['all14_static_passed'] is not True or prereq['baseline_numerical_CG_completed'] is not True:raise ValueError('Completed prerequisites required')
 evidence=[read(x) for x in prereq['baseline_evidence'][:3]];baseline_numeric_check(*evidence)
 for x in prereq['family_evidence']+prereq['baseline_evidence']:catalog.checked_record(x)
 reuse=prereq['reused_case_records'];expected={c['id']:c for c in catalog.catalog() if c['reused_original14_case']}
 if len(reuse)!=14 or {x['case_id'] for x in reuse}!=set(expected):raise ValueError('Exact14 reuse required')
 certified=worker.references()[2]
 for x in reuse:catalog.check_completed_reuse(expected[x['case_id']],x['source_result'],x['external_cached_result'],certified)
 if worker.sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 return p,path

def dispatch(protocol,digest,case_id):
 p,path=validate(protocol,digest)
 if sum(c['id']==case_id for c in p['cases'])!=1:raise ValueError('Only declared phase case')
 return worker.worker(p,path,case_id)

def run_phase(protocol,digest):
 p,path=validate(protocol,digest)
 if set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('DeclaredCPU2 affinity required')
 if p['resources']['physical_GPU_UUID'] in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):raise RuntimeError('Assigned GPU not idle')
 out=catalog.repo_path(p['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ledger=dict(pid=os.getpid(),completed=False,passed=False,protocol=worker.record(path),records=[],queued_case_ids=[c['id'] for c in p['cases']],retries=0,observations_or_roles_drawn=False)
 worker.write(out/'execution.json',ledger)
 try:
  for case in p['cases']:
   remaining=p['resources']['global_wall_cap_s']-(time.monotonic()-start)
   if remaining<=0:raise TimeoutError('Phase cap; preserve queue')
   worker.guard_resources(p);row=dict(case=case,terminal=False);ledger['records'].append(row);ledger['queued_case_ids'].remove(case['id'])
   def on_child(pid):row['pid']=pid;worker.write(out/'execution.json',ledger)
   command=[p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),sys.executable,'-u','-m','experiments.run_campaign_c38_nonlinear_table','worker','--protocol',protocol,'--sha',digest,'--case-id',case['id']]
   with (out/(case['id']+'.log')).open('x') as log:row.update(run_owned_command(command,environment={**os.environ,'CUDA_VISIBLE_DEVICES':p['resources']['physical_GPU_UUID'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=min(remaining,900),on_child=on_child))
   f=out/case['id']/'result.json';result=json.loads(f.read_text()) if f.exists() else {};row.update(terminal=True,completed=result.get('completed',False),passed=result.get('passed',False),result=worker.record(f) if f.exists() else None);worker.write(out/'execution.json',ledger)
   if row.get('exit_code')!=0 or not result.get('completed') or not result.get('physics_passed'):raise RuntimeError('Stop new launches on solver/physical/resource failure')
  ledger.update(completed=True,passed=all(x['passed'] for x in ledger['records']))
 except BaseException as e:ledger['failure']=dict(type=type(e).__name__,message=str(e))
 finally:ledger.update(terminal=True,seconds=time.monotonic()-start);worker.write(out/'execution.json',ledger)
 return ledger

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','validate','worker','phase']);parser.add_argument('--protocol');parser.add_argument('--sha');parser.add_argument('--case-id');a=parser.parse_args()
 if a.action=='prepare':print(json.dumps(prepare()))
 elif a.action=='validate':p,_=validate(a.protocol,a.sha);print(json.dumps(dict(validated=True,phase=p['phase_id'],new_cases=len(p['cases']),physics_called=False)))
 else:
  r=dispatch(a.protocol,a.sha,a.case_id) if a.action=='worker' else run_phase(a.protocol,a.sha);print(json.dumps(dict(completed=r.get('completed'),passed=r.get('passed'))));raise SystemExit(0 if r.get('completed') else 1)
