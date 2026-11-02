"""Additive timeout recovery: rerun only missing calls, retain original failure ledger."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json,queue,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics import oracle_campaign as original
from scripts.compositional_neural_mechanics.forward_evaluation import solve_forward,direct_forward,score_prediction
from scripts.compositional_neural_mechanics.feature_pack import load_frozen_laws
from scripts.compositional_neural_mechanics.direct_baseline import load_checkpoint
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.evaluation_protocol import paired_object_bootstrap
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=original.ROOT;R=original.REPORT;OUT=ROOT/'runs/compositional-neural-mechanics-v16/oracle-missing-recovery';PLAN=R/'oracle-missing-recovery-plan.json';ORIGINAL_RESULT=R/'oracle-reserved-forward-results.json'

def prepare():
 if PLAN.exists()or OUT.exists():raise ValueError('Preserve recovery plan/output')
 p,records=original.runtime_records();ledger=json.loads(ORIGINAL_RESULT.read_text());rows=ledger['rows'];missing=[];reused=[]
 if len(rows)!=288 or sum(r['state']=='complete'for r in rows)!=240:raise ValueError('Expected audited original240/48 census')
 for row in rows:
  key=dict(scene_id=row['scene_id'],cohort=row['cohort'],action=row['action'],method=row['method'])
  if row['state']=='complete':
   if sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Original artifact changed')
   reused.append(dict(**key,artifact=row['artifact'],sha256=row['sha256']))
  else:missing.append(key)
 plan=dict(scope=__doc__,runtime=dict(python_executable=str(Path(sys.executable).absolute()),torch=torch.__version__,numpy=np.__version__),parent_freeze_sha256=sha(original.FREEZE),original_result_sha256=sha(ORIGINAL_RESULT),records_sha256=sha(original.OUT/'reserved-records.json'),source_sha256=closure([Path(__file__).resolve()]),reused=reused,jobs=missing,devices=['cuda:0','cuda:1','cuda:2'],per_call_seconds=3600,stage_seconds=7200,method='Identical frozen model/weights,n4predictor,n8reference,q3,querypoints,commands. Only timeout granularity/budget andoutput paths changed.',original_diagnosis_sha256=sha(R/'oracle-recovery-original-diagnosis.json'))
 write(PLAN,plan)

def guard():
 p=json.loads(PLAN.read_text());plan,records=original.runtime_records()
 if p['runtime']!=dict(python_executable=str(Path(sys.executable).absolute()),torch=torch.__version__,numpy=np.__version__):raise ValueError('Use exact original .venv runtime for recovery')
 if p['parent_freeze_sha256']!=sha(original.FREEZE)or p['original_result_sha256']!=sha(ORIGINAL_RESULT)or p['records_sha256']!=sha(original.OUT/'reserved-records.json'):raise ValueError('Original frozen evidence changed')
 for path,h in p['source_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('Recovery source changed')
 for row in p['reused']:
  if sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Reused original changed')
 if len(p['jobs'])!=48 or len(p['reused'])!=240:raise ValueError('Fixed recovery census required')
 return p,plan,records

def worker(index,device):
 torch.set_num_threads(1);p,plan,records=guard();job=p['jobs'][index]
 if device not in p['devices']:raise ValueError('Device notallowed')
 record=next(r for r in records if r['scene']['name']==job['scene_id']);action=next(a for a in record['actions']if a['name']==job['action']);scene=record['scene'];kind=job['method'];folder=OUT/f'job-{index:03d}';folder.mkdir(exist_ok=False);row=dict(job);started=time.monotonic()
 try:
  bulk,interface=load_frozen_laws(*(ROOT/plan['checkpoints'][k]['path']for k in ('bulk','interface')),{k:plan['checkpoints'][k]['sha256']for k in ('bulk','interface')},device)
  if kind.startswith('custom_direct'):
   key='direct_long'if kind=='custom_direct_long'else'direct';model,_=load_checkpoint(ROOT/plan['checkpoints'][key]['path'],map_location=device);model=model.to(device)
   value=direct_forward(scene,action['pose'],np.asarray(plan['material_query']),predictor=model,nodes=plan['node_material'],bulk_law=bulk,interface_law=interface,probes=plan['probes'],checkpoint_sha256={k:plan['checkpoints'][k]['sha256']for k in ('bulk','interface')})
  else:value=solve_forward(scene,action['pose'],np.asarray(plan['material_query']),kind='matched_analytic'if kind=='reference'else kind,n=plan['reference_n']if kind=='reference'else plan['prediction_n'],order=plan['order'],device=device,bulk_law=bulk,interface_law=interface)
  original.validate_value(value,plan['material_query']);target=folder/'prediction.npz';npz(target,material_query=np.asarray(plan['material_query']),displacement=value['displacement'],wrenches=value['wrenches']);row.update(state='complete',artifact=str(target.relative_to(ROOT)),sha256=sha(target),diagnostics=value['diagnostics'])
 except Exception as error:row.update(state='failed',error=repr(error))
 row['seconds']=time.monotonic()-started;guard();write(folder/'complete.json',dict(plan_sha256=sha(PLAN),row=row))

def summarize():
 p,plan,records=guard();terminal=json.loads((OUT/'terminal.json').read_text())
 if terminal['plan_sha256']!=sha(PLAN)or len(terminal['jobs'])!=48 or {x['index']for x in terminal['jobs']}!=set(range(48)):raise ValueError('All48 recovery jobs terminal required')
 rows=[dict(r,state='complete',provenance='original_hash_verified')for r in p['reused']]
 for index,job in enumerate(p['jobs']):
  path=OUT/f'job-{index:03d}/complete.json'
  if path.exists():
   c=json.loads(path.read_text());row=c['row']
   if c['plan_sha256']!=sha(PLAN)or any(row[k]!=v for k,v in job.items()):raise ValueError('Recovery completion identity changed')
  else:row=dict(job,state='failed',error='Missing completed prediction after bounded recovery')
  rows.append(dict(row,provenance='missing_call_recovery'))
 for row in rows:
  if row['state']=='complete':
   if sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Merged artifact changed')
   with np.load(ROOT/row['artifact'],allow_pickle=False)as value:
    original.validate_value(value,plan['material_query'])
    if not np.array_equal(value['material_query'],np.asarray(plan['material_query'])):raise ValueError('Querycoordinates changed')
 refs=[r for r in rows if r['method']=='reference'];write(R/'oracle-recovery-reference-index.json',dict(plan_sha256=sha(PLAN),rows=refs))
 for row in rows:
  if row['state']!='complete'or row['method']=='reference':continue
  ref=next(r for r in refs if r['scene_id']==row['scene_id']and r['action']==row['action'])
  if ref['state']!='complete':continue
  scene=next(r['scene']for r in records if r['scene']['name']==row['scene_id'])
  with np.load(ROOT/row['artifact'])as value,np.load(ROOT/ref['artifact'])as reference:row['scores']=score_prediction(value,reference,GeometrySpec(**scene['geometry']).map(np.asarray(plan['material_query'])))
 summaries=[];paired=[]
 for cohort in ('reserved_composition','reserved_geometry'):
  names=[r['scene']['name']for r in records if r['role']==cohort];means={}
  for method in original.KINDS:
   selected=[r for r in rows if r['cohort']==cohort and r['method']==method];complete=all(r['state']=='complete'and'scores'in r for r in selected);entry=dict(cohort=cohort,method=method,objects=6,expected_commands=24,scored_commands=sum('scores'in r for r in selected),complete=complete)
   if complete:
    means[method]={k:[float(np.mean([r['scores'][k]for r in selected if r['scene_id']==name]))for name in names]for k in selected[0]['scores']};entry['object_mean_scores']={k:float(np.mean(v))for k,v in means[method].items()}
   summaries.append(entry)
  if'shared_laws'in means:
   for baseline in original.KINDS:
    if baseline=='shared_laws'or baseline not in means:continue
    for metric in ('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm'):paired.append(dict(cohort=cohort,baseline=baseline,metric=metric,**paired_object_bootstrap(means['shared_laws'][metric],means[baseline][metric])))
 write(R/'oracle-recovery-merged-results.json',dict(scope=plan['scope'],recovery_plan_sha256=sha(PLAN),original_result_sha256=sha(ORIGINAL_RESULT),original_missing_rows=48,reused_rows=240,recovered_rows=sum(r['state']=='complete'for r in rows if r['provenance']=='missing_call_recovery'),rows=rows,summaries=summaries,paired_object_comparisons=paired,failed_predictions=sum(r['state']!='complete'for r in rows),reference_index_sha256=sha(R/'oracle-recovery-reference-index.json')))

def run():
 p,_,_=guard();OUT.mkdir(parents=True,exist_ok=False);deadline=time.time()+p['stage_seconds'];pending=queue.Queue();jobs=[]
 for i in range(48):pending.put(i)
 write(OUT/'start.json',dict(plan_sha256=sha(PLAN),deadline_unix=deadline))
 def runner(device):
  while True:
   try:i=pending.get_nowait()
   except queue.Empty:return
   if time.time()>=deadline:jobs.append(dict(index=i,exit_code=None,error='Stage deadline'));continue
   with (OUT/f'worker-{i:03d}.log').open('x')as stream:
    child=subprocess.Popen([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.oracle_missing_recovery','--worker',str(i),'--device',device],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=child.wait(timeout=max(.01,min(p['per_call_seconds'],deadline-time.time())))
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
   jobs.append(dict(index=i,exit_code=code));write(OUT/f'exit-{i:03d}.json',jobs[-1])
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:list(pool.map(runner,p['devices']))
 write(OUT/'terminal.json',dict(plan_sha256=sha(PLAN),jobs=jobs));summarize()
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',type=int);parser.add_argument('--device');a=parser.parse_args()
 if a.prepare:prepare()
 elif a.run:run()
 else:worker(a.worker,a.device)
