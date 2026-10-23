"""Admission-only reuse of immutable private metrics; no truth/archive API.

All256 prediction files and PID closure precede reads of old metric contents.
Only identical adaptive metric inputs may reuse their already-computed rows.
"""
from pathlib import Path
import argparse,copy,json,os
import numpy as np
from experiments import run_campaign_resolved_repair_v04 as repair
from experiments import run_campaign_resolved_numerical_evaluation as evaluation
io=repair.io
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_resolved_admission_update.py'
RESOURCES=dict(CPU_cores=[34,35],threads=2,no_CUDA=True,RSS_bytes=2*1024**3,wall_s=120)
FIELDS=('category_probabilities','canonical_mean_positions_m','material_equal_tail_Pa','material_quantile_details','canonical_position_equal_tail_m','canonical_position_interval95_numerical_enclosure_m','canonical_coordinate_variance_m2','canonical_variance_integration_resolution_estimate_m2')


def sources():return sorted(set(repair.sources())|set(evaluation.sources())|{SOURCE,'tests/test_run_resolved_admission_update.py'})


def bit_equal(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
 if type(a)is not type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys()and all(bit_equal(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(bit_equal(x,y)for x,y in zip(a,b))
 if isinstance(a,(float,np.floating)):return np.float64(a).tobytes()==np.float64(b).tobytes()
 return a==b


def metric_input_identity(old,new):
 a,b=old['methods']['adaptive'],new['methods']['adaptive']
 if a.get('completed')is not True or b.get('completed')is not True:raise ValueError('Both adaptive predictions complete')
 proof={k:bit_equal(a[k],b[k])for k in FIELDS}
 if not all(proof.values()):raise ValueError('Changed adaptive metric input requires reviewed truth evaluation')
 return proof


def barrier(root,current_protocol):
 p=repair.validate(current_protocol['path'],current_protocol['sha256'],root=root)
 base=io.local(root,p['output']);er=io.record(root,base/'execution.json');e=io.read_json(root,er);sr=io.record(root,base/'prediction-seal.json');s=io.read_json(root,sr)
 if e.get('completed')is not True or e.get('technical_failure')is not False or e.get('all_owned_workers_reaped')is not True or e.get('live')!=[]or e.get('queued_job_ids')!=[]:raise ValueError('Current all256 sealed/reaped cohort required')
 pids=[e['pid']]+[r['pid']for r in e['records']]
 if any(Path('/proc',str(pid)).exists()for pid in pids):raise ValueError('Current owned PIDs must be absent')
 if s.get('model_executions')!=256 or s.get('all_numerical_controls_passed')is not True or len(s.get('model_records',[]))!=256:raise ValueError('ALL256 numerical PASS before old metric access')
 previous_ref=p['inputs']['v02_lineage']['refs']['prediction-seal.json'];previous=io.read_json(root,previous_ref)
 key=lambda r:(r['kind'],r['object_id'],r['model'])
 old={key(r):r for r in previous['model_records']};new={key(r):r for r in s['model_records']}
 if len(old)!=256 or len(new)!=256 or old.keys()!=new.keys():raise ValueError('Exact all256 prior/current cohort join')
 changed=[]
 for k,r in new.items():
  if not(r['completed']and r['numerical_controls_passed']and r['core_bit_identity_passed']):raise ValueError('No numerical/core negative admitted')
  for ref in r['files'].values():io.checked(root,ref)
  proof=io.read_json(root,r['core_bit_identity'])
  if proof.get('passed')is not True:raise ValueError('Independent all256 core proof required')
  if r['files']!=old[k]['files']:
   if k[0]!='prefix'or k[2]!='linear_rest_tangent':raise ValueError('Only two linear PREFIX prediction updates')
   prior=repair.old.load_saved_result(root,old[k]['files']);current=repair.old.load_saved_result(root,r['files'])
   changed.append(dict(object_id=k[1],kind=k[0],model=k[2],metric_input_identity=metric_input_identity(prior,current),previous_files=old[k]['files'],current_files=r['files']))
 if len(changed)!=2:raise ValueError('Exactly two prediction updates required')
 return dict(prediction_seal=sr,execution=er,old_prediction_seal=previous_ref,models_checked=256,all_PIDs_absent=True,changed=changed,truth_access=False)


def prepare(protocol,output,*,current_protocol,current_protocol_sha256,old_summary,root=ROOT):
 root=Path(root);dest=io.local(root,protocol)
 if dest.exists()or io.local(root,output).exists():raise FileExistsError('New exclusive admission namespace required')
 current=dict(path=current_protocol,sha256=current_protocol_sha256);proof=barrier(root,current)
 summary=io.record(root,io.local(root,old_summary));rows=io.record(root,io.local(root,str(Path(old_summary).parent/'prefix-linear_rest_tangent-PRIVATE-metrics.json')))
 p=dict(schema='RESOLVED-admission-only-update-v01',status='FROZEN_BEFORE_METRIC_ROW_CONTENT_ACCESS',current_protocol=current,old_summary=summary,old_metrics=rows,prediction_barrier=proof,output=output,source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),resources=RESOURCES,truth_or_observation_archive_access=False,only_admission_changes=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='RESOLVED-admission-only-update-v01'or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('runtime')!=io.runtime()or p.get('resources')!=RESOURCES or p.get('truth_or_observation_archive_access')is not False:raise ValueError('Frozen admission-only source contract')
 # This is the only pre-metric-content array access: declared predictions.
 if barrier(root,p['current_protocol'])!=p['prediction_barrier']:raise ValueError('Full current cohort proof changed')
 io.checked(root,p['old_summary']);io.checked(root,p['old_metrics']);return p


def update_rows(rows,tokens):
 if len(rows)!=64 or len({r['object_id']for r in rows})!=64:raise ValueError('All64 unique records required')
 rows=copy.deepcopy(rows);changed=[]
 for r in rows:
  if r['object_id']in tokens:
   if r.get('primary_prediction_available')is not True or r.get('admitted')is not False:raise ValueError('Only previously available negative primary predictions may be readmitted')
   r['previous_numerical_failures']=r['numerical_failures'];r['numerical_failures']=[];r['admitted']=True;r['admission_updated_without_truth_access']=True;changed.append(r['object_id'])
 if set(changed)!=set(tokens):raise ValueError('Exact two old metric rows required')
 return rows


def run(protocol,digest,*,root=ROOT):
 root=Path(root);p=validate(protocol,digest,root=root);io.guard(root,RESOURCES['CPU_cores']);dest=io.local(root,p['output'])
 if dest.exists():raise FileExistsError('No overwrite or retry')
 old=io.read_json(root,p['old_summary'])
 if old.get('completed')is not True or old.get('resolved_prediction_seal')!=p['prediction_barrier']['old_prediction_seal']:raise ValueError('Exact immutable previous private evaluation seal')
 rows=io.read_json(root,p['old_metrics']);tokens={r['object_id']for r in p['prediction_barrier']['changed']};newrows=update_rows(rows,tokens)
 result=dict(completed=True,protocol=dict(path=protocol,sha256=digest),current_prediction_seal=p['prediction_barrier']['prediction_seal'],old_summary=p['old_summary'],old_metrics=p['old_metrics'],all256_prediction_barrier=p['prediction_barrier'],updated_cohort='prefix/linear_rest_tangent',updated_metrics=evaluation.evaluation.metrics.aggregate(newrows),updated_coverage=evaluation.coverage_summary(newrows),other_cohorts_reused_from_old_summary=True,old_numerical_diagnostics_remain_historical=True,new_cohort_numerical_controls_passed=256,truth_or_observation_archive_access=False,new_population_test=False,all64_denominators_retained=True)
 dest.mkdir(parents=True,exist_ok=False);io.write_new(dest/'prefix-linear_rest_tangent-PRIVATE-metrics.json',io.jsonable(newrows));io.write_new(dest/'summary.json',io.jsonable(result));return result


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--current-protocol');a.add_argument('--current-protocol-sha256');a.add_argument('--old-summary');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,current_protocol=v.current_protocol,current_protocol_sha256=v.current_protocol_sha256,old_summary=v.old_summary))
 else:print(json.dumps(dict(completed=run(v.protocol,v.protocol_sha256)['completed'])))
