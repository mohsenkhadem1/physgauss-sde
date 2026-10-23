"""Separate numerical development diagnostic; no policies/inference or new data.

Require all128 resolved jobs/256 model bundles and all129 ownedPIDs absent before
selective TRUTH_KEYS access. Existing C38/C39 truth was previously examined; no
new population test, calibration, fitting, margins or observation draws occur.
"""
from pathlib import Path
import argparse,copy,json,os,resource,signal
import numpy as np
from experiments import run_campaign_resolved_inference as resolved
from experiments import run_campaign_resolved_repair_v02 as repair
from experiments import run_campaign_c38_prediction_evaluation as evaluation
from experiments import campaign_resolved_numerical_metrics as diagnostic
io=resolved.io;ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_resolved_numerical_evaluation.py'
METRIC_SOURCE='experiments/campaign_c38_prediction_metrics.py'
METRIC_SHA256='b08b2de1cf0534067707871cfc5011e9ebf0a76aa9922047a76455f563ad1b8a'
RESOURCES=dict(cores=[34,35],threads=2,hard_wall_s=180,RSS_bytes=2*1024**3,no_GPU=True)


def sources():return sorted(set(resolved.sources())|set(repair.sources())|{SOURCE,'experiments/campaign_resolved_numerical_metrics.py','tests/test_campaign_resolved_numerical_metrics.py','tests/test_run_campaign_resolved_numerical_evaluation.py',METRIC_SOURCE,'experiments/run_campaign_c38_prediction_evaluation.py'})


def prepare(protocol,output,*,resolved_protocol,resolved_protocol_sha256,observation_execution,observation_execution_sha256,repair_protocol=None,repair_protocol_sha256=None,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol)
 if dest.exists()or io.local(root,output).exists():raise FileExistsError('Fresh separate numericaldiagnostic only')
 r=dict(path=resolved_protocol,sha256=resolved_protocol_sha256);p=resolved.validate(r['path'],r['sha256'],root=root);truth=dict(path=observation_execution,sha256=observation_execution_sha256);io.checked(root,truth);io.checked(root,dict(path=METRIC_SOURCE,sha256=METRIC_SHA256))
 repair_record=None
 if repair_protocol is not None:
  repair_record=dict(path=repair_protocol,sha256=repair_protocol_sha256);q=repair.validate(repair_record['path'],repair_record['sha256'],root=root)
  if q['inputs']['original_protocol']!=r:raise ValueError('Samev01 source for mergedrepair')
 d=dict(schema='RESOLVED-private-numerical-evaluation-v01',status='FROZEN_BEFORE_RESOLVED_PRIVATE_ANALYSIS',resolved_protocol=r,expected_seal_path=(p if repair_record is None else q)['output']+'/prediction-seal.json',repair_protocol=repair_record,observation_execution=truth,output=output,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(root/n)for n in sources()},metrics=dict(path=METRIC_SOURCE,sha256=METRIC_SHA256),cohort=dict(objects=64,kinds=['prefix','SIDE'],models=list(io.MODELS),jobs=128,model_executions=256,physical_configurations=4,noise_repeats=16),private_archive_members=sorted(evaluation.metrics.TRUTH_KEYS),probe_Y_members_accessed=False,new_population_or_policy_experiment=False,truth_previously_examined=True,model_or_interval_adjustment=False,CDF_error_thresholds=list(diagnostic.CDF_THRESHOLDS),oracle_scope='TrueE/BC canonicalpublicpolynomial versus existingdirectwithheldtarget, onceper4fixedconfigurations; no newFEM or fitting',raw_private_rows_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,d);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='RESOLVED-private-numerical-evaluation-v01'or p.get('status')!='FROZEN_BEFORE_RESOLVED_PRIVATE_ANALYSIS'or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('metrics')!=dict(path=METRIC_SOURCE,sha256=METRIC_SHA256)or p.get('private_archive_members')!=sorted(evaluation.metrics.TRUTH_KEYS)or p.get('probe_Y_members_accessed')is not False or p.get('new_population_or_policy_experiment')is not False or p.get('model_or_interval_adjustment')is not False or p.get('truth_previously_examined')is not True or p.get('CDF_error_thresholds')!=list(diagnostic.CDF_THRESHOLDS):raise ValueError('Exactfrozen numericaldiagnostic/privacy/source contract')
 io.checked(root,p['metrics'])
 r=p['resolved_protocol'];rp=resolved.validate(r['path'],r['sha256'],root=root)
 destination=rp['output']
 if p.get('repair_protocol')is not None:
  q=repair.validate(p['repair_protocol']['path'],p['repair_protocol']['sha256'],root=root)
  if q['inputs']['original_protocol']!=p['resolved_protocol']:raise ValueError('Exactmergedrepair originalsource')
  destination=q['output']
 if p['expected_seal_path']!=destination+'/prediction-seal.json':raise ValueError('Exactfuture resolvedseal path')
 io.local(root,p['output']);return p,rp


def validate_complete_metadata(seal,execution,rp,protocol_record):
 """Pure no-I/O completecohort gate; numerical negatives retained, not filtered."""
 expected={(j['kind'],j['object_id']):j for j in rp['inputs']['jobs']}
 if len(rp['inputs']['jobs'])!=128 or len(expected)!=128 or set(k for k,t in expected)!={'prefix','SIDE'}or any(sum(k==kind for k,t in expected)!=64 for kind in ('prefix','SIDE')):raise ValueError('Exact128 plannedprefix/SIDE identitypairs')
 ids={t for k,t in expected}
 if len(ids)!=64 or {(k,t)for k in ('prefix','SIDE')for t in ids}!=set(expected):raise ValueError('Same64 identities forbothkinds')
 if seal.get('schema')!='RESOLVED-prefix-SIDE-prediction-seal-v01'or seal.get('completed')is not True or seal.get('count')!=128 or seal.get('model_executions')!=256 or seal.get('protocol')!=protocol_record or seal.get('initial_prediction_seal')!=rp['inputs']['initial_prediction_seal']or seal.get('old_SIDE_prediction_seal')!=rp['inputs']['SIDE_prediction_seal']:raise ValueError('Completehash-boundresolved256seal required')
 for flag in ('not_new_population_test','quantile_refinement_not_interval_widening','all_postpredictions_sealed_before_private_evaluation'):
  if seal.get(flag)is not True:raise ValueError('Resolvednumericalrole '+flag)
 if seal.get('private_target_or_latent_access')is not False:raise ValueError('No priorresolvedprivateaccess')
 rows=seal.get('objects',[])
 if len(rows)!=128 or {(r.get('kind'),r.get('object_id'))for r in rows}!=set(expected):raise ValueError('Everyresolvedidentity/kind retained')
 if execution.get('completed')is not True or execution.get('technical_failure')is not False or execution.get('all_owned_workers_reaped')is not True or execution.get('protocol')!=protocol_record or execution.get('private_target_or_latent_access')is not False or execution.get('live')!=[]or execution.get('queued_job_ids')!=[]:raise ValueError('Terminalcomplete/reaped/emptyqueue resolvedexecution')
 records=execution.get('records',[]);by={(r.get('kind'),r.get('object_id')):r for r in records}
 if len(records)!=128 or len(by)!=128 or set(by)!=set(expected):raise ValueError('Exact128execution records')
 for r in rows:
  key=(r['kind'],r['object_id']);e=by[key]
  if e.get('completed')is not True or e.get('reaped')is not True or e.get('exit_code')!=0 or e.get('result')!=r.get('result')or e.get('job_id')!=expected[key]['id']:raise ValueError('Everysealedjob completed/reaped/hash-bound')
 pids=[execution.get('pid')]+[r.get('pid')for r in records]
 if any(type(pid)is not int or pid<=1 for pid in pids)or len(set(pids))!=129:raise ValueError('All129 distinct ownedPIDs required')
 return ids,by,expected,pids


def original_barrier(root,p,rp):
 # Never open observationexecution/privatearrays before the entire newcohort.
 sr=io.record(root,io.local(root,rp['output']+'/prediction-seal.json'));seal=io.read_json(root,sr);er=io.record(root,io.local(root,rp['output']+'/execution.json'));execution=io.read_json(root,er)
 ids,rows,jobs,pids=validate_complete_metadata(seal,execution,rp,p['resolved_protocol'])
 if execution.get('prediction_seal')!=sr:raise ValueError('Terminalseal digest mismatch')
 if any(Path('/proc',str(pid)).exists()for pid in pids):raise ValueError('All129 resolvedprocesses must be absent')
 bound={}
 for key,row in rows.items():
  state=resolved.output_result(root,row,p['resolved_protocol']['sha256'])
  if state.get('old_result')!=jobs[key]['old_result']:raise ValueError('Exactoldcore-comparison source required')
  old=io.read_json(root,jobs[key]['old_result'])
  if old.get('completed')is not True or old.get('object_id')!=key[1]or set(old.get('models',{}))!=set(io.MODELS):raise ValueError('Exactoldtwo-model result')
  for name,model in state['models'].items():
   proof=io.read_json(root,model['core_bit_identity'])
   if type(proof.get('passed'))is not bool or not isinstance(proof.get('bit_mismatch_paths'),list)or proof['passed']!=(not proof['bit_mismatch_paths'])or not isinstance(proof.get('scope'),str):raise ValueError('Complete coherentcoreproof, includingnegative')
   if set(old['models'][name].get('files',{}))!=resolved.FILES:raise ValueError('Exactoldcomparison bundle')
   for record in old['models'][name]['files'].values():io.checked(root,record)
  bound[key]=(state,old)
 return ids,bound,dict(prediction_seal=sr,execution=er,all129_PIDs_absent=True,all_new_old_bundles_and_coreproofs_hash_verified=True)



def barrier(root,p,rp):
 ids,bound,proof=original_barrier(root,p,rp)
 if p.get('repair_protocol')is None:return ids,bound,proof
 record=p['repair_protocol'];q=repair.validate(record['path'],record['sha256'],root=root)
 if q['inputs']['original_protocol']!=p['resolved_protocol']:raise ValueError('Repair/v01protocol mismatch')
 mr=io.record(root,io.local(root,p['expected_seal_path']));merged=io.read_json(root,mr)
 if merged.get('schema')!='RESOLVED-v02-merged-model-seal-v01'or merged.get('completed')is not True or merged.get('count')!=128 or merged.get('model_executions')!=256 or merged.get('protocol')!=record or merged.get('original_v01_execution')!=proof['execution']or merged.get('original_v01_prediction_seal')!=proof['prediction_seal']or merged.get('private_target_or_latent_access')is not False or merged.get('not_new_population_test')is not True or merged.get('all_merged_predictions_sealed_before_separate_private_analysis')is not True:raise ValueError('Exactcomplete256model mergedseal and originals')
 terminal=io.read_json(root,merged['repair_execution']);er=io.record(root,io.local(root,q['output']+'/execution.json'));execution=io.read_json(root,er);jobs={j['id']:j for j in q['inputs']['jobs']}
 if terminal.get('completed')is not True or terminal.get('protocol')!=record or terminal.get('all_owned_workers_reaped')is not True or terminal.get('private_target_or_latent_access')is not False or execution.get('completed')is not True or execution.get('technical_failure')is not False or execution.get('all_owned_workers_reaped')is not True or execution.get('protocol')!=record or execution.get('prediction_seal')!=mr or execution.get('records')!=terminal.get('records')or execution.get('pid')!=terminal.get('pid')or execution.get('live')!=[]or execution.get('queued_job_ids')!=[]:raise ValueError('Completed/reapedrepair execution and finalseal binding')
 rows=terminal.get('records',[])
 if len(rows)!=len(jobs)or {r['job_id']for r in rows}!=set(jobs)or any(r.get('completed')is not True or r.get('reaped')is not True or r.get('exit_code')!=0 for r in rows):raise ValueError('All declared SINGLEmodel repair workers retained')
 pids=[terminal.get('pid')]+[r.get('pid')for r in rows]
 if any(type(pid)is not int or pid<=1 or Path('/proc',str(pid)).exists()for pid in pids)or len(set(pids))!=len(pids):raise ValueError('Allrepair controller/worker PIDs absent')
 expected=repair.merged_records(root,q,terminal,record['sha256'])
 denominators={**q['inputs']['denominators'],'v02_repaired_model_arms':len(jobs),'reused_v01_model_arms':len(q['inputs']['reused_models']),'merged_numerical_negative_arms':sum(not(r['completed']and r['numerical_controls_passed']and r['core_bit_identity_passed'])for r in expected)}
 if merged.get('denominators')!=denominators:raise ValueError('Mergedfailure/reuse denominators must match records')
 if merged.get('model_records')!=expected or len(expected)!=256 or {(r['kind'],r['object_id'],r['model'])for r in expected}!={(k,t,n)for k,t in bound for n in io.MODELS}:raise ValueError('Exactall256model origins/files/proofs, no subset or replacement')
 final={key:({**state,'models':{}},old)for key,(state,old)in bound.items()}
 for row in expected:
  for f in row['files'].values():io.checked(root,f)
  io.checked(root,row['source_result']);io.checked(root,row['source_protocol']);c=io.read_json(root,row['core_bit_identity'])
  if type(c.get('passed'))is not bool or not isinstance(c.get('bit_mismatch_paths'),list)or c['passed']!=(not c['bit_mismatch_paths'])or c['passed']!=row['core_bit_identity_passed']:raise ValueError('Coherentmergedcore proof')
  final[(row['kind'],row['object_id'])][0]['models'][row['model']]=dict(execution_completed=True,completed=row['completed'],numerical_controls_passed=row['numerical_controls_passed'],files=row['files'],core_bit_identity=row['core_bit_identity'],core_bit_identity_passed=row['core_bit_identity_passed'])
 return ids,final,dict(prediction_seal=mr,execution=er,original_v01_prediction_seal=proof['prediction_seal'],original_v01_execution=proof['execution'],repair_execution=merged['repair_execution'],all129_original_PIDs_absent=True,all_repair_PIDs_absent=True,all_new_old_bundles_and_coreproofs_hash_verified=True,denominators=merged['denominators'])


def guard():
 if set(os.sched_getaffinity(0))!=set(RESOURCES['cores'])or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU34/35 thread2 noGPU')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['RSS_bytes']:raise MemoryError('2GiB diagnosticcap')


def load_models(root,rp):
 """Canonicalpublicmodels only; no SIDE/DOWN measurement arrays are loaded."""
 from experiments.campaign_c38_candidate_point_forward import CandidatePointForward,ARCHIVE_KEYS
 i=rp['inputs']['model_inputs']
 with np.load(io.checked(root,i['candidate_archive']),allow_pickle=False)as z:
  if set(z.files)!=ARCHIVE_KEYS:raise ValueError('ExactpubliccandidateTABLE archive')
  nonlinear=CandidatePointForward({k:z[k].copy()for k in ARCHIVE_KEYS})
 return dict(zip(io.MODELS,(nonlinear,io.load_linear_library(root,i['linear_library']))))


def score_preserving_negatives(prediction,truth,public,proof):
 # Saved products remain immutable; this diagnostic alone denies admission on
 # an independently detected core regression. No numerical value is replaced.
 value={**prediction,'numerical_controls_passed':prediction.get('numerical_controls_passed')is True and proof['passed']};value['failures']=list(prediction.get('failures',[]))
 if not proof['passed']:value['failures'].append(dict(stage='CORE_BIT_IDENTITY',paths=proof['bit_mismatch_paths']))
 try:result=evaluation.metrics.score_prediction(value,truth,public)
 except(ValueError,KeyError,TypeError)as error:
  failed={**value,'numerical_controls_passed':False,'methods':{**value.get('methods',{}),'adaptive':dict(completed=False)},'failures':value['failures']+[dict(stage='SAVED_PRIMARY_NUMERICS',error=repr(error))]};result=evaluation.metrics.score_prediction(failed,truth,public)
 result['core_bit_identity']=diagnostic.core_group_status(proof);return result

def aggregate_diagnostics(rows):
 out=dict(records=len(rows),core_bit_identity_counts={k:sum(r['core_bit_identity'][k]for r in rows)for k in ('all_core_bit_identical','material','mean','variance')},methods={})
 for method in diagnostic.METHODS:
  values=[r['quantiles'][method]for r in rows];d={key:sum(v.get(key,0)for v in values)for key in ('expected_coordinates','available_coordinate_entries','expected_quantiles','available_quantiles','missing_quantiles','continuous_quantiles','atom_quantiles','invalid_continuous_CDF_enclosures','missing_continuous_CDF_bounds','unqualified_atom_jumps')}
  bounds=[v.get('maximum_continuous_CDF_error_bound')for v in values if v.get('maximum_continuous_CDF_error_bound')is not None];d['maximum_continuous_CDF_error_bound']=max(bounds)if bounds else None;d['continuous_CDF_error_count_greater_than']={str(t):sum(v.get('continuous_CDF_error_count_greater_than',{}).get(str(t),0)for v in values)for t in diagnostic.CDF_THRESHOLDS}
  for units in ('displacement','position'):
   keys=('available_coordinates','missing_coordinates','negative_width_count','zero_width_count','zero_width_nonconstant_count','zero_width_nonconstant_continuous_bounds_count','zero_width_nonconstant_atom_bounds_count','zero_width_constant_count');d[units]={key:sum(v.get(units,{}).get(key,0)for v in values)for key in keys}
  out['methods'][method]=d
 h=[r['oracle_error_relative_halfwidth']for r in rows];out['oracle_error_relative_halfwidth']={key:sum(x.get(key,0)for x in h)for key in ('available_coordinates','missing_coordinates','positive_halfwidth_count','negative_halfwidth_count','zero_halfwidth_zero_error_count','zero_halfwidth_positive_oracle_error_count','finite_ratio_count','positive_halfwidth_error_larger_than_halfwidth_count')};ratios=[x['maximum_finite_error_over_halfwidth']for x in h if x.get('maximum_finite_error_over_halfwidth')is not None];out['oracle_error_relative_halfwidth']['maximum_finite_error_over_halfwidth']=max(ratios)if ratios else None
 return out


def coverage_summary(rows):
 admitted=[r for r in rows if r['admitted']and r['primary_prediction_available']];available=[r for r in rows if r['primary_prediction_available']]
 def counts(selected,key):return dict(covered=sum(r[key]['covered_count']for r in selected),z_covered=sum(r[key]['z_covered_count']for r in selected),covered_by_axis=np.sum([r[key]['covered_count_by_axis']for r in selected],axis=0).tolist()if selected else[0,0,0])
 return dict(denominator_objects=64,denominator_coordinates_including_failures=64*27,denominator_z_including_failures=64*9,numerically_admitted=len(admitted),available_descriptive_predictions=len(available),raw95_admitted=counts(admitted,'canonical_pointwise95'),numerical_enclosure95_admitted=counts(admitted,'canonical_pointwise95_numerical_quantile_enclosure'),raw95_available_descriptive=counts(available,'canonical_pointwise95'),numerical_enclosure95_available_descriptive=counts(available,'canonical_pointwise95_numerical_quantile_enclosure'),scope='Descriptive conditional noise repetitions of4fixedphysics configurations; unavailable/unadmitted remain in full denominators. Not physicalpopulation calibration.')


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p,rp=validate(protocol,digest,root=root);guard();dest=io.local(root,p['output'])
 if dest.exists():raise FileExistsError('No numericaldiagnostic overwrite/retry')
 def cap(*args):raise TimeoutError('Numericaldiagnostic180scap')
 previous=signal.signal(signal.SIGALRM,cap);signal.alarm(180)
 try:
  ids,bound,proof=barrier(root,p,rp)
  # No private execution JSON or TRUTH_KEYS array member was opened above.
  execution=io.read_json(root,p['observation_execution']);records=execution.get('records',[])
  if execution.get('completed')is not True or len(records)!=64 or {r['object_id']for r in records}!=ids or {(r['private_configuration_index'],r['repeat_index'])for r in records}!={(c,r)for c in range(4)for r in range(16)}:raise ValueError('Sameoriginal complete4x16identity/targetsource')
  jobs={(j['kind'],j['object_id']):j for j in rp['inputs']['jobs']}
  for r in records:
   if r['public_prefix']!=jobs[('prefix',r['object_id'])]['prefix']or r['public_prefix']!=jobs[('SIDE',r['object_id'])]['prefix']:raise ValueError('Exactoriginalprefixbytes')
   io.checked(root,r['private_future']);io.checked(root,r['public_prefix'])
  models=load_models(root,rp);coeff={name:{c:np.asarray(model.observation_coefficients(c,1.,np.arange(63,dtype=np.int64)))for c in model.categories}for name,model in models.items()};topology={name:diagnostic.polynomial_ranges(c)for name,c in coeff.items()};design=evaluation.obs.load_frozen_design(root);metric_rows={k:{n:[]for n in io.MODELS}for k in resolved.KINDS};diagnostic_rows={k:{n:[]for n in io.MODELS}for k in resolved.KINDS};oracle_cache={};oracle_summary={};cfg_truth={}
  dest.mkdir(parents=True)
  for r in sorted(records,key=lambda r:r['object_id']):
   guard();token=r['object_id']
   with np.load(io.checked(root,r['private_future']),allow_pickle=False)as z:truth=evaluation.extract_truth_members(z)
   if truth['object_id']!=token or truth['private_configuration_index']!=r['private_configuration_index']or truth['repeat_index']!=r['repeat_index']:raise ValueError('Exacttruth/sourceidentity')
   public=evaluation.codec.load_public_prefix(root,r['public_prefix'],design=design);cfg=truth['private_configuration_index'];q=truth['canonical_target_positions_m'];context=(truth['young_Pa'],truth['support_category'],q.tobytes())
   if cfg in cfg_truth and cfg_truth[cfg]!=context:raise ValueError('Canonicaltarget/truephysics must agree across16noise repeats')
   cfg_truth[cfg]=context
   for name in io.MODELS:
    key=(cfg,name)
    if key not in oracle_cache:
     oracle=diagnostic.oracle_error(coeff[name][truth['support_category']],truth['young_Pa'],q);oracle_cache[key]=oracle;oracle_summary[str(cfg)+'/'+name]=dict(configuration=cfg,model=name,young_Pa=truth['young_Pa'],support_category=truth['support_category'],**{k:v for k,v in oracle.items()if k not in ('displacement_error_m','position_error_m')})
    for kind in resolved.KINDS:
     state,old=bound[(kind,token)];entry=state['models'][name];new=resolved.load_saved_result(root,entry['files']);prior=resolved.load_saved_result(root,old['models'][name]['files']);comparison=resolved.core_identity(prior,new);stored=io.read_json(root,entry['core_bit_identity'])
     if comparison['passed']!=stored['passed']or sorted(comparison['bit_mismatch_paths'])!=sorted(stored['bit_mismatch_paths']):raise ValueError('Recomputedcoreproof differs from frozenworkerproof')
     if new.get('object_id')!=token:raise ValueError('Savedprediction identity')
     scored=score_preserving_negatives(new,truth,public,comparison);metric_rows[kind][name].append(scored)
     try:quantiles=diagnostic.quantile_diagnostics(new,topology[name]);relative=diagnostic.error_relative_halfwidth(new,oracle_cache[key])
     except(ValueError,TypeError,KeyError)as error:
      quantiles={m:dict(expected_coordinates=27,available_coordinate_entries=0,expected_quantiles=81,available_quantiles=0,missing_quantiles=81,diagnostic_error=repr(error),displacement=dict(available_coordinates=0,missing_coordinates=27),position=dict(available_coordinates=0,missing_coordinates=27))for m in diagnostic.METHODS};relative=dict(available_coordinates=0,missing_coordinates=27,diagnostic_error=repr(error));scored['admitted']=False;scored['numerical_failures']=list(scored['numerical_failures'])+[dict(stage='SAVED_QUANTILE_DIAGNOSTIC',error=repr(error))]
     diagnostic_rows[kind][name].append(dict(object_id=token,private_configuration_index=cfg,repeat_index=truth['repeat_index'],core_bit_identity=diagnostic.core_group_status(comparison),quantiles=quantiles,oracle_error_relative_halfwidth=relative))
     del new,prior
  if len(cfg_truth)!=4:raise ValueError('All4fixedoraclephysics retained')
  models_summary={}
  for kind in resolved.KINDS:
   models_summary[kind]={}
   for name in io.MODELS:
    rows=metric_rows[kind][name];details=diagnostic_rows[kind][name]
    if len(rows)!=64 or len(details)!=64:raise ValueError('No droppeddenominators')
    io.write_new(dest/(kind+'-'+name+'-PRIVATE-metrics.json'),io.jsonable(rows));io.write_new(dest/(kind+'-'+name+'-PRIVATE-numerics.json'),io.jsonable(details));models_summary[kind][name]=dict(metrics=evaluation.metrics.aggregate(rows),coverage=coverage_summary(rows),numerical_diagnostics=aggregate_diagnostics(details))
  validate(protocol,digest,root=root);guard();summary=dict(completed=True,protocol=dict(path=protocol,sha256=digest),resolved_prediction_seal=proof['prediction_seal'],resolved_execution=proof['execution'],full_hash_and_PID_barrier=proof,observation_execution=p['observation_execution'],models=models_summary,public_model_oracle_fixed4=io.jsonable(oracle_summary),source_sha256=p['source_sha256'],selective_private_members=sorted(evaluation.metrics.TRUTH_KEYS),SIDE_DOWN_observation_arrays_read=False,core_mismatch_forces_negative_admission=True,all64_denominators_retained=True,model_or_interval_changes=False,new_population_test=False,truth_previously_analyzed=True,not_active_policy_rerun=True,raw_private_rows_local=True)
  io.write_new(dest/'summary.json',io.jsonable(summary));return summary
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--resolved-protocol');a.add_argument('--resolved-protocol-sha256');a.add_argument('--repair-protocol');a.add_argument('--repair-protocol-sha256');a.add_argument('--observation-execution');a.add_argument('--observation-execution-sha256');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,resolved_protocol=v.resolved_protocol,resolved_protocol_sha256=v.resolved_protocol_sha256,observation_execution=v.observation_execution,observation_execution_sha256=v.observation_execution_sha256,repair_protocol=v.repair_protocol,repair_protocol_sha256=v.repair_protocol_sha256))
 else:r=run(v.protocol,v.protocol_sha256);print(json.dumps(dict(completed=r['completed'])))
