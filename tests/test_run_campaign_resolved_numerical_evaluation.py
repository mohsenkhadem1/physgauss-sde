"""Complete temporary saved cohort; no actual prefix/physics/outcome access."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_resolved_numerical_evaluation as m
from test_campaign_c38_public_prefix_inference_v02 import design


def put(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def prediction(token):
 q=m.diagnostic.POINTS.copy();u=np.zeros((63,3));z=np.zeros((9,3,3));details=[dict(point_id=int(point),axis=axis,quantiles=[dict(value_m=0.,lower_m=0.,upper_m=0.,kind='common_constant_atom',CDF_left=dict(cdf=0.,lower=0.,upper=0.),CDF_at=dict(cdf=1.,lower=1.,upper=1.))for p in m.diagnostic.PROBABILITIES])for point in m.diagnostic.POINT_IDS for axis in range(3)]
 row=dict(completed=True,stage='COMPLETED',log_normalizer=0.,normalizer_error_estimate=0.,scaled_normalizer=1.,normalizer_relative_error_estimate=0.,category_probabilities=np.array([.5,.5]),material_CDF=np.linspace(0,1,17),material_equal_tail_Pa=[75000.,float(np.sqrt(75000.*150000.)),150000.],material_quantile_details=[dict(unit_bracket=dict(lower=0.,upper=.001)),dict(unit_bracket=dict(lower=.49,upper=.51)),dict(unit_bracket=dict(lower=.999,upper=1.))],canonical_mean_displacement_m=u.copy(),canonical_mean_positions_m=q.copy(),canonical_coordinate_variance_m2=u.copy(),canonical_mean_coordinate_variance_m2=0.,canonical_moment_numerics={'error':0.},canonical_variance_integration_resolution_estimate_m2=u.copy(),canonical_variance_resolution_scope='Manufactured constanttarget',canonical_displacement_equal_tail_m=z.copy(),canonical_position_equal_tail_m=q[m.diagnostic.POINT_IDS,:,None]+z,canonical_position_interval95_numerical_enclosure_m=np.repeat(q[m.diagnostic.POINT_IDS,:,None],2,axis=2),canonical_quantile_details=details)
 return dict(object_id=token,completed=True,numerical_controls_passed=True,failures=[],categories=list(m.evaluation.metrics.CATEGORIES),material_CDF_grid_Pa=np.geomspace(75000.,150000.,17),canonical_point_ids=np.arange(63),canonical_interval_point_ids=m.diagnostic.POINT_IDS.copy(),equal_tail_probabilities=list(m.diagnostic.PROBABILITIES),gates=m.resolved.prefix.GATES.copy(),methods={n:copy.deepcopy(row)for n in m.resolved.prefix.METHODS})


def bundle(root,directory,value):
 p=root/directory;p.mkdir(parents=True);m.io.save_full_result(p,value);return {name:m.io.record(root,p/name)for name in m.resolved.FILES}


def fixture(root,monkeypatch,*,core_mismatch=True):
 jobs=[];sealed=[];executed=[];obs=[];protocol_record=put(root,'resolved-protocol.json',dict(manufactured_only=True));pid=9000000
 for i in range(64):
  token=format(i+1,'032x');public=m.evaluation.obs.public_prefix_payload(token,m.diagnostic.POINTS[m.diagnostic.POINT_IDS].copy(),design=design());pp=root/'PUBLIC'/(token+'.npz');pp.parent.mkdir(exist_ok=True);np.savez(pp,**m.evaluation.codec.encode_public(public,design=design()));prefix=m.io.record(root,pp)
  truth=dict(object_id=np.asarray(token),private_configuration_index=np.asarray(i//16,dtype=np.int64),repeat_index=np.asarray(i%16,dtype=np.int64),young_Pa=np.asarray(100000.,dtype=np.float64),support_category=np.asarray(m.evaluation.metrics.CATEGORIES[i//32]),canonical_target_Fz_N=np.asarray(1.,dtype=np.float64),canonical_target_point_ids=np.arange(63),canonical_target_positions_m=m.diagnostic.POINTS.copy(),candidate_probe_observations_m=np.asarray([object()],dtype=object),SIDE_probe_observations_m=np.asarray([object()],dtype=object));tp=root/'PRIVATE'/(token+'.npz');tp.parent.mkdir(exist_ok=True);np.savez(tp,**truth);obs.append(dict(object_id=token,private_configuration_index=i//16,repeat_index=i%16,public_prefix=prefix,private_future=m.io.record(root,tp)))
  for kind in m.resolved.KINDS:
   jid=token+'__'+kind;oldmodels={};models={}
   for name in m.io.MODELS:
    before=prediction(token);after=copy.deepcopy(before)
    if core_mismatch and i==0 and kind=='prefix'and name==m.io.MODELS[0]:after['methods']['adaptive']['canonical_mean_positions_m'][0,0]+=1e-12
    if i==1 and kind=='SIDE'and name==m.io.MODELS[1]:after['numerical_controls_passed']=False
    oldmodels[name]=dict(execution_completed=True,files=bundle(root,'old/'+jid+'/'+name,before));files=bundle(root,'resolved/objects/'+jid+'/'+name,after);comparison=m.resolved.core_identity(before,after);proof=put(root,'resolved/objects/'+jid+'/'+name+'/core.json',comparison);models[name]=dict(execution_completed=True,completed=True,numerical_controls_passed=after['numerical_controls_passed'],files=files,core_bit_identity=proof,core_bit_identity_passed=comparison['passed'])
   old=put(root,'old/'+jid+'/result.json',dict(completed=True,object_id=token,models=oldmodels));job=dict(id=jid,kind=kind,object_id=token,prefix=prefix,old_result=old);jobs.append(job);pid+=1;record=put(root,'resolved/objects/'+jid+'/result.json',dict(completed=True,pid=pid,object_id=token,kind=kind,protocol_sha256=protocol_record['sha256'],models=models,old_result=old,private_target_or_latent_access=False));sealed.append(dict(kind=kind,object_id=token,result=record));executed.append(dict(job_id=jid,kind=kind,object_id=token,result=record,pid=pid,exit_code=0,completed=True,reaped=True))
 initial=dict(path='initial-seal',sha256='b'*64);side=dict(path='SIDE-seal',sha256='c'*64);rp=dict(output='resolved',inputs=dict(jobs=jobs,initial_prediction_seal=initial,SIDE_prediction_seal=side));seal=put(root,'resolved/prediction-seal.json',dict(schema='RESOLVED-prefix-SIDE-prediction-seal-v01',completed=True,count=128,model_executions=256,objects=sealed,protocol=protocol_record,initial_prediction_seal=initial,old_SIDE_prediction_seal=side,not_new_population_test=True,quantile_refinement_not_interval_widening=True,private_target_or_latent_access=False,all_postpredictions_sealed_before_private_evaluation=True,all_numerical_controls_passed=False));put(root,'resolved/execution.json',dict(completed=True,technical_failure=False,all_owned_workers_reaped=True,protocol=protocol_record,private_target_or_latent_access=False,records=executed,live=[],queued_job_ids=[],pid=9000000,prediction_seal=seal));observation=put(root,'original-execution.json',dict(completed=True,records=obs));p=dict(output='evaluation',expected_seal_path=seal['path'],resolved_protocol=protocol_record,observation_execution=observation,source_sha256={});monkeypatch.setattr(m,'validate',lambda*a,**k:(p,rp));monkeypatch.setattr(m,'guard',lambda:None);monkeypatch.setattr(m.evaluation.obs,'load_frozen_design',lambda*a:design())
 class Model:
  categories=m.evaluation.metrics.CATEGORIES
  def observation_coefficients(self,c,F,ids):assert F==1.;return np.zeros((3,3*len(ids)))
 monkeypatch.setattr(m,'load_models',lambda*a:dict(zip(m.io.MODELS,(Model(),Model()))));return p,rp,seal


def test_full_saved128_jobs_256models_preserves_all_denominators_and_selective_truth(tmp_path,monkeypatch):
 p,rp,seal=fixture(tmp_path,monkeypatch);original=np.load;private_reads=[]
 class Private:
  def __init__(self,z):self.z=z
  def __enter__(self):return self
  def __exit__(self,*a):self.z.close()
  def __getitem__(self,key):
   private_reads.append(key)
   if key not in m.evaluation.metrics.TRUTH_KEYS:raise AssertionError('UnusedprobeY access')
   return self.z[key]
 def load(path,*a,**kw):
  z=original(path,*a,**kw);return Private(z)if Path(path).parent.name=='PRIVATE'else z
 monkeypatch.setattr(np,'load',load);s=m.run('recipe','d'*64,root=tmp_path)
 assert s['completed']and s['all64_denominators_retained']and not s['SIDE_DOWN_observation_arrays_read']and set(private_reads)==m.evaluation.metrics.TRUTH_KEYS
 assert len(s['public_model_oracle_fixed4'])==8
 assert s['models']['prefix'][m.io.MODELS[0]]['coverage']['numerically_admitted']==63
 assert s['models']['SIDE'][m.io.MODELS[1]]['coverage']['numerically_admitted']==63
 assert all(v['coverage']['denominator_coordinates_including_failures']==1728 for rows in s['models'].values()for v in rows.values())
 d=s['models']['prefix'][m.io.MODELS[0]]['numerical_diagnostics']['core_bit_identity_counts'];assert d==dict(all_core_bit_identical=63,material=64,mean=63,variance=64)


def test_missing_last_newbundle_hash_blocks_ALL_private_arrays_and_execution(tmp_path,monkeypatch):
 p,rp,seal=fixture(tmp_path,monkeypatch);last=rp['inputs']['jobs'][-1];state=m.io.read_json(tmp_path,m.io.record(tmp_path,tmp_path/'resolved/objects'/last['id']/'result.json'));f=state['models'][m.io.MODELS[-1]]['files']['posterior-arrays.npz'];(tmp_path/f['path']).unlink()
 monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('Arrayread beforefullhashbarrier')));original=m.io.read_json
 def read(root,record):
  if record==p['observation_execution']:raise AssertionError('Privateexecution read beforeallbundles')
  return original(root,record)
 monkeypatch.setattr(m.io,'read_json',read)
 with pytest.raises(FileNotFoundError):m.run('recipe','d'*64,root=tmp_path)
 assert not(tmp_path/'evaluation').exists()


def test_exact_missing_job_and_live_pid_metadata_rejected_without_arrays(tmp_path,monkeypatch):
 p,rp,sr=fixture(tmp_path,monkeypatch);s=m.io.read_json(tmp_path,sr);e=m.io.read_json(tmp_path,m.io.record(tmp_path,tmp_path/'resolved/execution.json'))
 monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('Noarrays in metadata gate')))
 m.validate_complete_metadata(s,e,rp,p['resolved_protocol'])
 bad=copy.deepcopy(s);bad['objects'].pop()
 with pytest.raises(ValueError):m.validate_complete_metadata(bad,e,rp,p['resolved_protocol'])
 bad=copy.deepcopy(e);bad['records'][-1]['reaped']=False
 with pytest.raises(ValueError):m.validate_complete_metadata(s,bad,rp,p['resolved_protocol'])


def merged_fixture(root,monkeypatch):
 p,rp,oldseal=fixture(root,monkeypatch,core_mismatch=False);original=p['resolved_protocol'];repair_record=put(root,'repair-protocol.json',dict(manufactured_guard_repair=True));reused=[];failed=None
 original_execution=m.io.read_json(root,m.io.record(root,root/'resolved/execution.json'));byid={(r['kind'],r['object_id']):r for r in original_execution['records']}
 for j in rp['inputs']['jobs']:
  state=m.io.read_json(root,byid[(j['kind'],j['object_id'])]['result'])
  for name,v in state['models'].items():
   entry=dict(object_id=j['object_id'],kind=j['kind'],model=name,files=v['files'],core_bit_identity=v['core_bit_identity'],completed=v['completed'],numerical_controls_passed=v['numerical_controls_passed'],core_bit_identity_passed=v['core_bit_identity_passed'],source_result=byid[(j['kind'],j['object_id'])]['result'],source_protocol=original)
   if entry['numerical_controls_passed']:reused.append(dict(entry,origin='REUSED_v01'))
   else:failed=dict(id=j['id']+'__'+name,object_id=j['object_id'],kind=j['kind'],model=name,public_job=j,v01_failed_model=entry)
 assert failed is not None and len(reused)==255
 q=dict(output='repair',inputs=dict(original_protocol=original,reused_models=reused,jobs=[failed],denominators=dict(total_model_arms=256,v01_completed_numeric_core_pass=255,v01_failed_model_arms=1,v01_core_bit_mismatches=0)));monkeypatch.setattr(m.repair,'validate',lambda*a,**k:q);monkeypatch.setattr(m.resolved,'validate',lambda*a,**k:rp)
 value=m.resolved.load_saved_result(root,failed['v01_failed_model']['files']);value['numerical_controls_passed']=True;files=bundle(root,'repair/objects/one',value);old=m.io.read_json(root,failed['public_job']['old_result']);before=m.resolved.load_saved_result(root,old['models'][failed['model']]['files']);proof=put(root,'repair/objects/one/core.json',m.resolved.core_identity(before,value));source=put(root,'repair/objects/one/result.json',dict(completed=True,pid=9000201,object_id=failed['object_id'],kind=failed['kind'],model=failed['model'],protocol_sha256=repair_record['sha256'],private_target_or_latent_access=False,origin='REPAIRED_v02',files=files,core_bit_identity=proof,core_bit_identity_passed=True,model_completed=True,numerical_controls_passed=True));row=dict(job_id=failed['id'],object_id=failed['object_id'],kind=failed['kind'],model=failed['model'],pid=9000201,result=source,reaped=True,exit_code=0,completed=True)
 terminal=dict(completed=True,pid=9000200,protocol=repair_record,records=[row],all_owned_workers_reaped=True,private_target_or_latent_access=False);terminal_record=put(root,'repair/terminal-executions.json',terminal);records=m.repair.merged_records(root,q,terminal,repair_record['sha256']);er=m.io.record(root,root/'resolved/execution.json');den={**q['inputs']['denominators'],'v02_repaired_model_arms':1,'reused_v01_model_arms':255,'merged_numerical_negative_arms':0};sr=put(root,'repair/prediction-seal.json',dict(schema='RESOLVED-v02-merged-model-seal-v01',completed=True,count=128,model_executions=256,model_records=records,original_v01_execution=er,original_v01_prediction_seal=oldseal,repair_execution=terminal_record,protocol=repair_record,private_target_or_latent_access=False,not_new_population_test=True,all_merged_predictions_sealed_before_separate_private_analysis=True,denominators=den));put(root,'repair/execution.json',dict(**terminal,prediction_seal=sr,technical_failure=False,live=[],queued_job_ids=[]));p.update(repair_protocol=repair_record,expected_seal_path=sr['path']);return p,rp,sr,files


def test_model_level_merged_full256_origin_hashes_and_all64_denominators(tmp_path,monkeypatch):
 p,rp,seal,files=merged_fixture(tmp_path,monkeypatch);s=m.run('recipe','d'*64,root=tmp_path)
 assert s['completed']and s['resolved_prediction_seal']==seal and s['full_hash_and_PID_barrier']['all129_original_PIDs_absent']and s['full_hash_and_PID_barrier']['all_repair_PIDs_absent']
 assert s['full_hash_and_PID_barrier']['denominators']['reused_v01_model_arms']==255
 assert all(v['coverage']['numerically_admitted']==64 and v['coverage']['denominator_coordinates_including_failures']==1728 for rows in s['models'].values()for v in rows.values())


def test_missing_repaired_model_file_blocks_ALL_private_members(tmp_path,monkeypatch):
 p,rp,seal,files=merged_fixture(tmp_path,monkeypatch);(tmp_path/files['posterior-arrays.npz']['path']).unlink();monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('Anyarraybeforefullmergedhashbarrier')));original=m.io.read_json
 def read(root,record):
  if record==p['observation_execution']:raise AssertionError('Privateexecution beforelastrepairedhash')
  return original(root,record)
 monkeypatch.setattr(m.io,'read_json',read)
 with pytest.raises(FileNotFoundError):m.run('recipe','d'*64,root=tmp_path)
 assert not(tmp_path/'evaluation').exists()


def test_prepare_validate_only_metadata_and_exact_immutable_scorer(tmp_path,monkeypatch):
 original_validate=m.validate
 source=put(tmp_path,'source.py',dict(source_fixture=True));monkeypatch.setattr(m,'sources',lambda:['source.py'])
 metric=tmp_path/m.METRIC_SOURCE;metric.parent.mkdir(parents=True,exist_ok=True);metric.write_bytes((m.ROOT/m.METRIC_SOURCE).read_bytes())
 rp=dict(output='resolved');monkeypatch.setattr(m.resolved,'validate',lambda*a,**k:rp);observation=put(tmp_path,'private-execution.json',dict(metadata_only=True));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrayreads in preparation')))
 h=m.prepare('op/recipe.json','report',resolved_protocol='resolved.json',resolved_protocol_sha256='a'*64,observation_execution=observation['path'],observation_execution_sha256=observation['sha256'],root=tmp_path);p,r=original_validate('op/recipe.json',h,root=tmp_path)
 assert p['private_archive_members']==sorted(m.evaluation.metrics.TRUTH_KEYS)and p['repair_protocol']is None and not(tmp_path/'report').exists()
 metric.write_text('Different scorer; no immutablemetric claim')
 with pytest.raises(ValueError,match='Changed input'):original_validate('op/recipe.json',h,root=tmp_path)
