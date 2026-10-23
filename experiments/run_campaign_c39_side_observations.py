"""Trusted new SIDE acquisition only, after qualified forward/all64 choices.

Reuses original64 opaque IDs, prefixes, private config/repeat mapping and target
physics. Never opens old DOWN observations/posteriors or canonical target files.
One new fixed-seed27-coordinate SIDE noise draw per identity, no regeneration.
"""
from pathlib import Path
import argparse,json,os,shutil,signal,time
import numpy as np
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import run_campaign_c38_controlled_observations as oldcodec
from experiments import campaign_c38_controlled_observations as oldobs
from experiments import prepare_campaign_c39_side_points as sideio
from experiments import campaign_c39_side_point_forward as side
from experiments import campaign_c39_side_observations as public
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c39_side_observations.py'
RESOURCES=sideio.RESOURCES.copy()
INITIAL_SEAL_PATH='runs/gaussian-campaign-v01/c38-initial-public-inference-v03/prediction-seal.json'
INITIAL_SEAL_SHA256='4b8c8dbaed0a566d4cfd72a51a60921f2384be8b78c29e2b20d416b32fdc91e7'
POLICIES={'information','target_risk','fixed_down','fixed_side','random'}


def sources():return sorted(set(sideio.sources())|{SOURCE,'experiments/campaign_c39_side_observations.py'})


def gate(root,validation_record,choices_record,public_record,original_execution_record):
 io.checked(root,dict(path=public.DESIGN_PATH,sha256=public.DESIGN_SHA256));v=io.read_json(root,validation_record)
 checks=v.get('validation',{})
 if v.get('completed')is not True or checks.get('completed')is not True or checks.get('finite_checks_passed')is not True or checks.get('tolerance_m')!=1e-6 or len(checks.get('checks',[]))!=8 or not all(r.get('passed')is True for r in checks['checks']):raise ValueError('All10SIDEphysics/all8fixed1um withheld checks before draws')
 archives=v['archives']
 if set(archives)!={'CANDIDATE_SIDE','PRIVATE_WITHHELD_SIDE','LINEAR_SIDE'}:raise ValueError('ExactSIDEcandidate/private/linear archive roles')
 for r in archives.values():io.checked(root,r)
 choices=io.read_json(root,choices_record);initial=dict(path=INITIAL_SEAL_PATH,sha256=INITIAL_SEAL_SHA256);io.checked(root,initial)
 if choices.get('schema')!='C39-typed-static-policy-choices-seal-v01'or choices.get('completed')is not True or choices.get('count')!=64 or choices.get('model')!='nonlinear_table'or choices.get('model_executions')!=128 or choices.get('extension_design')!=dict(path=public.DESIGN_PATH,sha256=public.DESIGN_SHA256)or choices.get('initial_prediction_seal')!=initial or choices.get('private_SIDE_outcomes_access')is not False or choices.get('acquired_DOWN_observations_or_posteriors_access')is not False or choices.get('private_target_or_latent_access')is not False or choices.get('all_choices_sealed_before_SIDE_noise')is not True:raise ValueError('All64 preSIDE/preDOWNY typedchoices and frozen design required')
 if choices.get('forward_point_validation')!=validation_record or choices.get('candidate_side_archive_sha256')!=archives['CANDIDATE_SIDE']['sha256']or choices.get('linear_side_archive_sha256')!=archives['LINEAR_SIDE']['sha256']:raise ValueError('ExactsameSIDEforward libraries used for choices and observationtruth')
 catalog=choices.get('action_catalog',[])
 if not isinstance(catalog,list)or len(catalog)!=2 or any(set(r)!={'action_id','force_world_N'}for r in catalog)or {r['action_id']:r['force_world_N']for r in catalog}!={'middle_down_2N':[0.,0.,-2.],side.SIDE:[0.,2.,0.]}:raise ValueError('Exacttwo typedprobe worldvectors, no scalarFz remapping')
 prefixes=io.public_plan(root,public_record);ids={r['object_id']for r in prefixes};rows=choices.get('choices',[])
 if len(rows)!=64 or [r['object_id']for r in rows]!=sorted(ids):raise ValueError('All64 originalopaque choices identities')
 modelrows=choices.get('model_choices',{})
 if set(modelrows)!=set(io.MODELS)or any(not isinstance(modelrows[name],list)or len(modelrows[name])!=64 or [r.get('object_id')for r in modelrows[name]]!=sorted(ids)for name in io.MODELS):raise ValueError('Exactall64 identities for BOTHmodel choiceclosures')
 def validate_actions(a):
  if not isinstance(a,dict)or set(a)!=POLICIES or a['fixed_down']!='middle_down_2N'or a['fixed_side']!=side.SIDE or any(x not in public.PROBE_ACTIONS for x in a.values()):raise ValueError('Fiveoneprobe typedpolicies fixedbefore noise')
 for index,r in enumerate(rows):
  a=r['choices']
  validate_actions(a)
  state=io.read_json(root,r['result'])
  if state.get('completed')is not True or state.get('object_id')!=r['object_id']or set(state.get('models',{}))!=set(io.MODELS)or state['models']['nonlinear_table']['choices']!=a:raise ValueError('Policyresult identity/source/two-model hash')
  for name in io.MODELS:
   model=state['models'][name];sealed=modelrows[name][index];validate_actions(sealed.get('choices'));validate_actions(model.get('choices'))
   if sealed['choices']!=model['choices']or sealed.get('result')!=model.get('result'):raise ValueError('ExactBOTHmodel peridentity choices/result hashbinding')
   io.checked(root,model['result'])
 original=io.read_json(root,original_execution_record)
 if original.get('completed')is not True or original.get('identity_mapping_persisted_before_noise')is not True or original.get('public_manifest')!=public_record or len(original.get('records',[]))!=64 or {r['object_id']for r in original['records']}!=ids:raise ValueError('Exactoriginal completedobservation identity source')
 oldmap=original['identity_mapping'];mapping=io.read_json(root,oldmap)
 if mapping.get('mapping_created_before_first_noise')is not True or len(mapping.get('records',[]))!=64 or {r['object_id']for r in mapping['records']}!=ids:raise ValueError('Originalprivate mapping preserved; no newIDs')
 expected_configs=[dict(private_configuration_index=i,young_Pa=E,support_category=b)for i,(E,b)in enumerate(oldobs.controlled_configurations())]
 if mapping.get('configurations')!=expected_configs:raise ValueError('Sameoriginal4physical configurations, no redraw')
 seen=set();byid={r['object_id']:r['prefix']for r in prefixes};bound=[]
 if any(r['public_prefix']!=byid[r['object_id']]for r in original['records']):raise ValueError('Exactoriginal prefixbytes/source bindings')
 for r in mapping['records']:
  c=r['private_configuration_index'];rep=r['repeat_index'];key=(c,rep)
  if type(c)is not int or not 0<=c<4 or type(rep)is not int or not 0<=rep<16 or key in seen:raise ValueError('Exactoriginal4x16 config/repeatmap')
  seen.add(key);bound.append(dict(object_id=r['object_id'],private_configuration_index=c,repeat_index=rep,prefix=byid[r['object_id']]))
 if seen!={(c,r)for c in range(4)for r in range(16)}:raise ValueError('Everyoriginalidentity retained')
 return dict(validation=validation_record,choices=choices_record,public_manifest=public_record,original_execution=original_execution_record,identity_mapping=oldmap,archives=archives,rows=sorted(bound,key=lambda r:r['object_id']))


def prepare(protocol,output,*,validation,choices,public_manifest,original_execution,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol);out=io.local(root,output)
 if dest.exists()or out.exists():raise FileExistsError('Fresh exclusiveSIDEobservationprotocol/output')
 records={k:io.record(root,io.local(root,x))for k,x in dict(validation=validation,choices=choices,public_manifest=public_manifest,original_execution=original_execution).items()};inputs=gate(root,records['validation'],records['choices'],records['public_manifest'],records['original_execution'])
 p=dict(schema='C39-trusted-SIDE-observation-execution-v01',status='FROZEN_BEFORE_NEW_SIDE_NOISE',source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),resources=RESOURCES,extension_design=dict(path=public.DESIGN_PATH,sha256=public.DESIGN_SHA256),inputs=inputs,output=output,new_noise_seed_recipe='SeedSequence([2026102305,original_private_config,original_repeat,0])',same_original64IDs_and_prefix=True,no_old_DOWN_observation_or_posterior_access=True,no_new_DOWN_noise=True,no_private_canonical_target_access=True,noise_sigma_m=.0002,retries=0,raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='C39-trusted-SIDE-observation-execution-v01'or p.get('status')!='FROZEN_BEFORE_NEW_SIDE_NOISE'or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('extension_design')!=dict(path=public.DESIGN_PATH,sha256=public.DESIGN_SHA256)or p.get('noise_sigma_m')!=.0002 or p.get('retries')!=0 or p.get('new_noise_seed_recipe')!='SeedSequence([2026102305,original_private_config,original_repeat,0])':raise ValueError('FrozenexactSIDEsource/runtime/prior/noise/resources')
 for k in ('same_original64IDs_and_prefix','no_old_DOWN_observation_or_posterior_access','no_new_DOWN_noise','no_private_canonical_target_access'):
  if p.get(k)is not True:raise ValueError('Original-data reuse/newSIDEonly boundary')
 i=p['inputs']
 if i!=gate(root,i['validation'],i['choices'],i['public_manifest'],i['original_execution']):raise ValueError('CompleteSIDEgate closure')
 io.local(root,p['output']);return p


def load_SIDE_configurations(root,r):
 with np.load(io.checked(root,r),allow_pickle=False)as z:
  if set(z.files)!=side.ARCHIVE_KEYS:raise ValueError('ExactprivateWITHHELD SIDE q63 archive')
  a={k:z[k].copy()for k in side.ARCHIVE_KEYS}
 if a['information_role'].shape!=()or str(a['information_role'])!='C39_SIDE_PRIVATE_WITHHELD_POINT_VALIDATION_NOT_FEATURES'or a['action_id'].shape!=()or str(a['action_id'])!=side.SIDE or not np.array_equal(a['force_vector_N'],[0.,2.,0.])or not np.array_equal(a['material_probes_m'],side.POINTS)or a['probe_positions_m'].shape!=(8,63,3)or not np.isfinite(a['probe_positions_m']).all():raise ValueError('Finite8WITHHELD SIDE fields, nevercanonical targets')
 expected={2*i+j:(c,n)for i,c in enumerate(side.side_catalog())if c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION'for j,n in enumerate(side.MESHES)}
 if a['source_index'].shape!=(8,)or a['source_index'].dtype.kind not in'iu'or a['cells'].shape!=(8,3)or a['cells'].dtype.kind not in'iu'or len(set(a['source_index'].tolist()))!=8 or set(a['source_index'].tolist())!=set(expected):raise ValueError('Exact8SIDEwithheld sources')
 result={}
 for i,index in enumerate(a['source_index']):
  c,n=expected[int(index)]
  if str(a['case_ids'][i])!=c['id']or str(a['support'][i])!=c['support']or a['young_Pa'][i]!=c['young_Pa']or str(a['material_use'][i])!='WITHHELD_INTERPOLATION_VALIDATION'or tuple(a['cells'][i])!=n:raise ValueError('ExactSIDEtruth/source/role/meshbinding')
  if n==(40,20,20):result[(c['young_Pa'],c['support'])]=a['probe_positions_m'][i].copy()
 configs=oldobs.controlled_configurations()
 if set(result)!=set(configs):raise ValueError('Exactsamefouroriginalphysical configs')
 return tuple(result[k]for k in configs)


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root);sideio.guard(root);out=io.local(root,p['output'])
 if out.exists():raise FileExistsError('ExistingSIDEoutput forbidsnoise regeneration/retry')
 priv=out/'PRIVATE';pub=out/'PUBLIC';priv.mkdir(parents=True);pub.mkdir();start=time.monotonic();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),records=[],same_original_identity_mapping=p['inputs']['identity_mapping'],mapping_bound_before_first_SIDE_noise=True,old_DOWN_observations_or_posteriors_read=False,canonical_targets_read=False,new_IDs_drawn=False,new_DOWN_noise_drawn=False,raw_local=True)
 def save():state['seconds']=time.monotonic()-start;io.atomic(priv/'execution.json',state)
 def cap(*a):raise TimeoutError('SIDEobservation120scap')
 prior=signal.signal(signal.SIGALRM,cap);signal.alarm(120);save()
 try:
  # All64choices and allforward/source hashes have already passed. Private
  # fields here are ONLY the new SIDE physicalresponses, not held targetfields.
  configs=load_SIDE_configurations(root,p['inputs']['archives']['PRIVATE_WITHHELD_SIDE']);design=oldobs.load_frozen_design(root);records=[];groups={str(i):0 for i in range(4)}
  for row in p['inputs']['rows']:
   sideio.guard(root);token=row['object_id'];c=row['private_configuration_index'];r=row['repeat_index'];prefix=oldcodec.load_public_prefix(root,row['prefix'],design=design)
   truth=configs[c][oldobs.POINT_IDS];error=public.side_measurement_error(c,r);observation=truth+error;payload=public.public_probe(prefix,side.SIDE,observation,design=design);path=pub/(token+'__side2.npz');oldcodec.exclusive_npz(path,public.encode(prefix,payload,design=design))
   with np.load(path,allow_pickle=False)as z:public.decode(prefix,{k:z[k].copy()for k in z.files},design=design)
   records.append(dict(object_id=token,acquisition=dict(path=path.name,sha256=io.sha(path))));groups[str(c)]+=1;state['records'].append(dict(**row,public_acquisition=io.record(root,path)));save()
  if len(records)!=64 or groups!={str(i):16 for i in range(4)}:raise ValueError('All64SIDEobservations retained')
  manifest=dict(schema='C39-public-acquired-SIDE-probes-v01',completed=True,count=64,records=records,extension_design=p['extension_design'],choices_seal=p['inputs']['choices'],action_id=side.SIDE,force_vector_N=[0.,2.,0.],same_original_identities_and_prefixes=True,new_SIDE_noise_only=True,no_new_DOWN_noise=True,private_labels_or_targets_included=False,is_RGBD_perception=False,is_temporal_forecast=False)
  io.write_new(pub/'manifest.json',manifest);state.update(completed=True,public_manifest=io.record(root,pub/'manifest.json'),private_configuration_counts=groups);save()
 except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));save();raise
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,prior);save()
 return state

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--validation');a.add_argument('--choices');a.add_argument('--public-manifest');a.add_argument('--original-execution');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,validation=v.validation,choices=v.choices,public_manifest=v.public_manifest,original_execution=v.original_execution))
 else:print(json.dumps({'completed':run(v.protocol,v.protocol_sha256)['completed']}))
