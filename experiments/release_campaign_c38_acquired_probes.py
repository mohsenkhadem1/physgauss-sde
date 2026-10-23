"""Trusted release of once-generated measurements AFTER all64 choices seal.

This sole private bridge extracts only saved action observations, identity,
known action forces. Reset is the frozen design convention, not a latent field.
No target/material/support member is read.
The predictor imports only the separate pure public codec/compiler. No new noise
or identities; each policy consumes one of two counterfactual public acquisitions.
"""
from pathlib import Path
import argparse,json,os,shutil,signal,time
import numpy as np
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import run_campaign_c38_controlled_observations as prefix_codec
from experiments import campaign_c38_controlled_observations as obs
from experiments.campaign_c38_public_probe_codec import encode_probe,decode_probe
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/release_campaign_c38_acquired_probes.py'
RESOURCES=dict(CPU_affinity_cores=[34,35],CPU_threads=2,RSS_cap_bytes=2*1024**3,wall_cap_s=120,no_CUDA=True,disk_free_reserve_bytes=30*1024**3,projected_output_bytes=10*1024**2)


def sources():return sorted(set(io.source_names())|{SOURCE,'experiments/campaign_c38_postprobe_inference.py','experiments/campaign_c38_public_probe_codec.py'})


def acquisition_gate(root,initial_record,choice_record):
 initial=io.read_json(root,initial_record);seal=io.read_json(root,choice_record)
 if initial.get('schema')!='C38-initial-public-inference-seal-v03' or initial.get('completed')is not True or initial.get('count')!=64 or initial.get('model_executions')!=128 or initial.get('private_truth_access')is not False or initial.get('all_predictions_sealed_before_private_evaluation')is not True:raise ValueError('Complete64x2 initialprediction seal required')
 ids=[r['object_id'] for r in initial['objects']]
 if len(ids)!=64 or len(set(ids))!=64:raise ValueError('Exact unique64 initialidentities')
 if seal.get('schema')!='C38-static-policy-choices-seal-v02' or seal.get('completed')is not True or seal.get('count')!=64 or seal.get('model')!='nonlinear_table' or seal.get('model_executions')!=128 or seal.get('private_probe_outcomes_access')is not False or seal.get('private_target_or_latent_access')is not False or seal.get('all_choices_sealed_before_private_evaluation')is not True or seal.get('initial_prediction_seal')!=initial_record:raise ValueError('Complete pre-outcome64policy choice seal required')
 choices=seal.get('choices',[])
 if len(choices)!=64 or [r['object_id'] for r in choices]!=sorted(ids):raise ValueError('All64 sorted unique policy choices required')
 for r in choices:
  c=r['choices']
  if set(c)!={'information','target_risk','fixed_down','fixed_up','random'} or c['fixed_down']!=-2. or c['fixed_up']!=2. or any(v not in (-2.,2.)for v in c.values()):raise ValueError('EqualONE-probe fivepolicy force choices required')
  result=io.read_json(root,r['result'])
  if result.get('completed')is not True or result.get('object_id')!=r['object_id']:raise ValueError('Choice result hash/identity required')
  if set(result.get('models',{}))!=set(io.MODELS)or result['models']['nonlinear_table']['choices']!=c or result.get('private_probe_outcomes_access')is not False or result.get('private_target_or_latent_access')is not False:raise ValueError('Both pre-outcome policy model results bound to choices required')
  for model in result['models'].values():io.checked(root,model['result'])
 for r in initial['objects']:
  a=io.read_json(root,r['result'])
  if a.get('completed')is not True or a.get('object_id')!=r['object_id'] or set(a.get('models',{}))!=set(io.MODELS):raise ValueError('Both initialmodel results required')
  for model in a['models'].values():
   if model.get('execution_completed')is not True or set(model.get('files',{}))!={'posterior-arrays.npz','posterior-structure.json','numerics.json'}:raise ValueError('All saved posterior outputs required before release')
   for f in model['files'].values():io.checked(root,f)
 return set(ids)


def input_plan(root,initial_record,choice_record,public_record,private_execution_record):
 ids=acquisition_gate(root,initial_record,choice_record)
 public_rows=io.public_plan(root,public_record)
 if {r['object_id']for r in public_rows}!=ids:raise ValueError('Same64 acquired public prefix identities required')
 # Metadata only. Every byte hash checked before ANY actual-probe extraction.
 private=io.read_json(root,private_execution_record)
 if private.get('completed')is not True or private.get('identity_mapping_persisted_before_noise')is not True or len(private.get('records',[]))!=64 or {r['object_id']for r in private['records']}!=ids:raise ValueError('Complete SAME once-generated64 observation records required')
 byid={r['object_id']:r['prefix']for r in public_rows};rows=[]
 for r in sorted(private['records'],key=lambda x:x['object_id']):
  if r['public_prefix']!=byid[r['object_id']]:raise ValueError('PRIVATE measurement bundle/publicprefix binding required')
  io.checked(root,r['private_future']);rows.append(dict(object_id=r['object_id'],prefix=r['public_prefix'],saved_measurements=r['private_future']))
 return rows


def prepare(protocol,output,*,initial_seal,choices_seal,public_manifest,observation_execution,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol);out=io.local(root,output)
 if dest.exists()or out.exists():raise FileExistsError('Fresh releaseprotocol/output required')
 inputs={k:io.record(root,io.local(root,v))for k,v in dict(initial=initial_seal,choices=choices_seal,public_manifest=public_manifest,observation_execution=observation_execution).items()}
 rows=input_plan(root,inputs['initial'],inputs['choices'],inputs['public_manifest'],inputs['observation_execution'])
 p=dict(schema='C38-acquired-probe-release-v01',status='FROZEN_BEFORE_PRIVATE_MEASUREMENT_RELEASE',source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),resources=RESOURCES,inputs=inputs,rows=rows,output=output,measurement_member_allowlist=['object_id','candidate_probe_Fz_N','candidate_probe_observations_m'],new_noise_draws=False,new_identity_draws=False,target_or_latent_member_access=False,one_acquisition_per_policy=True,raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='C38-acquired-probe-release-v01'or p.get('status')!='FROZEN_BEFORE_PRIVATE_MEASUREMENT_RELEASE'or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}:raise ValueError('Frozen exactrelease/source/runtime/resource protocol required')
 if p.get('new_noise_draws')is not False or p.get('new_identity_draws')is not False or p.get('target_or_latent_member_access')is not False or p.get('one_acquisition_per_policy')is not True:raise ValueError('Reuse saved measureONE perpolicy; no targets/labels/noise/identity draws')
 expected=['object_id','candidate_probe_Fz_N','candidate_probe_observations_m']
 if p.get('measurement_member_allowlist')!=expected:raise ValueError('Fixed three private-to-public measuredmembers only')
 i=p['inputs']
 if p['rows']!=input_plan(root,i['initial'],i['choices'],i['public_manifest'],i['observation_execution']):raise ValueError('Release closure changed')
 io.local(root,p['output']);return p


def extract_saved_measurements(z,object_id):
 # Do NOT iterate archive members or access any latent/target/private config.
 token=np.asarray(z['object_id']);forces=np.asarray(z['candidate_probe_Fz_N']);measurements=np.asarray(z['candidate_probe_observations_m'])
 if token.shape!=()or token.dtype.kind!='U'or str(token.item())!=object_id or forces.dtype!=np.float64 or not np.array_equal(forces,np.array([-2.,2.])) or measurements.dtype!=np.float64 or measurements.shape!=(2,9,3)or not np.isfinite(measurements).all():raise ValueError('Exact saved identity/knownforces/finite27measurements required')
 return forces.copy(),measurements.copy()


def guard(root):
 if set(os.sched_getaffinity(0))!={34,35} or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU34/35 thread2 noCUDA release guard')
 rss=io.current_rss(os.getpid())
 if rss is None or rss>RESOURCES['RSS_cap_bytes']:raise MemoryError('Release2GiB RSS cap')
 if shutil.disk_usage(root).free-RESOURCES['projected_output_bytes']<RESOURCES['disk_free_reserve_bytes']:raise OSError('Release30GiB+10MiB reserve')


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root);out=io.local(root,p['output'])
 if out.exists():raise FileExistsError('Existing release forbids overwrite/retry')
 guard(root);out.mkdir(parents=True);start=time.monotonic();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),records=[],raw_local=True,target_or_latent_members_read=False,new_noise_draws=False)
 def save():state['seconds']=time.monotonic()-start;io.atomic(out/'execution.json',state)
 def cap(*a):raise TimeoutError('120s savedprobe releasecap')
 previous=signal.signal(signal.SIGALRM,cap);signal.alarm(120);save()
 try:
  design=obs.load_frozen_design(root)
  # input_plan has hashed ALL64 savedmeasurement files and ALLchoice/posterior
  # files before the first saved outcome is extracted.
  for row in p['rows']:
   guard(root);prefix=prefix_codec.load_public_prefix(root,row['prefix'],design=design)
   with np.load(io.checked(root,row['saved_measurements']),allow_pickle=False)as z:forces,y=extract_saved_measurements(z,row['object_id'])
   acqs=[]
   for j,F in enumerate(forces):
    payload=dict(information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_PROBE',object_id=row['object_id'],observation_kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',acquisition_index=1,force_Fz_N=float(F),qualified_point_ids=prefix['qualified_point_ids'].copy(),reference_points_m=prefix['reference_points_m'].copy(),observed_positions_m=y[j].copy(),sensor_covariance_m2=prefix['sensor_covariance_m2'].copy(),reset_from_common_rest=True,is_RGBD_perception=False,is_temporal_forecast=False)
    filename=row['object_id']+('__down2.npz'if F<0 else'__up2.npz');path=out/filename
    prefix_codec.exclusive_npz(path,encode_probe(prefix,payload,design=design))
    with np.load(path,allow_pickle=False)as z:decode_probe(prefix,{k:z[k].copy()for k in z.files},design=design)
    acqs.append(dict(force_Fz_N=float(F),acquisition=dict(path=filename,sha256=io.sha(path))))
   state['records'].append(dict(object_id=row['object_id'],acquisitions=acqs));save()
  manifest=dict(schema='C38-public-acquired-probes-v01',completed=True,count=128,object_count=64,records=state['records'],initial_prediction_seal=p['inputs']['initial'],choices_seal=p['inputs']['choices'],one_acquisition_per_policy=True,counterfactual_fixed_policy_actions_share_same_saved_measurement=True,new_noise_draws=False,target_or_latent_members_included=False,is_RGBD_perception=False,is_temporal_forecast=False)
  io.write_new(out/'manifest.json',manifest);state.update(completed=True,public_acquisition_manifest=io.record(root,out/'manifest.json'));save()
 except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));save();raise
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous);save()
 return state

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--initial-seal');a.add_argument('--choices-seal');a.add_argument('--public-manifest');a.add_argument('--observation-execution');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,initial_seal=v.initial_seal,choices_seal=v.choices_seal,public_manifest=v.public_manifest,observation_execution=v.observation_execution))
 else:print(json.dumps({'completed':run(v.protocol,v.protocol_sha256)['completed']}))
