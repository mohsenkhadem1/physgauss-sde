"""Future guarded producer of64 static ideal-point noise-repeat prefixes.

Source only until a root-reviewed execution protocol binds qualified forward
archives AND specialised posterior controls. Opaque identity mapping is saved
before the first noise call. Existing output forbids retry/regeneration. Public
and private codecs/manifests are separate; no future truth path enters B inputs.
"""
from pathlib import Path
import argparse,hashlib,json,os,platform,resource,shutil,signal,time,traceback
import numpy as np
from experiments import campaign_c38_controlled_observations as obs
from experiments import prepare_campaign_c38_point_table as table

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c38_controlled_observations.py'
RESOURCES=dict(CPU_affinity_cores=[34,35],CPU_threads=2,RSS_cap_bytes=2*1024**3,disk_free_reserve_bytes=30*1024**3,projected_output_bytes=10*1024**2,wall_cap_s=120)
STRINGS={'information_role','object_id','observation_kind'}
BOOLS={'is_RGBD_perception','is_temporal_forecast'}
FLOATS={'prefix_force_Fz_N','known_body_gravity_m_s2','known_density_kg_m3','known_poisson_ratio'}
INTEGERS={'acquisition_indices','qualified_point_ids'}
PRIVATE_KEYS={'source_records','information_role','object_id','private_configuration_index','repeat_index','young_Pa','support_category','candidate_probe_Fz_N','candidate_probe_observations_m','canonical_target_Fz_N','canonical_target_point_ids','canonical_target_positions_m','noise_seed_recipe','common_actual_action_observations_across_policies','same_latent_reset_from_rest','independent_population_object'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(root,path):
 root=Path(root).resolve();path=Path(path).resolve()
 if not path.is_relative_to(root):raise ValueError('Repository-local record required')
 return dict(path=str(path.relative_to(root)),sha256=sha(path))
def checked(root,r):return table.checked(r,root)
def sources():return sorted(set(table.source_names())|{SOURCE,'experiments/campaign_c38_controlled_observations.py'})
def runtime():return dict(python=platform.python_version(),numpy=np.__version__)
def atomic(path,value):
 path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def encode_public(payload,*,design):
 obs.validate_public_prefix(payload,design=design);out={}
 for k,v in payload.items():
  if k in STRINGS:
   if type(v) is not str:raise ValueError('Python public string scalar required')
   out[k]=np.asarray(v)
  elif k in BOOLS:
   if type(v) is not bool:raise ValueError('Python public boolean scalar required')
   out[k]=np.asarray(v,dtype=np.bool_)
  elif k in FLOATS:
   if type(v) is not float:raise ValueError('Python public float scalar required')
   out[k]=np.asarray(v,dtype=np.float64)
  else:
   a=np.asarray(v);dtype=np.dtype('int64') if k in INTEGERS else np.dtype('float64')
   if a.dtype!=dtype:raise ValueError('Exact public array dtype required: '+k)
   out[k]=a.copy()
 return out


def decode_public_members(data,*,design):
 if set(data)!=obs.PUBLIC_KEYS:raise ValueError('Exact public NPZ allowlist required')
 result={}
 for k,a in data.items():
  a=np.asarray(a)
  if k in STRINGS:
   if a.shape!=() or a.dtype.kind!='U':raise ValueError('Scalar Unicode public string required')
   result[k]=str(a.item())
  elif k in BOOLS:
   if a.shape!=() or a.dtype!=np.bool_:raise ValueError('Scalar boolean public flag required')
   result[k]=bool(a.item())
  elif k in FLOATS:
   if a.shape!=() or a.dtype!=np.float64:raise ValueError('Scalar float64 public control required')
   result[k]=float(a.item())
  else:
   expected=np.dtype('int64') if k in INTEGERS else np.dtype('float64')
   if a.dtype!=expected:raise ValueError('Exact public array dtype required: '+k)
   result[k]=a.copy()
 return obs.validate_public_prefix(result,design=design)


def load_public_prefix(root,r,*,design):
 with np.load(checked(root,r),allow_pickle=False) as z:
  if set(z.files)!=obs.PUBLIC_KEYS:raise ValueError('Exact public prefix NPZ member allowlist required')
  return decode_public_members({k:z[k].copy() for k in z.files},design=design)


def encode_private(bundle):
 if set(bundle)!=PRIVATE_KEYS:raise ValueError('Exact PRIVATE future/latent bundle required')
 if bundle['information_role']!='PRIVATE_GENERATOR_EVALUATOR_FUTURE_AND_LATENT_NOT_MODEL_FEATURES' or np.asarray(bundle['candidate_probe_observations_m']).shape!=(2,9,3) or np.asarray(bundle['canonical_target_positions_m']).shape!=(63,3):raise ValueError('Private fixed probe/canonical-target shapes required')
 out={k:np.asarray(v) for k,v in bundle.items() if k!='source_records'}
 if any(v.dtype.kind=='O' for v in out.values()):raise ValueError('No pickle/object dtype in private archive')
 out['source_records_json']=np.asarray(json.dumps(bundle['source_records'],sort_keys=True,separators=(',',':')))
 return out


def preflight_records(root,validation_record,posterior_controls_record):
 """All prerequisite metadata/hashes checked without identities/noise/NPZ reads."""
 root=Path(root).resolve();obs.load_frozen_design(root)
 validation=json.loads(checked(root,validation_record).read_text());controls=json.loads(checked(root,posterior_controls_record).read_text())
 if validation.get('completed') is not True or validation.get('validation',{}).get('finite_checks_passed') is not True or validation['validation'].get('tolerance_m')!=1e-6:raise ValueError('All finite point-forward checks at1um required')
 candidate=validation.get('archives',{}).get('TABLE_NODE');direct=validation.get('archives',{}).get('WITHHELD_INTERPOLATION_VALIDATION')
 for r in (candidate,direct):checked(root,r)
 if controls.get('completed') is not True or controls.get('all_numerical_controls_passed') is not True or controls.get('design_sha256')!=obs.DESIGN_SHA256 or controls.get('point_table_validation_sha256')!=validation_record['sha256'] or controls.get('candidate_archive_sha256')!=candidate['sha256']:raise ValueError('Design AND exact candidate/validation-bound posterior controls required')
 return dict(design=dict(path=obs.DESIGN_PATH,sha256=obs.DESIGN_SHA256),point_validation=validation_record,posterior_controls=posterior_controls_record,candidate_archive=candidate,direct_private_archive=direct)


def prepare(protocol,output,*,point_validation,posterior_controls,root=ROOT):
 root=Path(root).resolve();destination=(root/protocol).resolve();out=(root/output).resolve()
 if not destination.is_relative_to(root) or not out.is_relative_to(root) or destination.exists() or out.exists():raise ValueError('Fresh exclusive protocol/output required; no retry')
 prerequisites=preflight_records(root,record(root,root/point_validation),record(root,root/posterior_controls))
 p=dict(schema='C38-controlled-observation-execution-v01',status='FROZEN_BEFORE_IDENTITIES_AND_OBSERVATIONS',output=str(out.relative_to(root)),source_sha256={n:sha(root/n) for n in sources()},runtime=runtime(),prerequisites=prerequisites,resources=RESOURCES,counts=dict(controlled_configurations=4,repeats_per_configuration=16,public_prefixes=64,independent_population_objects=False),identity_policy='Independent128bit opaque entropy; private mapping persisted before first noise; no regeneration/retry; public manifest orderedbyopaqueID',actual_noise_policy='Unmodified producer SeedSequence([2026102301,private_config,repeat,channel]);0gravity/1down2/2up2; one shared actual draw per object/action across policies',information_barrier='Predictor consumes only candidate archive plus PUBLIC manifest/prefixes. PRIVATE mapping/future/target/labels never part of public files.',raw_local=True,no_auto_resume_or_retry=True)
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
 return sha(destination)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();path=(root/protocol).resolve()
 if not path.is_relative_to(root) or sha(path)!=digest:raise ValueError('Changed/outside execution protocol')
 p=json.loads(path.read_text())
 if p.get('schema')!='C38-controlled-observation-execution-v01' or p.get('status')!='FROZEN_BEFORE_IDENTITIES_AND_OBSERVATIONS' or p.get('resources')!=RESOURCES or p.get('runtime')!=runtime() or p.get('counts')!=dict(controlled_configurations=4,repeats_per_configuration=16,public_prefixes=64,independent_population_objects=False) or p.get('no_auto_resume_or_retry') is not True:raise ValueError('Exact frozen64prefix/resource/runtime/no-retry design required')
 if p.get('source_sha256')!={n:sha(root/n) for n in sources()}:raise ValueError('Full current source closure required')
 old=p['prerequisites'];expected=preflight_records(root,old['point_validation'],old['posterior_controls'])
 if old!=expected:raise ValueError('Forward/posterior/archive prereq changed')
 out=Path(p.get('output',''))
 if not p.get('output') or out.is_absolute() or '..' in out.parts or out.as_posix()!=p['output']:raise ValueError('Canonical fresh output required')
 return p


def guard(root):
 if set(os.sched_getaffinity(0))!=set(RESOURCES['CPU_affinity_cores']) or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU34/35 threads2/noCUDA required')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['RSS_cap_bytes']:raise MemoryError('2GiB producer RSS cap')
 if shutil.disk_usage(root).free-RESOURCES['projected_output_bytes']<RESOURCES['disk_free_reserve_bytes']:raise OSError('30GiB+write reserve required')


def exclusive_npz(path,data):
 with Path(path).open('xb') as f:np.savez_compressed(f,**data)


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root);guard(root);out=root/p['output']
 if out.exists():raise FileExistsError('Existing output forbids rerun/resume or identity regeneration')
 public_dir=out/'PUBLIC';private_dir=out/'PRIVATE';private_dir.mkdir(parents=True,exist_ok=False);public_dir.mkdir()
 start=time.monotonic();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),identity_mapping_persisted_before_noise=False,records=[],raw_local=True,no_auto_retry=True)
 def save():state['seconds']=time.monotonic()-start;atomic(private_dir/'execution.json',state)
 def cap(*_):raise TimeoutError('Frozen120s observation producer wall cap')
 old=signal.signal(signal.SIGALRM,cap);signal.setitimer(signal.ITIMER_REAL,120);save()
 try:
  pre=p['prerequisites'];design=obs.load_frozen_design(root)
  # Prove/bind all private4 inputs before the first independent identity draw.
  bound=[obs.load_qualified_private_responses(root,pre['direct_private_archive'],pre['point_validation'],pre['posterior_controls'],private_configuration_index=i) for i in range(4)]
  ids=obs.mint_opaque_ids(64)
  mapping=[dict(object_id=ids[16*c+r],private_configuration_index=c,repeat_index=r) for c in range(4) for r in range(16)]
  identity_path=private_dir/'identity-map.json'
  with identity_path.open('x') as f:json.dump(dict(information_role='PRIVATE_IDENTITY_AND_CONFIGURATION_MAPPING',records=mapping,configurations=[dict(private_configuration_index=i,young_Pa=E,support_category=s) for i,(E,s) in enumerate(obs.controlled_configurations())],mapping_created_before_first_noise=True),f,indent=2);f.write('\n')
  state.update(identity_mapping_persisted_before_noise=True,identity_mapping=record(root,identity_path));save()
  public_records=[];groups={str(i):0 for i in range(4)}
  for row in mapping:
   guard(root);c=row['private_configuration_index'];r=row['repeat_index'];token=row['object_id']
   public,future=obs.private_repeat(c,r,token,bound[c],design=design)
   pubpath=public_dir/(token+'.npz');privpath=private_dir/(token+'.npz')
   exclusive_npz(pubpath,encode_public(public,design=design));exclusive_npz(privpath,encode_private(future))
   # Strict public round-trip catches accidental scalar/object/private leakage.
   loaded=load_public_prefix(root,record(root,pubpath),design=design)
   if loaded['object_id']!=token:raise ValueError('Public identity codec mismatch')
   public_records.append(dict(object_id=token,prefix=dict(path=pubpath.name,sha256=sha(pubpath))))
   state['records'].append({**row,'public_prefix':record(root,pubpath),'private_future':record(root,privpath)});groups[str(c)]+=1;save()
  if len(public_records)!=64 or groups!={str(i):16 for i in range(4)}:raise ValueError('All64 and four complete16repeat groups required')
  for r in state['records']:
   for k in ('public_prefix','private_future'):checked(root,r[k])
  manifest=dict(schema='C38-public-controlled-ideal-prefixes-v01',completed=True,information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',records=sorted(public_records,key=lambda r:r['object_id']),count=64,physical_configurations=4,noise_repeats_per_configuration=16,independent_population_objects=False,design_sha256=obs.DESIGN_SHA256,is_RGBD_perception=False,is_temporal_forecast=False)
  with (public_dir/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,allow_nan=False);f.write('\n')
  state.update(completed=True,private_configuration_completed_counts=groups,public_manifest=record(root,public_dir/'manifest.json'),observations_drawn=True,actual_actions_selected=False,posterior_computed=False,private_targets_released_to_predictor=False)
 except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));state['traceback']=traceback.format_exc();save();raise
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);save()
 return state

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--point-validation');a.add_argument('--posterior-controls');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,point_validation=v.point_validation,posterior_controls=v.posterior_controls))
 else:print(json.dumps({'completed':run(v.protocol,v.protocol_sha256)['completed']}))
