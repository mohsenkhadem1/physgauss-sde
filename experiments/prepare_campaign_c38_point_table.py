"""Source-only recipe/executor for future fixed63-point table validation.

No execution protocol may be prepared until the complete40-state summary is
available. All80 capture bytes/case/cache identities are checked before any
NPZ member is extracted; only the two material-point members are ever read.
TABLE_NODE and direct withheld fields remain in separate LOCAL archives.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from experiments.campaign_c38_nonlinear_table_catalog import catalog
from experiments.run_campaign_c38_graded_family_warp import source_closure as physical_source_closure
from experiments.campaign_c38_table_validation import validate_finite_table,POINTS

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/prepare_campaign_c38_point_table.py'
SUMMARY='runs/operator-dev/c38-nonlinear-table-v01/completed-summary.json'
CACHE='runs/operator-dev/c38-nonlinear-table-v01/terminal-case-cache.json'
PREREQUISITE='runs/operator-dev/c38-nonlinear-table-v01/prerequisite.json'
TOLERANCE_M=1e-6
SOURCE_NAMES=(SOURCE,'experiments/campaign_c38_table_validation.py','experiments/campaign_c38_inverse_modulus_table.py','experiments/campaign_c38_nonlinear_table_catalog.py')
RESOURCE_LIMITS=dict(CPU_threads=2,wall_cap_s=120,RSS_cap_bytes=12*1024**3,disk_free_reserve_bytes=30*1024**3,projected_write_bytes=10*1024**2,CPU_affinity_cores=[34,35])
def source_names():return sorted(set(physical_source_closure())|set(SOURCE_NAMES))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(r,root):
 p=Path(r['path'])
 if p.is_absolute() or '..' in p.parts or p.as_posix()!=r['path']:raise ValueError('Canonical relative record required')
 p=(Path(root)/p).resolve()
 if not p.is_relative_to(Path(root).resolve()) or sha(p)!=r['sha256']:raise ValueError('Changed/outside table record')
 return p

def canonical(r):return {'path':r['path'],'sha256':r['sha256']}
def physical(c):return (c['id'],c['young_Pa'],c['support'],c['Fz_N'])

def plan_completed_rows(summary,*,root):
 """Terminal metadata only, preserving mechanical failures and material roles."""
 expected={c['id']:c for c in catalog()};rows=summary.get('rows')
 if summary.get('completed') is not True or not isinstance(rows,list) or len(rows)!=40 or {r['case']['id'] for r in rows}!=set(expected):raise ValueError('Complete exact40-state metadata required; no missing/filtering')
 fields=[]
 for row in rows:
  case=row['case'];c=expected[case['id']]
  if physical(case)!=physical(c) or case.get('material_use')!=c['material_use']:raise ValueError('Exact material/support/force identity and fixed source-index role required')
  original=checked(row['source_result'],root);cached=checked(row['external_cache'],root)
  if original.read_bytes()!=cached.read_bytes() or row.get('completed') is not True:raise ValueError('Exact independent complete case-byte cache required')
  source=json.loads(original.read_text())
  if not source.get('completed') or physical(source['case'])!=physical(c) or source.get('passed')!=row.get('passed') or source.get('physics_passed')!=row.get('physics_passed'):raise ValueError('Source case completion/identity/outcome mismatch')
  meshes=row['meshes']
  if len(meshes)!=2 or {tuple(m['cells']) for m in meshes}!={(32,16,16),(40,20,20)}:raise ValueError('Exact unique32/40 fields required')
  for m in meshes:
   cap=canonical(m['capture']);source_meshes=[s for s in source['meshes'] if s['cells']==m['cells']]
   if len(source_meshes)!=1 or canonical(source_meshes[0]['capture'])!=cap:raise ValueError('Capture not bound to original case bytes')
   if m.get('completed') is not True or type(m.get('physics_passed')) is not bool or type(row.get('passed')) is not bool or source_meshes[0].get('physics_passed')!=m['physics_passed']:raise ValueError('Explicit matching source physical/spatial outcomes required')
   fields.append(dict(source_index=2*list(expected).index(c['id'])+(0 if m['cells'][0]==32 else 1),case=c,cells=m['cells'],capture=cap,source_result=canonical(row['source_result']),external_cache=canonical(row['external_cache']),mechanics_passed=m['physics_passed'],finite_spatial_passed=row['passed']))
 return sorted(fields,key=lambda r:r['source_index'])


def verify_external_index(fields,cache_index,prerequisite):
 allowed=set()
 for r in cache_index['records']:allowed.add((r['path'],r['sha256'],r['cache']))
 for r in prerequisite['reused_case_records']:allowed.add((r['source_result']['path'],r['source_result']['sha256'],r['external_cached_result']['path']))
 for row in fields:
  a,b=row['source_result'],row['external_cache']
  if a['sha256']!=b['sha256'] or (a['path'],a['sha256'],b['path']) not in allowed:raise ValueError('Independent terminal-cache/prerequisite binding required')


def load_point_rows(fields,*,root):
 """Global hash barrier precedes first extraction; never read positions_m."""
 if len(fields)!=80:raise ValueError('All80 fields required')
 for r in fields:
  for name in ('source_result','external_cache','capture'):checked(r[name],root)
 output=[]
 for r in fields:
  with np.load(checked(r['capture'],root),allow_pickle=False) as z:
   if not {'material_probes_m','probe_positions_m'}<=set(z.files):raise ValueError('Required point members absent')
   X=z['material_probes_m'].copy();Q=z['probe_positions_m'].copy()
  if not np.array_equal(X,POINTS) or Q.shape!=(63,3) or not np.isfinite(Q).all():raise ValueError('Exact finite63-point binding required')
  output.append({**r,'material_probes_m':X,'probe_positions_m':Q})
 return output


def split_archives(rows):
 """Never promote validation nodes or refit after observing errors."""
 groups={name:[r for r in rows if r['case']['material_use']==name] for name in ('TABLE_NODE','WITHHELD_INTERPOLATION_VALIDATION')}
 if len(groups['TABLE_NODE'])!=48 or len(groups['WITHHELD_INTERPOLATION_VALIDATION'])!=32:raise ValueError('Fixed48TABLE_NODE/32directwithheld source indices required')
 result={}
 for name,group in groups.items():
  result[name]=dict(source_index=np.asarray([r['source_index'] for r in group],dtype=np.int64),case_ids=np.asarray([r['case']['id'] for r in group]),material_use=np.asarray([name]*len(group)),information_role=np.asarray('CANDIDATE_POINT_FORWARD_TABLE_NOT_ADMITTED' if name=='TABLE_NODE' else 'PRIVATE_DIRECT_WITHHELD_INTERPOLATION_VALIDATION'),cells=np.asarray([r['cells'] for r in group],dtype=np.int64),support=np.asarray([r['case']['support'] for r in group]),young_Pa=np.asarray([r['case']['young_Pa'] for r in group]),Fz_N=np.asarray([r['case']['Fz_N'] for r in group]),material_probes_m=POINTS.copy(),probe_positions_m=np.stack([r['probe_positions_m'] for r in group]))
 return result


def prepare_execution(protocol,output,*,root=ROOT):
 root=Path(root).resolve();destination=(root/protocol).resolve();out=(root/output).resolve()
 if destination.exists() or out.exists() or not destination.is_relative_to(root) or not out.is_relative_to(root):raise ValueError('Fresh exclusive repository-local protocol/output required')
 names={'summary':SUMMARY,'cache_index':CACHE,'prerequisite':PREREQUISITE};metadata={k:dict(path=n,sha256=sha(root/n)) for k,n in names.items()};objects={k:json.loads(checked(r,root).read_text()) for k,r in metadata.items()}
 fields=plan_completed_rows(objects['summary'],root=root);verify_external_index(fields,objects['cache_index'],objects['prerequisite'])
 sources=source_names()
 p=dict(schema='C38-fixed-point-table-execution-v01',status='FROZEN_METADATA_BEFORE_POINT_ARRAY_READS',metadata=metadata,fields=fields,source_sha256={n:sha(root/n) for n in sources},output=str(out.relative_to(root)),tolerance_m=TOLERANCE_M,counts=dict(states=40,fields=80,table_fields=48,withheld_fields=32),no_observations_noise_posterior_or_admission=True,source_indices_not_population_roles=True,resources=RESOURCE_LIMITS,activation='Requires root review of exact protocol SHA before run; preparation alone does not authorize reads')
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
 return sha(destination)


def validate_execution(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();path=(root/protocol).resolve()
 if not path.is_relative_to(root) or sha(path)!=digest:raise ValueError('Changed/outside execution protocol')
 p=json.loads(path.read_text())
 if p.get('schema')!='C38-fixed-point-table-execution-v01' or p.get('status')!='FROZEN_METADATA_BEFORE_POINT_ARRAY_READS' or p.get('tolerance_m')!=TOLERANCE_M or p.get('no_observations_noise_posterior_or_admission') is not True:raise ValueError('Exact fixed table/noise-free numerical recipe required')
 if set(p.get('source_sha256',{}))!=set(source_names()) or p.get('resources')!=RESOURCE_LIMITS or set(p.get('metadata',{}))!={'summary','cache_index','prerequisite'}:raise ValueError('Exact source closure/CPU budget/metadata required')
 for n,h in p['source_sha256'].items():checked(dict(path=n,sha256=h),root)
 if {k:r['path'] for k,r in p['metadata'].items()}!={'summary':SUMMARY,'cache_index':CACHE,'prerequisite':PREREQUISITE}:raise ValueError('Exact completed-table metadata paths required')
 objects={k:json.loads(checked(r,root).read_text()) for k,r in p['metadata'].items()};fields=plan_completed_rows(objects['summary'],root=root);verify_external_index(fields,objects['cache_index'],objects['prerequisite'])
 if fields!=p['fields']:raise ValueError('Frozen80-field plan changed')
 out=Path(p.get('output',''))
 if not p.get('output') or out.is_absolute() or '..' in out.parts or out.as_posix()!=p['output']:raise ValueError('Canonical execution output required')
 return p


def run_execution(protocol,digest,*,root=ROOT):
 """Future explicit call only; no invocation occurs on import or recipe writing."""
 import os,signal,resource,time,shutil
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or set(os.sched_getaffinity(0))!=set(RESOURCE_LIMITS['CPU_affinity_cores']):raise ValueError('CPU34/35 threads2/noCUDA execution required')
 root=Path(root).resolve();p=validate_execution(protocol,digest,root=root);out=(root/p['output']).resolve()
 if not out.is_relative_to(root) or out.exists():raise ValueError('Fresh local output required')
 out.mkdir(parents=True);start=time.monotonic();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),no_dataset_admission=True,raw_local=True)
 def save():
  state['seconds']=time.monotonic()-start;temporary=out/'summary.json.tmp';temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temporary.replace(out/'summary.json')
 def cap(*_):raise TimeoutError('Frozen120s point-table CPU wall cap')
 old=signal.signal(signal.SIGALRM,cap);signal.setitimer(signal.ITIMER_REAL,120);save()
 try:
  if shutil.disk_usage(root).free-RESOURCE_LIMITS['projected_write_bytes']<RESOURCE_LIMITS['disk_free_reserve_bytes']:raise OSError('30GiB+prospective archive reserve required')
  rows=load_point_rows(p['fields'],root=root)
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['resources']['RSS_cap_bytes']:raise MemoryError('12GiB point-table RSS cap')
  checks=validate_finite_table(rows,tolerance_m=TOLERANCE_M);archives=split_archives(rows);saved={}
  for role,filename in [('TABLE_NODE','candidate-table-node-points.npz'),('WITHHELD_INTERPOLATION_VALIDATION','PRIVATE-direct-withheld-points.npz')]:
   target=out/filename;np.savez_compressed(target,**archives[role]);saved[role]=dict(path=str(target.relative_to(root)),sha256=sha(target))
  state.update(completed=True,validation=checks,archives=saved,source_indices_not_population_roles=True,continuous_material_bound=False)
 except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));save();raise
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);save()
 return state


def write_recipe(destination,*,root=ROOT):
 """Prospective source-only document; never an executable data-read protocol."""
 root=Path(root).resolve();path=(root/destination).resolve()
 if not path.is_relative_to(root) or path.exists():raise ValueError('Fresh local recipe required')
 doc=dict(schema='C38-point-table-RECIPE-SOURCE-ONLY-v01',status='NOT_AN_EXECUTION_PROTOCOL_NO_ARRAY_READ_AUTHORIZATION',required_completed_summary=SUMMARY,required_cache_index=CACHE,required_reused14_prerequisite=PREREQUISITE,counts=dict(states=40,fields=80,table_source_fields=48,private_direct_withheld_fields=32),source_sha256={n:sha(root/n) for n in SOURCE_NAMES},resources=RESOURCE_LIMITS,tolerance_m=TOLERANCE_M,point_scope='Exact63 reference material points only; no full nodal/surface/image qualification',role_rules='TABLE_NODE material nodes remain candidate forward table; withheld direct nodes remain separate private numerical validation; no refit/promote/drop',execution_steps=['Wait all40 completed metadata and external caches; do not read active outputs','Prepare exact hash-bound execution protocol from completed summary/cache/prerequisite; root reviews exact SHA','Recheck entire metadata/source/capture closure before first array extraction','Read ONLY material_probes_m/probe_positions_m from each of80 captures','Retain every helper check/failure at fixed1e-6m; save separate candidate48/private32 LOCAL archives'],no_observations_noise_posterior_or_admission=True)
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
 return sha(path)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('recipe','prepare','run'));a.add_argument('--protocol');a.add_argument('--protocol-sha256');a.add_argument('--output');v=a.parse_args()
 if v.mode in ('recipe','prepare') and not v.output:a.error('--output required for recipe/prepare')
 if v.mode=='recipe':print(write_recipe(v.output))
 elif v.mode=='prepare':print(prepare_execution(v.protocol,v.output))
 else:print(json.dumps({'completed':run_execution(v.protocol,v.protocol_sha256)['completed']}))
