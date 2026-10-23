"""Prospective q63-only SIDE table export; never solve, fit data or draw noise.

All20 nonlinear capture hashes plus4 fine-linear capture/result hashes precede
point extraction. THREE table nodes fit the fixed inverse-E quadratic; TWO
withheld nodes remain private validation at unchanged1um. Finite checks are not
a continuous-material, whole-field, pixel or peak-stress certificate.
"""
from pathlib import Path
import argparse,json,os,shutil,signal,time
import numpy as np
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import run_campaign_c38_controlled_observations as npzio
from experiments import campaign_c39_side_point_forward as side
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/prepare_campaign_c39_side_points.py'
DESIGN_PATH='runs/operator-dev/c39-controlled-extension-design-v01/design.json'
DESIGN_SHA256='3555d4ad8235efa72b1c606e626cbab262a1fb7c1729b4f79f6a9fd669f0e321'
CATALOG_PATH='runs/operator-dev/c39-side-nonlinear-table-v01/catalog.json'
CATALOG_SHA256='1efbf29b9fc1b27df489fd130ee0e026990f3d21c4ff811bdfaf080cd4188746'
LINEAR_PROTOCOL_PATH='runs/operator-dev/c39-fine-linear-screen-v01/protocol.json'
LINEAR_PROTOCOL_SHA256='85a382162073b9786425faadd7883a432908cd3ae20abeb1d14f5b29f587d696'
RESOURCES=dict(CPU_affinity_cores=[34,35],CPU_threads=2,RSS_cap_bytes=2*1024**3,wall_cap_s=120,disk_free_reserve_bytes=30*1024**3,projected_output_bytes=10*1024**2,no_CUDA=True)
TOLERANCE_M=1e-6


def sources():return sorted(set(io.source_names())|{SOURCE,'experiments/campaign_c39_side_point_forward.py','experiments/run_campaign_c39_side_table.py','experiments/campaign_c39_known_actions.py','experiments/run_c39_fine_linear_screen.py','experiments/run_c39_coarse_linear_screen.py'})


def metadata_plan(root,summary_record,catalog_record,linear_protocol_record,linear_execution_record,linear_compact_record):
 io.checked(root,dict(path=DESIGN_PATH,sha256=DESIGN_SHA256))
 if catalog_record!=dict(path=CATALOG_PATH,sha256=CATALOG_SHA256)or linear_protocol_record!=dict(path=LINEAR_PROTOCOL_PATH,sha256=LINEAR_PROTOCOL_SHA256):raise ValueError('ExactfrozenSIDEcatalog/fineLINEARprotocol required')
 catalog=io.read_json(root,catalog_record);summary=io.read_json(root,summary_record)
 # Catalog is an independently frozen prospective receipt. Its role here is
 # byte identity; exact scientific rows are also reconstructed below.
 if summary.get('completed')is not True or summary.get('all10_physics_and_q63_passed')is not True or len(summary.get('rows',[]))!=10:raise ValueError('Complete10 nonlinearSIDE physical/q63 gates before extraction')
 expected={c['id']:c for c in side.side_catalog()};rows=[];seen=set()
 for r in summary['rows']:
  c=r['case'];cid=c['id']
  if cid in seen or cid not in expected or any(c.get(k)!=v for k,v in expected[cid].items())or c.get('Fz_N')!=0. or not all(r.get(k)is True for k in ('completed','physics_passed','passed')):raise ValueError('ExacttypedSIDEcase/material/BC/role, no scalarFz compiler')
  a=io.checked(root,r['source_result']);b=io.checked(root,r['external_cache'])
  if r['source_result']['sha256']!=r['external_cache']['sha256']or a.read_bytes()!=b.read_bytes():raise ValueError('Independent exactcompletedcase cache required')
  raw=json.loads(a.read_text())
  if raw.get('case')!=c or not all(raw.get(k)is True for k in ('completed','physics_passed','passed')):raise ValueError('Sourcecase metadata binding')
  meshes=r['meshes']
  if len(meshes)!=2 or {tuple(x['cells'])for x in meshes}!=set(side.MESHES):raise ValueError('Exact32/40pairs')
  for mesh in meshes:
   n=tuple(mesh['cells']);actual=[x for x in raw['meshes']if tuple(x['cells'])==n]
   if len(actual)!=1 or actual[0]['capture']!=mesh['capture']or mesh.get('completed')is not True or mesh.get('physics_passed')is not True or actual[0].get('physics_passed')is not True:raise ValueError('Source mesh/capture/physical binding')
   io.checked(root,mesh['capture']);index=2*list(expected).index(cid)+side.MESHES.index(n);rows.append(dict(source_index=index,case=c,cells=list(n),capture=mesh['capture']))
  seen.add(cid)
 if seen!=set(expected):raise ValueError('All10 scientific cases mandatory')
 lp=io.read_json(root,linear_protocol_record);le=io.read_json(root,linear_execution_record);lc=io.read_json(root,linear_compact_record)
 if lp.get('young_Pa')!=75000. or le.get('protocol_sha256')!=linear_protocol_record['sha256']or not all(le.get(k)is True for k in ('completed','terminal','all_numeric_passed','all_q63_spatial_passed'))or not all(lc.get(k)is True for k in ('completed','all_numeric_passed','all_q63_spatial_passed')):raise ValueError('CompletedfineLINEAR Eref75k exactprotocol/q63 required')
 for name,digest in lp['source_sha256'].items():
  if io.sha(io.local(root,name))!=digest:raise ValueError('FrozenfineLINEARsource changed')
 linear=[]
 if len(le['records'])!=2 or {r['support']for r in le['records']}!=set(side.CATEGORIES):raise ValueError('Exact2linearBCs')
 for r in le['records']:
  if r.get('completed')is not True or r.get('passed')is not True or r.get('exit_code')!=0:raise ValueError('Linearworkerterminal')
  io.checked(root,r['result']);cached=io.local(root,r['external_cache'])
  if io.sha(cached)!=r['result']['sha256']:raise ValueError('FineLINEARexternalcasebytes')
  matches=[s for s in lc['support_summaries']if s['LOCAL_full_response_result']==r['result']]
  if len(matches)!=1:raise ValueError('Compactlinear summary binds rawresult')
  for model in matches[0]['models']:
   n=tuple(model['cells'])
   if n not in side.MESHES:raise ValueError('FineLINEARmesh identity')
   io.checked(root,model['capture']);linear.append(dict(support=r['support'],cells=list(n),source_result=r['result'],capture=model['capture']))
 if len(linear)!=4 or len({(r['support'],tuple(r['cells']))for r in linear})!=4:raise ValueError('All4 linearSIDE fields required')
 return dict(summary=summary_record,catalog=catalog_record,nonlinear_rows=sorted(rows,key=lambda r:r['source_index']),linear_protocol=linear_protocol_record,linear_execution=linear_execution_record,linear_compact=linear_compact_record,linear_rows=sorted(linear,key=lambda r:(r['support'],r['cells'][0])))


def prepare(protocol,output,*,summary,catalog,linear_protocol,linear_execution,linear_compact,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol);out=io.local(root,output)
 if dest.exists()or out.exists():raise FileExistsError('Fresh exclusiveSIDEpointprotocol/output')
 records={k:io.record(root,io.local(root,v))for k,v in dict(summary=summary,catalog=catalog,linear_protocol=linear_protocol,linear_execution=linear_execution,linear_compact=linear_compact).items()}
 plan=metadata_plan(root,records['summary'],records['catalog'],records['linear_protocol'],records['linear_execution'],records['linear_compact'])
 p=dict(schema='C39-SIDE-point-export-v01',status='FROZEN_BEFORE_Q63_EXTRACTION',source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),design=dict(path=DESIGN_PATH,sha256=DESIGN_SHA256),metadata=plan,output=output,resources=RESOURCES,withheld_interpolation_tolerance_m=TOLERANCE_M,extract_members=['material_probes_m','probe_positions_m'],candidate_TABLE_NODE_rows=12,PRIVATE_withheld_rows=8,linear_rows=4,no_observations_or_draws=True,raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='C39-SIDE-point-export-v01'or p.get('status')!='FROZEN_BEFORE_Q63_EXTRACTION'or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('design')!=dict(path=DESIGN_PATH,sha256=DESIGN_SHA256)or p.get('withheld_interpolation_tolerance_m')!=TOLERANCE_M or p.get('extract_members')!=['material_probes_m','probe_positions_m']or p.get('no_observations_or_draws')is not True:raise ValueError('ExactSIDEqualification/source/resource protocol')
 m=p['metadata'];expected=metadata_plan(root,m['summary'],m['catalog'],m['linear_protocol'],m['linear_execution'],m['linear_compact'])
 if m!=expected:raise ValueError('SIDEpointclosure changed')
 io.local(root,p['output']);return p


def pack(rows,positions,role):
 return dict(source_index=np.array([r['source_index']for r in rows],dtype=np.int64),case_ids=np.array([r['case']['id']for r in rows]),material_use=np.array([r['case']['material_use']for r in rows]),information_role=np.asarray(role),cells=np.array([r['cells']for r in rows],dtype=np.int64),support=np.array([r['case']['support']for r in rows]),young_Pa=np.array([r['case']['young_Pa']for r in rows]),action_id=np.asarray(side.SIDE),force_vector_N=np.array([0.,2.,0.]),material_probes_m=side.POINTS.copy(),probe_positions_m=np.asarray(positions))


def extract_and_validate(root,p):
 plan=p['metadata'];rows=plan['nonlinear_rows']
 # Mandatory global20+4 bytehash barrier before ANY NPZ q-member is touched.
 for row in rows:io.checked(root,row['capture'])
 for row in plan['linear_rows']:io.checked(root,row['capture']);io.checked(root,row['source_result'])
 positions=[]
 for row in rows:
  with np.load(io.checked(root,row['capture']),allow_pickle=False)as z:points=z['material_probes_m'].copy();q=z['probe_positions_m'].copy()
  if not np.array_equal(points,side.POINTS)or q.shape!=(63,3)or not np.isfinite(q).all():raise ValueError('ExactfiniteSIDEq63 only')
  positions.append(q)
 ti=[i for i,r in enumerate(rows)if r['case']['material_use']=='TABLE_NODE'];wi=[i for i,r in enumerate(rows)if r['case']['material_use']=='WITHHELD_INTERPOLATION_VALIDATION']
 candidate=pack([rows[i]for i in ti],[positions[i]for i in ti],'C39_SIDE_CANDIDATE_TABLE_NODE_NOT_ADMITTED');private=pack([rows[i]for i in wi],[positions[i]for i in wi],'C39_SIDE_PRIVATE_WITHHELD_POINT_VALIDATION_NOT_FEATURES');checks=[]
 for cells in side.MESHES:
  f=side.SidePointForward(candidate,cells=cells)
  for i in wi:
   row=rows[i]
   if tuple(row['cells'])!=cells:continue
   c=row['case'];coef=f.observation_coefficients(c['support'],side.SIDE,np.arange(63));x=75000/c['young_Pa'];q=side.POINTS+((coef[2]*x+coef[1])*x+coef[0]).reshape(63,3);error=float(np.linalg.norm(q-positions[i],axis=1).max());checks.append(dict(case_id=c['id'],support=c['support'],young_Pa=c['young_Pa'],cells=list(cells),maximum_point_error_m=error,passed=error<=TOLERANCE_M))
 linear=[];cache={}
 for r in plan['linear_rows']:
  with np.load(io.checked(root,r['capture']),allow_pickle=False)as z:points=z['material_probes_m'].copy()
  if not np.array_equal(points,side.POINTS):raise ValueError('FineLINEAR q63 scope binding')
  key=r['source_result']['path']
  if key not in cache:cache[key]=io.read_json(root,r['source_result'])
  raw=cache[key]
  if raw.get('completed')is not True or raw.get('passed')is not True:raise ValueError('RawLINEARterminal metadata')
  models=[m for m in raw['models']if m['cells']==r['cells']]
  if len(models)!=1 or models[0]['capture']!=r['capture']:raise ValueError('Rawlinear meshcapture binding')
  responses=[a for a in models[0]['responses']if a['action']==side.SIDE]
  if len(responses)!=1:raise ValueError('ExactlySIDEresponse, nototheraction')
  metadata=responses[0]['load_metadata']
  if metadata.get('action_id')!=side.SIDE or metadata.get('force_vector_N')!=[0.,2.,0.]:raise ValueError('TypedLINEARside worldload binding')
  u=np.asarray(responses[0]['displacement63_m'],dtype=float);linear.append(dict(support=r['support'],cells=r['cells'],young_Pa=75000.,action_id=side.SIDE,force_vector_N=[0.,2.,0.],material_probes_m=side.POINTS.copy(),displacement63_m=u))
 side.LinearSidePointForward(linear)
 return candidate,private,linear,dict(completed=True,checks=checks,tolerance_m=TOLERANCE_M,finite_checks_passed=len(checks)==8 and all(r['passed']for r in checks),maximum_withheld_point_error_m=max(r['maximum_point_error_m']for r in checks),continuous_material_certificate=False)


def guard(root):
 if set(os.sched_getaffinity(0))!={34,35}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('SIDEexport CPU34/35 thread2/noCUDA')
 rss=io.current_rss(os.getpid())
 if rss is None or rss>RESOURCES['RSS_cap_bytes']:raise MemoryError('SIDEexport2GiB cap')
 if shutil.disk_usage(root).free-RESOURCES['projected_output_bytes']<RESOURCES['disk_free_reserve_bytes']:raise OSError('SIDEexport30GiB reserve')


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root);out=io.local(root,p['output'])
 if out.exists():raise FileExistsError('NoSIDEexport overwrite/retry')
 guard(root);out.mkdir(parents=True);start=time.monotonic();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),raw_local=True,no_observations_or_noise_draws=True)
 def save():state['seconds']=time.monotonic()-start;io.atomic(out/'summary.json',state)
 def cap(*a):raise TimeoutError('120sSIDEpointcap')
 old=signal.signal(signal.SIGALRM,cap);signal.alarm(120);save()
 try:
  candidate,private,linear,check=extract_and_validate(root,p);guard(root)
  a=out/'candidate-SIDE-table-points.npz';b=out/'PRIVATE-SIDE-withheld-points.npz';c=out/'linear-SIDE-point-hypotheses.npz'
  npzio.exclusive_npz(a,candidate);npzio.exclusive_npz(b,private);npzio.exclusive_npz(c,dict(information_role=np.asarray('C39_SIDE_LINEAR_HYPOTHESES_NOT_OBSERVED_LABELS'),support=np.array([r['support']for r in linear]),cells=np.array([r['cells']for r in linear]),young_Pa=np.full(4,75000.),action_id=np.asarray(side.SIDE),force_vector_N=np.array([0.,2.,0.]),material_probes_m=side.POINTS.copy(),displacement63_m=np.stack([r['displacement63_m']for r in linear])))
  state.update(completed=True,validation=check,archives=dict(CANDIDATE_SIDE=io.record(root,a),PRIVATE_WITHHELD_SIDE=io.record(root,b),LINEAR_SIDE=io.record(root,c)),source_index_rows=p['metadata']['nonlinear_rows'],finite_engineering_points_only=True);save()
 except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));save();raise
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);save()
 return state

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--summary');a.add_argument('--catalog');a.add_argument('--linear-protocol');a.add_argument('--linear-execution');a.add_argument('--linear-compact');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,summary=v.summary,catalog=v.catalog,linear_protocol=v.linear_protocol,linear_execution=v.linear_execution,linear_compact=v.linear_compact))
 else:print(json.dumps({'completed':run(v.protocol,v.protocol_sha256)['completed']}))
