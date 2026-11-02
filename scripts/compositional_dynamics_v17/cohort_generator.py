"""PREPARE or guarded generation of a fresh four-world finite-resolution cohort.

No draw occurs in prepare. Run requires an independently approved final protocol
AND a frozen estimator recipe. Simulator labels remain under private/.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,multiprocessing,importlib.util,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.materials import teacher_energy
PROPOSAL=REPORT/'fresh-cohort-proposal.json';RAW=ROOT/'runs/compositional-dynamics-v17/fresh-cohort';RESULT=REPORT/'fresh-cohort-generation.json'
def prepare():
 if PROPOSAL.exists()or RAW.exists():raise ValueError('Preserve prior proposal/data')
 paths=closure([Path(__file__),ROOT/'scripts/compositional_dynamics_v17/prototype.py'])+[ROOT/'scripts/compositional_dynamics_v17/test_cohort_generator.py'];profile=REPORT/'cohort-teacher-profile.json'
 settings=dict(world_ids=[f'world-{i}'for i in range(4)],private_seed=17261102,balanced_families=['mooney_rivlin','mooney_rivlin','yeoh','yeoh'],log_scale_range=[-.35,.35],shape_range=[-1.,1.],contact_stiffness_N_m3_range=[2e7,8e7],dimensions_m=[.12,.06,.06],density_kg_m3=625.,mass_kg=.27,gravity_m_s2=9.8,physical_damping='none',roles=[dict(name='calibration',clearance_m=.01,duration_s=.20),dict(name='forecast-40mm',clearance_m=.04,duration_s=.30),dict(name='forecast-80mm',clearance_m=.08,duration_s=.30)],primary_n=4,sensitivity_n=6,dt_s=25e-6,half_dt_s=12.5e-6,volume_order=3,surface_order=8,sample_interval_s=.001,sensitivity_selection='First declared opaqueID of each privately assigned family; fixed before response outcomes. Calibration+80mm n6, plus80mm halfstep at BOTH n4/n6.',expected_jobs=20,cpu_workers=2,gpu_devices=['cuda:0','cuda:2'],seconds=7200,minimum_available_memory_GiB=32,gpu_minimum_free_MiB=6144,gpu_maximum_utilization_percent=80,gpu_fallback='CPU per-lane if launch-time headroom unavailable; record census, same physics, no unrelatedjobchanges',runtime=dict(python_executable='runs/compositional-dynamics-v17/render-env/bin/python',torch='2.4.0+cu121',numpy='1.26.4'),invalid_reference_policy='All original worlds/jobs/failed prefixes retained. No replacement, height reduction, survivor-only primary mean, or claim unexamined references are qualified.',scope='Fresh finite-resolution synthetic load-magnitude extrapolation only, knowngeometry/mass and oracle3Dtracks. Not continuumtruth, rawvideo inference, or interaction-type generalization.',reference_diagnostics=dict(minimum_J=.1,maximum_penetration_m=.0012,energy_drift_relative=.01,spatial_world_coordinate_RMS_m=.0003,spatial_shape_coordinate_RMS_m=.00015,temporal_world_coordinate_RMS_m=.0003,temporal_shape_coordinate_RMS_m=.00015))
 write(PROPOSAL,dict(status='PROPOSAL_ONLY_NO_DRAWS',generator_settings=settings,source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},input_sha256={str(profile.relative_to(ROOT)):sha(profile)},method_recipe=None,required_approval='Final protocol+methodfreeze+independent prefitaudit exact SHA bindings before run'))
def guard(protocol_path,freeze_path,approval_path,protocol_sha,freeze_sha):
 if sha(protocol_path)!=protocol_sha or sha(freeze_path)!=freeze_sha:raise ValueError('Requested protocol/freeze digest mismatch')
 p=json.loads(Path(protocol_path).read_text());f=json.loads(Path(freeze_path).read_text());a=json.loads(Path(approval_path).read_text())
 if not p.get('method_recipe')or f.get('protocol_sha256')!=protocol_sha:raise ValueError('Final estimator recipe/protocol binding required')
 if not a.get('passed')or a.get('protocol_sha256')!=protocol_sha or a.get('method_freeze_sha256')!=freeze_sha:raise ValueError('Independent approval exact bindings required')
 for group in ['source_sha256','input_sha256']:
  for name,h in f.get(group,{}).items():
   if sha(ROOT/name)!=h:raise ValueError('Frozen source/input changed')
  for name,h in p.get(group,{}).items():
   if f.get(group,{}).get(name)!=h or sha(ROOT/name)!=h:raise ValueError('Generator source/input omitted or changed in methodfreeze')
 if f.get('source_sha256',{}).get(str(Path(__file__).relative_to(ROOT)))!=sha(__file__):raise ValueError('Generator itself not frozen')
 return p

def private_draw(settings):
 rng=np.random.default_rng(settings['private_seed']);families=rng.permutation(settings['balanced_families']);worlds=[]
 for name,family in zip(settings['world_ids'],families):
  worlds.append(dict(name=name,family=str(family),latent=[float(rng.uniform(*settings['log_scale_range'])),float(rng.uniform(*settings['shape_range']))],contact_stiffness_N_m3=float(np.exp(rng.uniform(*np.log(settings['contact_stiffness_N_m3_range']))))))
 return worlds

def jobs(settings,worlds):
 roles={r['name']:r for r in settings['roles']};selected=[]
 for family in ['mooney_rivlin','yeoh']:selected.append(next(w['name']for w in worlds if w['family']==family))
 result=[]
 def add(world,role,n,dt,kind,lane):
  name=f'{world}/{role}/n{n}-dt{dt:.8f}';result.append(dict(id=name,world_id=world,role=role,n=n,dt_s=dt,kind=kind,lane=lane,**{k:v for k,v in roles[role].items()if k!='name'}))
 for role in roles:
  for world in settings['world_ids']:add(world,role,4,settings['dt_s'],'primary','cpu')
 for index,world in enumerate(selected):
  lane=f'gpu{index}'
  for role in ['calibration','forecast-80mm']:add(world,role,6,settings['dt_s'],'spatial_sensitivity',lane)
  add(world,'forecast-80mm',4,settings['half_dt_s'],'temporal_primary','cpu');add(world,'forecast-80mm',6,settings['half_dt_s'],'temporal_sensitivity',lane)
 if len(result)!=settings['expected_jobs']or len({j['id']for j in result})!=len(result):raise ValueError('Declared job count/identity mismatch')
 return result,selected

def worker(args):
 job,device,deadline,bindings,runtime_sha=args;torch.set_num_threads(1);p=guard(**bindings);s=p['generator_settings'];runtime_path=RAW/'private/runtime-input-freeze.json'
 if sha(runtime_path)!=runtime_sha:raise ValueError('Runtime draw manifest changed')
 runtime=json.loads(runtime_path.read_text());worldpath=RAW/'private/worlds.json'
 if sha(worldpath)!=runtime['private_worlds_sha256']:raise ValueError('Private worlds changed')
 world=next(w for w in json.loads(worldpath.read_text())['worlds']if w['name']==job['world_id']);folder=RAW/'references'/job['id'];folder.mkdir(parents=True,exist_ok=True);identity=dict(job=job,job_id=job['id'],world_id=job['world_id'],role=job['role'],n=job['n'],dt_s=job['dt_s'],device=device,protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],runtime_manifest_sha256=runtime_sha,path=str(folder.relative_to(ROOT)))
 try:
  spec=importlib.util.spec_from_file_location('private_fresh_cohort',ROOT/'scripts/compositional_dynamics_v17/prototype.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl);impl.PLAN=Path(bindings['protocol_path']);impl.RAW=folder/'work';constructor=impl.DynamicContactModel
  def configured(*a,**kw):
   kw.update(device=device,order=s['volume_order'],gravity=(0,0,-s['gravity_m_s2']),bulk_energy=lambda F,z:teacher_energy(F,z,world['family']),material_latent_field=lambda q:q.new_tensor(world['latent']).expand(*q.shape[:-1],2));return constructor(*a,**kw)
  impl.DynamicContactModel=configured;settings=dict(dimensions=s['dimensions_m'],young_Pa=12000.,poisson=.35,density=s['density_kg_m3'],floor_z=-.5*s['dimensions_m'][2]-job['clearance_m'],penalty_N_m3=world['contact_stiffness_N_m3'],compression=0.,surface_order=s['surface_order'],sample_interval_s=s['sample_interval_s'],duration=dict(drop=job['duration_s']))
  row=impl.run_case('drop',job['n'],job['dt_s'],settings,deadline);src=impl.RAW/f"drop-n{job['n']}-dt{job['dt_s']:.8f}";shutil.copyfile(src/'trajectory.npz',folder/'trajectory.npz');row['prototype_energy_scale_J']=row['energy_scale_J'];row['energy_scale_J']=max(abs(row['initial_energy_J']),s['mass_kg']*s['gravity_m_s2']*job['clearance_m'],1e-10)
  with np.load(folder/'trajectory.npz')as d:row['maximum_relative_energy_drift']=float(np.max(abs(d['ledger'][:,1]-row['initial_energy_J']))/row['energy_scale_J'])
  row.update(identity);row['finite_resolution_generator']=True;row['reference_qualified']=False;write(folder/'complete.json',row);return row
 except Exception as ex:
  row=dict(**identity,completed=False,error=f'{type(ex).__name__}: {ex}',reference_qualified=False);write(folder/'complete.json',row);return row

def public_row(row):
 allowed=['job','job_id','world_id','role','n','dt_s','device','completed','error','path','trajectory_sha256','minimum_J','maximum_dense_penetration_m','maximum_relative_energy_drift','seconds','reference_qualified','protocol_sha256','method_freeze_sha256','runtime_manifest_sha256']
 return {k:row[k]for k in allowed if k in row}
def run(bindings):
 p=guard(**bindings);s=p['generator_settings']
 if RAW.exists():raise ValueError('No replacement/reuse of existing fresh cohort')
 if torch.__version__!=s['runtime']['torch']or np.__version__!=s['runtime']['numpy']:raise ValueError('Pinned generation runtime required')
 available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
 if available<s['minimum_available_memory_GiB']*2**30:raise RuntimeError('Insufficient memory headroom')
 census=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);gpu={int(l.split(',')[0]):[int(x)for x in l.split(',')[1:]]for l in census.splitlines()};devices={'cpu':'cpu'}
 for i,name in enumerate(s['gpu_devices']):
  index=int(name.split(':')[1]);free,util=gpu[index];devices[f'gpu{i}']=name if free>=s['gpu_minimum_free_MiB']and util<=s['gpu_maximum_utilization_percent']else'cpu'
 RAW.mkdir(parents=True);worlds=private_draw(s);worldpath=RAW/'private/worlds.json';write(worldpath,dict(protocol_sha256=bindings['protocol_sha'],worlds=worlds));tasks,subset=jobs(s,worlds);runtime=RAW/'private/runtime-input-freeze.json';write(runtime,dict(protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],private_worlds_sha256=sha(worldpath),jobs=tasks,sensitivity_world_ids=subset,devices=devices,census=census,memory_available_bytes=available));runtime_sha=sha(runtime)
 start=time.monotonic();deadline=start+s['seconds']-45;report=dict(protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],runtime_manifest_sha256=runtime_sha,expected_jobs=len(tasks),expected_keys=[j['id']for j in tasks],rows=[],terminal=False,scope=s['scope']);ctx=multiprocessing.get_context('spawn');pools={lane:ctx.Pool(s['cpu_workers']if lane=='cpu'else 1)for lane in devices};pending=[]
 try:
  for job in tasks:pending.append((job,pools[job['lane']].apply_async(worker,((job,devices[job['lane']],deadline-20,bindings,runtime_sha),))))
  while pending:
   if time.monotonic()>=deadline:raise TimeoutError('Joint cohort cap; missing jobs retained')
   remain=[]
   for job,future in pending:
    if future.ready():report['rows'].append(public_row(future.get()));write(RESULT,report)
    else:remain.append((job,future))
   pending=remain
   if pending:time.sleep(.5)
 except Exception as ex:report['error']=f'{type(ex).__name__}: {ex}'
 finally:
  for pool in pools.values():pool.terminate();pool.join()
  rows=[]
  for job in tasks:
   f=RAW/'references'/job['id']/'complete.json'
   if f.exists():
    r=json.loads(f.read_text())
    if r['job']!=job or r['protocol_sha256']!=bindings['protocol_sha']or r['runtime_manifest_sha256']!=runtime_sha:raise ValueError('Harvest identity mismatch')
    rows.append(public_row(r))
  guard(**bindings);report.update(rows=rows,terminal=True,completed=sum(r['completed']for r in rows),missing_keys=[j['id']for j in tasks if j['id']not in{r['job_id']for r in rows}],seconds=time.monotonic()-start);write(RESULT,report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
 for name in ['protocol_path','freeze_path','approval_path','protocol_sha','freeze_sha']:ap.add_argument('--'+name.replace('_','-'))
 args=ap.parse_args()
 if args.prepare:prepare()
 else:
  bindings={k:getattr(args,k)for k in ['protocol_path','freeze_path','approval_path','protocol_sha','freeze_sha']}
  if not all(bindings.values()):ap.error('Exact final protocol, methodfreeze, independent approval and SHA bindings required')
  run(bindings)
