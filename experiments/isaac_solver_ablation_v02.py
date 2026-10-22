"""Prospective GPU1-only native PGS iteration/time causal ablation.

No simulation at import. Four fresh controls; retain all old C36 failures.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
try:
 from . import isaac_block_gravity_suite as old
except ImportError:
 import isaac_block_gravity_suite as old

ROOT=old.ROOT;sha=old.sha;write=old.write
PROTOCOL='runs/operator-dev/isaac-solver-ablation-v02/protocol.json'
OUTPUT='runs/gaussian-campaign-v01/isaac-solver-ablation-v02'
NEW=['experiments/isaac_solver_ablation_v02.py','experiments/isaac_solver_ablation_worker_v02.py','tests/test_isaac_solver_ablation_v02.py']
OLD_STAGE='runs/operator-dev/isaac-block-gravity-refinement-v02/protocol.json'

def cases():
 return [dict(id=f'gravity_left_hz{hz}_it{iterations}',kind='gravity',boundary='left',young_Pa=300000.,duration_s=.6,hz=hz,iterations=iterations) for hz in (240,480) for iterations in (32,128)]

def source_map():
 stage=json.loads((ROOT/OLD_STAGE).read_text())
 return {name:sha(ROOT/name) for name in list(stage['source_sha256'])+NEW}

def validate_stage(path,digest):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Prospective protocol changed')
 d=json.loads(p.read_text())
 if d['schema']!='isaac-pgs-iteration-time-ablation-v02' or d['cases']!=cases() or d['gpu']!=1 or d['CUDA_VISIBLE_DEVICES']!='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb' or d['wall_cap_s']!=300 or d['app_cap_s']!=100 or d['threshold_m']!=20e-6:raise ValueError('Changed prospective scope')
 if d['source_sha256']!=source_map():raise ValueError('Frozen source changed')
 if sha(ROOT/d['old_stage']['path'])!=d['old_stage']['sha256']:raise ValueError('Old stage changed')
 if sha(ROOT/d['preserved_resource_failure']['path'])!=d['preserved_resource_failure']['sha256']:raise ValueError('Preserved resource failure changed')
 base=old.validate_protocol(d['original_protocol']['path'],d['original_protocol']['sha256'])
 return d,base

def cell_protocol(stage,base,case,record):
 return {**base,'schema':'isaac-pgs-iteration-time-cell-v02','cases':[case],'physics_hz':case['hz'],'deform_iterations':case['iterations'],
  'gpu':1,'CUDA_VISIBLE_DEVICES':'GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','output':OUTPUT,'source_sha256':stage['source_sha256'],'stage_protocol':record}

def validate_worker_protocol(path,digest):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Worker protocol changed')
 cell=json.loads(p.read_text());r=cell['stage_protocol'];stage,base=validate_stage(r['path'],r['sha256'])
 if len(cell.get('cases',[]))!=1 or cell['cases'][0] not in cases() or cell!=cell_protocol(stage,base,cell['cases'][0],r):raise ValueError('Undeclared physical/solver cell')
 return cell

def compare(captures,threshold=20e-6):
 if set(captures)!={(h,i) for h in (240,480) for i in (32,128)}:raise ValueError('All four cells required')
 first=next(iter(captures.values()))
 for x in captures.values():
  for k in ('rest_m','tets','faces','fixed_node_ids','times_s'):np.testing.assert_array_equal(x[k],first[k])
  if x['positions_m'].shape!=(19,len(first['rest_m']),3) or not np.isfinite(x['positions_m']).all():raise ValueError('Complete finite matching19frames required')
 def pair(a,b,kind):
  delta=captures[a]['positions_m'].astype(float)-captures[b]['positions_m'].astype(float)
  distance=np.linalg.norm(delta,axis=-1);t,n=np.unravel_index(distance.argmax(),distance.shape)
  return dict(a=list(a),b=list(b),kind=kind,maximum_material_query_difference_m=float(distance[t,n]),coordinate_RMS_m=float(np.sqrt(np.mean(delta**2))),
   maximum_at_time_s=float(first['times_s'][t]),maximum_at_node=int(n),passed=bool(distance[t,n]<=threshold))
 rows=[pair((h,32),(h,128),'iterations_at_fixed_dt') for h in (240,480)]+[pair((240,i),(480,i),'dt_at_fixed_iterations') for i in (32,128)]
 return dict(completed=True,threshold_m=threshold,comparisons=rows,all_pairs_pass=all(r['passed'] for r in rows),
  no_solver_residual_available=True,no_continuum_or_constitutive_equivalence_claim=True,iteration_improvement_not_assumed=True)

def prepare():
 p=ROOT/PROTOCOL
 if p.exists():raise FileExistsError('Preserve prospective protocol')
 d=dict(schema='isaac-pgs-iteration-time-ablation-v02',status='FROZEN_BEFORE_GPU_CALLS',cases=cases(),gpu=1,CUDA_VISIBLE_DEVICES='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb',CPU_threads=2,
  wall_cap_s=300,app_cap_s=100,threshold_m=20e-6,source_sha256=source_map(),original_protocol=dict(path=old.PROTOCOL,sha256=sha(ROOT/old.PROTOCOL)),
  old_stage=dict(path=OLD_STAGE,sha256=sha(ROOT/OLD_STAGE)),output=OUTPUT,
  preserved_resource_failure=dict(path='runs/operator-dev/isaac-solver-ablation-v01/failed-terminal.json',sha256=sha(ROOT/'runs/operator-dev/isaac-solver-ablation-v01/failed-terminal.json')),
  resource_correction='UUID CVD + active_cuda_gpus[0]/physics_gpu0/Worldcuda0/limit_cpu_threads2; no physical settings changed',
  same_physics=dict(size_m=[.24,.08,.08],cells=[12,4,4],rho_kg_m3=625.,E_Pa=300000.,poisson=.35,gravity_m_s2=[0,0,-9.81],
   boundary='Exactfixedleftface',solver='PGS',linear_damping=0,elasticity_damping=0,settling_damping=0,sleep_and_settling_thresholds=0,
   self_collision=False,no_floor_no_handles_no_force_callbacks=True,external_forces_every_iteration=False),
  native_verification='Preserve C36 rest/tet/E/nu/flags/gravity/clock/initialstate/Jpositive/fixeddrift guards; OWN PID solelyphysicalGPU1',
  interpretation='Iteration32to128 and240to480 at fixedphysics. Any improvement diagnostic only;128 not automatically converged.',
  old_GPU0_32_reproduction='Compare atmatchingHz separately, not reused asGPU1control',all_four_fresh=True,no_retries_or_extra_cells=True)
 write(p,d);return dict(path=PROTOCOL,sha256=sha(p))

def gpu_apps():return subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout

def run(path,digest):
 stage,base=validate_stage(path,digest);out=ROOT/OUTPUT
 if out.exists():raise FileExistsError('Fresh ablation output required')
 enumeration=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
 gpu1=next(r.split(',')[1].strip() for r in enumeration.splitlines() if r.split(',')[0].strip()=='1')
 if any(r.split(',')[1].strip()==gpu1 for r in gpu_apps().splitlines() if ',' in r):raise RuntimeError('GPU1 notidle; do not disturb unrelated work')
 out.mkdir(parents=True);start=time.monotonic();ledger=dict(terminal=False,completed=False,pid=os.getpid(),protocol=dict(path=path,sha256=digest),gpu1_uuid=gpu1,cells=[])
 write(out/'execution.json',ledger)
 try:
  for c in cases():
   validate_stage(path,digest);remaining=300-(time.monotonic()-start)
   if remaining<=0:raise TimeoutError('Global300s cap')
   cp=out/(c['id']+'-protocol.json');write(cp,cell_protocol(stage,base,c,dict(path=path,sha256=digest)));rel=cp.relative_to(ROOT).as_posix();validate_worker_protocol(rel,sha(cp))
   command=[base['isaac_python'],'-u',str(ROOT/NEW[1]),'--protocol',rel,'--protocol-sha256',sha(cp),'--case',c['id']]
   row=dict(id=c['id'],command=command,state='running',output=OUTPUT+'/'+c['id']);ledger['cells'].append(row)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':'GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMNI_KIT_ACCEPT_EULA':'YES'}
   with (out/(c['id']+'.log')).open('x') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);row['pid']=child.pid;write(out/'execution.json',ledger)
    try:code=child.wait(timeout=min(100,remaining))
    except BaseException:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=3)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     row['exit_code']=child.returncode;row['state']='failed';raise
   sp=ROOT/row['output']/'summary.json';summary=json.loads(sp.read_text()) if sp.exists() else {}
   row.update(exit_code=code,state='completed' if code==0 and summary.get('completed') is True else 'failed')
   if sp.exists():row['summary_sha256']=sha(sp)
   write(out/'execution.json',ledger)
   if row['state']!='completed':raise RuntimeError('Preserved failed app; stop new launches')
  captures={}
  for c in cases():
   with np.load(out/c['id']/'trajectory.npz',allow_pickle=False) as f:captures[(c['hz'],c['iterations'])]={k:f[k] for k in ['rest_m','tets','faces','fixed_node_ids','times_s','positions_m']}
  result=compare(captures);reproduction=[]
  for hz in (240,480):
   oldpath=ROOT/f'runs/gaussian-campaign-v01/isaac-block-gravity-refinement-v02/hz{hz}/gravity_left/trajectory.npz'
   with np.load(oldpath,allow_pickle=False) as f:
    for key in ('rest_m','tets','faces','fixed_node_ids','times_s'):np.testing.assert_array_equal(f[key],captures[(hz,32)][key])
    delta=captures[(hz,32)]['positions_m'].astype(float)-f['positions_m'].astype(float)
   reproduction.append(dict(hz=hz,old_gpu0_archive_sha256=sha(oldpath),maximum_difference_m=float(np.linalg.norm(delta,axis=-1).max()),coordinate_RMS_m=float(np.sqrt(np.mean(delta**2)))))
  result['old_GPU0_reproduction']=reproduction;write(out/'comparison.json',result);ledger.update(completed=True,comparison_sha256=sha(out/'comparison.json'))
 except BaseException as e:
  ledger['error']=type(e).__name__+': '+str(e);raise
 finally:
  ledger.update(terminal=True,elapsed_s=time.monotonic()-start,source_pins_unchanged=source_map()==stage['source_sha256']);write(out/'execution.json',ledger)
 return ledger

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol',default=PROTOCOL);p.add_argument('--sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.sha256);print(json.dumps(r))
