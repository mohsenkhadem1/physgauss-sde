"""New bounded high-rate native PGS numerical refinement, physical inputs fixed."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
try:
 from . import isaac_solver_ablation_v02 as previous
except ImportError:
 import isaac_solver_ablation_v02 as previous
ROOT=previous.ROOT;old=previous.old;sha=previous.sha;write=previous.write
PROTOCOL='runs/operator-dev/isaac-solver-refinement-v03/protocol.json'
OUTPUT='runs/gaussian-campaign-v01/isaac-solver-refinement-v03'
NEW=['experiments/isaac_solver_refinement_v03.py','experiments/isaac_solver_refinement_worker_v03.py','tests/test_isaac_solver_refinement_v03.py']
GPU_UUID='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb'
PREVIOUS_OUT=previous.OUTPUT

def cases():
 return [dict(id=f'gravity_left_hz{hz}_it{iterations}',kind='gravity',boundary='left',young_Pa=300000.,duration_s=.6,hz=hz,iterations=iterations) for hz,iterations in [(960,128),(1920,128),(3840,128),(3840,255)]]

def record(path):return dict(path=path,sha256=sha(ROOT/path))
def source_map():
 p=json.loads((ROOT/previous.PROTOCOL).read_text())
 return {name:sha(ROOT/name) for name in list(p['source_sha256'])+NEW}

def validate_stage(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed prospective refinement protocol')
 d=json.loads((ROOT/path).read_text())
 if d['schema']!='isaac-pgs-highrate-refinement-v03' or d['cases']!=cases() or d['gpu_uuid']!=GPU_UUID or d['wall_cap_s']!=600 or d['app_cap_s']!=180 or d['threshold_m']!=20e-6:raise ValueError('Changed refinement scope')
 if d['source_sha256']!=source_map():raise ValueError('Changed frozen source closure')
 for r in d['previous'].values():
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Prior completed evidence changed')
 p,base=previous.validate_stage(d['previous']['protocol']['path'],d['previous']['protocol']['sha256'])
 execution=json.loads((ROOT/d['previous']['execution']['path']).read_text())
 if execution.get('terminal') is not True or execution.get('completed') is not True or len(execution['cells'])!=4 or any(r['state']!='completed' for r in execution['cells']):raise ValueError('All4 previous cells must be terminalcomplete')
 return d,base

def cell_protocol(stage,base,case,r):
 return {**base,'schema':'isaac-pgs-highrate-cell-v03','cases':[case],'physics_hz':case['hz'],'deform_iterations':case['iterations'],'gpu':1,
  'CUDA_VISIBLE_DEVICES':GPU_UUID,'output':OUTPUT,'source_sha256':stage['source_sha256'],'stage_protocol':r}

def validate_worker_protocol(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed worker protocol')
 cell=json.loads((ROOT/path).read_text());r=cell['stage_protocol'];stage,base=validate_stage(r['path'],r['sha256'])
 if len(cell.get('cases',[]))!=1 or cell['cases'][0] not in cases() or cell!=cell_protocol(stage,base,cell['cases'][0],r):raise ValueError('Undeclared cell or physical setting')
 return cell

def compare(captures):
 expected={(480,128),(960,128),(1920,128),(3840,128),(3840,255)}
 if set(captures)!=expected:raise ValueError('Exact fourfresh plusoneprevious reference required')
 first=captures[(480,128)]
 for x in captures.values():
  for k in ('rest_m','tets','faces','fixed_node_ids','times_s'):np.testing.assert_array_equal(x[k],first[k])
  if x['positions_m'].shape!=(19,len(first['rest_m']),3) or not np.isfinite(x['positions_m']).all():raise ValueError('Complete finite19frames required')
 rows=[]
 for a,b,kind,gate in [((480,128),(960,128),'coarser_temporal_context',False),((960,128),(1920,128),'temporal',True),((1920,128),(3840,128),'temporal',True),((3840,128),(3840,255),'iterations',True)]:
  delta=captures[a]['positions_m'].astype(float)-captures[b]['positions_m'].astype(float);distance=np.linalg.norm(delta,axis=-1);t,n=np.unravel_index(distance.argmax(),distance.shape)
  rows.append(dict(a=list(a),b=list(b),kind=kind,required_gate=gate,maximum_material_query_difference_m=float(distance[t,n]),coordinate_RMS_m=float(np.sqrt(np.mean(delta**2))),maximum_at_time_s=float(first['times_s'][t]),maximum_at_node=int(n),passed=bool(distance[t,n]<=20e-6)))
 return dict(completed=True,threshold_m=20e-6,comparisons=rows,finest_declared_numerical_checks_passed=all(r['passed'] for r in rows if r['required_gate']),
  prior480_context_does_not_replace_finest_checks=True,no_extrapolated_rigorous_error_bound=True,no_residual_or_continuum_or_constitutive_equivalence_claim=True)

def prepare():
 if (ROOT/PROTOCOL).exists():raise FileExistsError('Preserve prior declaration')
 prior=dict(protocol=record(previous.PROTOCOL),execution=record(PREVIOUS_OUT+'/execution.json'),trajectory=record(PREVIOUS_OUT+'/gravity_left_hz480_it128/trajectory.npz'),summary=record(PREVIOUS_OUT+'/gravity_left_hz480_it128/summary.json'))
 d=dict(schema='isaac-pgs-highrate-refinement-v03',status='FROZEN_BEFORE_FOUR_NEW_APPS',cases=cases(),previous=prior,source_sha256=source_map(),
  gpu=1,gpu_uuid=GPU_UUID,CPU_threads=2,wall_cap_s=600,app_cap_s=180,threshold_m=20e-6,
  exact_same_physics='Original C36 leftface gravity,E300000/nu.35/rho625/mesh12x4x4/duration.6s/PGS/allphysicaldampingzero/nohandles/nocontact',
  settings='UUID CVD,active_cuda_gpus[0],physics0,Worldcuda0,limit_cpu_threads2; samev02 resourceguard',
  gates='Full19commonframes/all325fixedmaterialqueries:960-1920 AND1920-3840 at128iterations;3840/128-255 separateiterationcontrol;eachmax<=20um',
  no_extrapolated_error_bound=True,no_retry_no_unlisted_cases=True,prior_accuracy_failures_preserved=True,
  runtime_estimate_s=[210,250],budget='600sglobal/180sperapp,processgroupcleanup;abortnonfinite/inversion/nativeguardfailure')
 write(ROOT/PROTOCOL,d);return record(PROTOCOL)

def load_capture(path):
 with np.load(ROOT/path,allow_pickle=False) as f:return {k:f[k] for k in ['rest_m','tets','faces','fixed_node_ids','times_s','positions_m']}

def run(path,digest):
 stage,base=validate_stage(path,digest);out=ROOT/OUTPUT
 if out.exists():raise FileExistsError('Fresh output only')
 if any(r.split(',')[1].strip()==GPU_UUID for r in previous.gpu_apps().splitlines() if ',' in r):raise RuntimeError('GPU1 notidle')
 out.mkdir(parents=True);start=time.monotonic();ledger=dict(terminal=False,completed=False,pid=os.getpid(),protocol=dict(path=path,sha256=digest),cells=[])
 write(out/'execution.json',ledger)
 try:
  for c in cases():
   validate_stage(path,digest);remaining=600-(time.monotonic()-start)
   if remaining<=0:raise TimeoutError('Global600s cap')
   cp=out/(c['id']+'-protocol.json');write(cp,cell_protocol(stage,base,c,dict(path=path,sha256=digest)));rel=cp.relative_to(ROOT).as_posix();validate_worker_protocol(rel,sha(cp))
   command=[base['isaac_python'],'-u',str(ROOT/NEW[1]),'--protocol',rel,'--protocol-sha256',sha(cp)]
   command+=['--case',c['id']];row=dict(id=c['id'],command=command,state='running',output=OUTPUT+'/'+c['id']);ledger['cells'].append(row)
   with (out/(c['id']+'.log')).open('x') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU_UUID,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMNI_KIT_ACCEPT_EULA':'YES'})
    row['pid']=child.pid;write(out/'execution.json',ledger)
    try:code=child.wait(timeout=min(180,remaining))
    except BaseException:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=3)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     row.update(exit_code=child.returncode,state='failed');raise
   sp=ROOT/row['output']/'summary.json';s=json.loads(sp.read_text()) if sp.exists() else {}
   row.update(exit_code=code,state='completed' if code==0 and s.get('completed') is True else 'failed')
   if sp.exists():row['summary_sha256']=sha(sp)
   write(out/'execution.json',ledger)
   if row['state']!='completed':raise RuntimeError('Preserved failed cell; stop new launches')
  captures={(480,128):load_capture(stage['previous']['trajectory']['path'])}
  for c in cases():captures[(c['hz'],c['iterations'])]=load_capture(OUTPUT+'/'+c['id']+'/trajectory.npz')
  result=compare(captures);write(out/'comparison.json',result);ledger.update(completed=True,comparison_sha256=sha(out/'comparison.json'))
 except BaseException as exc:ledger['error']=type(exc).__name__+': '+str(exc);raise
 finally:ledger.update(terminal=True,elapsed_s=time.monotonic()-start,source_pins_unchanged=stage['source_sha256']==source_map());write(out/'execution.json',ledger)
 return ledger

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol',default=PROTOCOL);p.add_argument('--sha256');a=p.parse_args();print(json.dumps(prepare() if a.action=='prepare' else run(a.protocol,a.sha256)))
