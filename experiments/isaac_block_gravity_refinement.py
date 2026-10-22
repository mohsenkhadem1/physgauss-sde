"""Separate prospective native timestep audit, same mesh and physical inputs.

Runs only after complete original eight-case suite. Three fresh leftface gravity
captures at240/480/960Hz. No cross-engine constitutive/contact accuracy claim.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
try:
 from . import isaac_block_gravity_suite as original
except ImportError:
 import isaac_block_gravity_suite as original
ROOT=original.ROOT;sha=original.sha;write=original.write
PROTOCOL='runs/operator-dev/isaac-block-gravity-refinement-v01/protocol.json'
OUTPUT='runs/gaussian-campaign-v01/isaac-block-gravity-refinement-v01'
EXTRAS=['experiments/isaac_block_gravity_refinement.py','experiments/isaac_block_gravity_refinement_worker.py','tests/test_isaac_block_gravity_refinement.py']
SOURCES=original.SOURCE_NAMES+EXTRAS


def validate_stage(path,digest):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Changed temporal declaration')
 d=json.loads(p.read_text());old=d['original_protocol'];base=original.validate_protocol(old['path'],old['sha256'])
 if d['schema']!='isaac-block-gravity-temporal-v0.1' or d['rates_hz']!=[240,480,960] or d['max_query_difference_m']!=20e-6 or d['wall_cap_s']!=300:raise ValueError('Changed declared rates/gate/budget')
 if d['source_sha256']!={s:sha(ROOT/s) for s in SOURCES}:raise ValueError('Changed temporal source closure')
 return d,base


def validate_worker_protocol(path,digest):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Changed temporal worker protocol')
 d=json.loads(p.read_text());r=d['stage_protocol'];stage,base=validate_stage(r['path'],r['sha256'])
 hz=d['physics_hz'];expected={**base,'schema':'isaac-block-gravity-temporal-cell-v0.1','cases':[base['cases'][0]],'physics_hz':hz,'output':OUTPUT+'/hz'+str(hz),'source_sha256':stage['source_sha256'],'stage_protocol':r}
 if hz not in stage['rates_hz'] or d!=expected:raise ValueError('Temporal cell differs from exact physical inputs')
 return d


def temporal_metrics(captures,maxdifference=20e-6):
 if len(captures)!=3:raise ValueError('All three rates required')
 reference=captures[0];rows=[]
 for x in captures:
  for key in ['rest_m','tets','faces','fixed_node_ids','times_s']:
   np.testing.assert_array_equal(x[key],reference[key])
  if x['positions_m'].shape!=(19,len(reference['rest_m']),3) or not np.isfinite(x['positions_m']).all():raise ValueError('19finite complete frames required')
 for i,j in [(0,1),(1,2),(0,2)]:
  delta=captures[i]['positions_m'].astype(float)-captures[j]['positions_m'].astype(float)
  maximum=float(np.linalg.norm(delta,axis=-1).max());rows.append(dict(rates_hz=([240,480,960][i],[240,480,960][j]),max_fixed_material_query_difference_m=maximum,coordinate_RMS_m=float(np.sqrt(np.mean(delta**2))),passed=maximum<=maxdifference))
 return dict(pairs=rows,passed=all(r['passed'] for r in rows),threshold_m=maxdifference,query_semantics='All325identicalmaterialnodes,all19commondeclared30Hzframes; no nearestpoint/interpolation or fittedregistration')


def prepare():
 if (ROOT/PROTOCOL).exists():raise FileExistsError('Retain temporal declaration')
 old=dict(path=original.PROTOCOL,sha256=sha(ROOT/original.PROTOCOL))
 d=dict(schema='isaac-block-gravity-temporal-v0.1',status='PROSPECTIVE_BEFORE_REFINEMENT_CALLS',original_protocol=old,original_complete_execution_required=original.OUT+'/execution.json',
  rates_hz=[240,480,960],wall_cap_s=300,app_cap_s=100,source_sha256={s:sha(ROOT/s) for s in SOURCES},
  max_query_difference_m=20e-6,all_three_pairs_required=True,full_original_native_guards_retained=True,cells=[12,4,4],boundary='left',E_Pa=300000.,
  native_constitutive_equivalence_to_CPU_NOT_claimed=True,qualifications=['Native temporal discretization diagnostic only, not spatial/continuum/crossengine qualification.','PGS32 retained; iterationcount not evidence of accuracy.','No contact in temporal study.','Threefreshcaptures,no retry/changedgate after results.'])
 write(ROOT/PROTOCOL,d);return dict(path=PROTOCOL,sha256=sha(ROOT/PROTOCOL))


def run(path,digest):
 d,base=validate_stage(path,digest);execution=json.loads((ROOT/d['original_complete_execution_required']).read_text())
 if execution.get('terminal') is not True or execution.get('completed') is not True or [r['id'] for r in execution['cells']]!=[r['id'] for r in base['cases']] or any(r['state']!='completed' for r in execution['cells']):raise ValueError('Exact original8complete prerequisite')
 out=ROOT/OUTPUT
 if out.exists():raise FileExistsError('Fresh refinement output required')
 out.mkdir(parents=True);start=time.monotonic();ledger=dict(terminal=False,completed=False,controller_pid=os.getpid(),protocol=dict(path=path,sha256=digest),original_execution_sha256=sha(ROOT/d['original_complete_execution_required']),cells=[])
 write(out/'execution.json',ledger)
 for hz in d['rates_hz']:
  remaining=300-(time.monotonic()-start)
  if remaining<=0:ledger['error']='Global300sdeadline';break
  cell={**base,'schema':'isaac-block-gravity-temporal-cell-v0.1','cases':[base['cases'][0]],'physics_hz':hz,'output':OUTPUT+'/hz'+str(hz),'source_sha256':d['source_sha256'],'stage_protocol':dict(path=path,sha256=digest)}
  p=out/f'protocol-hz{hz}.json';write(p,cell);rel=p.relative_to(ROOT).as_posix();validate_worker_protocol(rel,sha(p))
  command=[base['isaac_python'],'-u',str(ROOT/'experiments/isaac_block_gravity_refinement_worker.py'),'--protocol',rel,'--protocol-sha256',sha(p),'--case','gravity_left']
  with (out/f'hz{hz}.log').open('x') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMNI_KIT_ACCEPT_EULA':'YES'})
   row=dict(hz=hz,pid=child.pid,command=command,state='running');ledger['cells'].append(row);write(out/'execution.json',ledger)
   try:code=child.wait(timeout=min(100,remaining))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;row['timeout']=True
  sp=ROOT/cell['output']/'gravity_left/summary.json';summary=json.loads(sp.read_text()) if sp.exists() else {}
  row.update(state='completed' if code==0 and summary.get('completed') is True else 'failed',exit_code=code,output=cell['output']+'/gravity_left')
  if sp.exists():row['summary_sha256']=sha(sp)
  write(out/'execution.json',ledger)
  if row['state']!='completed':break
 if len(ledger['cells'])==3 and all(r['state']=='completed' for r in ledger['cells']):
  captures=[]
  for row in ledger['cells']:
   with np.load(ROOT/row['output']/'trajectory.npz',allow_pickle=False) as f:captures.append({k:f[k] for k in ['rest_m','tets','faces','fixed_node_ids','times_s','positions_m']})
  metrics=temporal_metrics(captures);write(out/'comparison.json',dict(completed=True,**metrics,native_temporal_only=True,CPUconstitutive_equivalence=False,spatial_qualification=False));ledger['comparison_sha256']=sha(out/'comparison.json');ledger['completed']=True
 ledger.update(terminal=True,elapsed_s=time.monotonic()-start);write(out/'execution.json',ledger);return ledger

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol',default=PROTOCOL);p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r))
 if a.action=='run':raise SystemExit(0 if r['completed'] else 1)
