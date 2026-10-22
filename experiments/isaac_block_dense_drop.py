"""Observation-only dense240fps recapture of the frozen two drop demos."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
try:
 from . import isaac_block_gravity_suite as original
except ImportError:
 import isaac_block_gravity_suite as original
ROOT=original.ROOT;sha=original.sha;write=original.write
PROTOCOL='runs/operator-dev/isaac-block-dense-drop-v01/protocol.json';OUTPUT='runs/gaussian-campaign-v01/isaac-block-dense-drop-v01'
EXTRAS=['experiments/isaac_block_dense_drop.py','experiments/isaac_block_dense_drop_worker.py','tests/test_isaac_block_dense_drop.py']
SOURCES=original.SOURCE_NAMES+EXTRAS


def validate_protocol(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed dense-capture protocol')
 d=json.loads((ROOT/path).read_text());r=d['original_suite_protocol'];base=original.validate_protocol(r['path'],r['sha256'])
 expected={**base,'schema':'isaac-block-dense-drop-observation-v0.1','cases':[c for c in base['cases'] if c['kind']=='drop'],
  'fps':240,'output':OUTPUT,'source_sha256':{s:sha(ROOT/s) for s in SOURCES},'global_wall_cap_s':180,'app_wall_cap_s':90,
  'original_suite_protocol':r,'prerequisite_original_complete_execution':original.OUT+'/execution.json',
  'prerequisite_temporal_terminal_execution':'runs/gaussian-campaign-v01/isaac-block-gravity-refinement-v02/execution.json',
  'capture_change_only':'30to240fps;all145actualphysicsstates,nointerpolation;notindependentphysicsreplicates'}
 if d!=expected:raise ValueError('Dense capture may only change recording cadence, scope/output/deadlines')
 return d


def prepare():
 if (ROOT/PROTOCOL).exists():raise FileExistsError('Retain capture declaration')
 r=dict(path=original.PROTOCOL,sha256=sha(ROOT/original.PROTOCOL));base=original.validate_protocol(r['path'],r['sha256'])
 d={**base,'schema':'isaac-block-dense-drop-observation-v0.1','cases':[c for c in base['cases'] if c['kind']=='drop'],
  'fps':240,'output':OUTPUT,'source_sha256':{s:sha(ROOT/s) for s in SOURCES},'global_wall_cap_s':180,'app_wall_cap_s':90,
  'original_suite_protocol':r,'prerequisite_original_complete_execution':original.OUT+'/execution.json',
  'prerequisite_temporal_terminal_execution':'runs/gaussian-campaign-v01/isaac-block-gravity-refinement-v02/execution.json',
  'capture_change_only':'30to240fps;all145actualphysicsstates,nointerpolation;notindependentphysicsreplicates'}
 write(ROOT/PROTOCOL,d);return dict(path=PROTOCOL,sha256=sha(ROOT/PROTOCOL))


def dense_mechanics(rest,tets,positions,weights):
 X=np.asarray(rest,dtype=float);q=np.asarray(positions,dtype=float);w=np.asarray(weights,dtype=float);T=np.asarray(tets)
 if q.ndim!=3 or q.shape[1:]!=X.shape or not np.isfinite(q).all() or w.shape!=(len(X),) or np.any(w<=0) or abs(w.sum()-1)>1e-12:raise ValueError('Finite complete mesh trajectory and volume weights required')
 Dm=np.swapaxes(X[T[:,1:]]-X[T[:,:1]],-1,-2);inverse=np.linalg.inv(Dm)
 F=np.einsum('teij,ejk->teik',np.swapaxes(q[:,T[:,1:]]-q[:,T[:,:1]],-1,-2),inverse)
 sv=np.linalg.svd(F,compute_uv=False);strain=abs(sv-1);imax=np.unravel_index(np.argmax(strain),strain.shape)
 center0=w@X;A=X-center0;nonrigid=[]
 for frame in q:
  center=w@frame;B=frame-center;u,_,vt=np.linalg.svd(A.T@(w[:,None]*B));sign=np.eye(3);sign[2,2]=np.linalg.det(u@vt);R=u@sign@vt
  residual=B-A@R;nonrigid.append(float(np.sqrt(np.sum(w[:,None]*residual**2)/3)))
 return dict(maximum_abs_principal_stretch_minus1=float(strain.max()),principal_stretch_min=float(sv.min()),principal_stretch_max=float(sv.max()),strain_max_frame=int(imax[0]),strain_max_tet=int(imax[1]),maximum_massweighted_rigid_removed_coordinate_RMS_m=max(nonrigid),nonrigid_max_frame=int(np.argmax(nonrigid)),perframe_nonrigid_coordinate_RMS_m=nonrigid,perframe_max_abs_principal_stretch_minus1=strain.max(axis=(1,2)).tolist(),qualification='CompatibleP1F principalstretchdeparture, notstress/plasticstrain; bodymeanrigidremovedcoordinateRMS distinctfromlocalmaxstrain')


def run(path,digest):
 d=validate_protocol(path,digest)
 old=json.loads((ROOT/d['prerequisite_original_complete_execution']).read_text());prior=json.loads((ROOT/d['prerequisite_temporal_terminal_execution']).read_text())
 if old.get('completed') is not True or old.get('terminal') is not True or prior.get('terminal') is not True:raise ValueError('Originalcomplete8 and temporalterminal prerequisites required')
 out=ROOT/OUTPUT
 if out.exists():raise FileExistsError('Fresh densecapture output required')
 out.mkdir(parents=True);start=time.monotonic();ledger=dict(terminal=False,completed=False,controller_pid=os.getpid(),protocol=dict(path=path,sha256=digest),cells=[])
 write(out/'execution.json',ledger)
 for case in d['cases']:
  validate_protocol(path,digest);remaining=180-(time.monotonic()-start)
  if remaining<=0:ledger['error']='Global180sdeadline';break
  command=[d['isaac_python'],'-u',str(ROOT/'experiments/isaac_block_dense_drop_worker.py'),'--protocol',path,'--protocol-sha256',digest,'--case',case['id']]
  with (out/(case['id']+'.log')).open('x') as log:
   child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMNI_KIT_ACCEPT_EULA':'YES'})
   row=dict(id=case['id'],pid=child.pid,command=command,state='running');ledger['cells'].append(row);write(out/'execution.json',ledger)
   try:code=child.wait(timeout=min(90,remaining))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;row['timeout']=True
  sp=out/case['id']/'summary.json';s=json.loads(sp.read_text()) if sp.exists() else {}
  row.update(state='completed' if code==0 and s.get('completed') is True and s.get('frames')==145 else 'failed',exit_code=code,output=OUTPUT+'/'+case['id'])
  if sp.exists():row['summary_sha256']=sha(sp)
  if row['state']=='completed':
   with np.load(out/case['id']/'trajectory.npz',allow_pickle=False) as f:
    mass,w,_=original.mass_and_reference(f['rest_m'],f['tets']);metrics=dense_mechanics(f['rest_m'],f['tets'],f['positions_m'],w)
   write(out/case['id']/'dense-mechanics.json',dict(completed=True,**metrics,protocol=dict(path=path,sha256=digest),trajectory_sha256=sha(out/case['id']/'trajectory.npz')));row['dense_mechanics_sha256']=sha(out/case['id']/'dense-mechanics.json')
  write(out/'execution.json',ledger)
  if row['state']!='completed':break
 ledger.update(terminal=True,completed=len(ledger['cells'])==2 and all(r['state']=='completed' for r in ledger['cells']),elapsed_s=time.monotonic()-start);write(out/'execution.json',ledger);return ledger

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol',default=PROTOCOL);p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r))
 if a.action=='run':raise SystemExit(0 if r['completed'] else 1)
