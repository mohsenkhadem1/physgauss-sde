"""NEW eight-case Isaac rectangular-block gravity/drop engineering suite.

Pure geometry/config helpers and bounded sequential controller. Native physics
is explicitly not a qualified continuum/contact reference. No handles or forces.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
try:
 from .boundary_block import block_mesh,lumped_masses
except ImportError:
 from boundary_block import block_mesh,lumped_masses
ROOT=Path(__file__).resolve().parents[1]
ISAAC='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python'
OUT='runs/gaussian-campaign-v01/isaac-block-gravity-v01'
PROTOCOL='runs/operator-dev/isaac-block-gravity-v01/protocol.json'
SOURCE_NAMES=['experiments/isaac_block_gravity_suite.py','experiments/isaac_block_gravity_worker.py','experiments/boundary_block.py','tests/test_isaac_block_gravity_suite.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def cases():
 rows=[dict(id='gravity_'+b,kind='gravity',boundary=b,young_Pa=300000.,duration_s=.6) for b in ['left','right','bothends','bottom','two_bottom_patches']]
 rows += [dict(id='freefall_300k',kind='freefall',boundary='none',young_Pa=300000.,duration_s=.2)]
 rows += [dict(id='drop_'+str(e//1000)+'k',kind='drop',boundary='none',young_Pa=float(e),duration_s=.6) for e in [300000,600000]]
 return rows

def geometry_and_fixed(case):
 # Canonical static domain begins x=0,y=-.04,z=.30; drop bottom is .05.
 bottom=.05 if case['kind']=='drop' else .30
 X,T,F=block_mesh(size=(.24,.08,.08),cells=(12,4,4),center=(.12,0,bottom+.04))
 local=X-X.min(0);eps=1e-12;b=case['boundary']
 left=abs(local[:,0])<eps;right=abs(local[:,0]-.24)<eps;floor=abs(local[:,2])<eps
 masks={'left':left,'right':right,'bothends':left|right,'bottom':floor,
        'two_bottom_patches':floor&((local[:,0]<=.06+eps)|(local[:,0]>=.18-eps)), 'none':np.zeros(len(X),bool)}
 if b not in masks:raise ValueError('Undeclared fixed boundary')
 return X,T,F,np.flatnonzero(masks[b]).astype(np.int64)

def mass_and_reference(X,T):
 masses=lumped_masses(X,T,625.);det=np.linalg.det(X[T[:,1:]]-X[T[:,:1]])
 return masses,masses/masses.sum(),det

def ballistic_reference(gravity,dt):
 dt=np.asarray(dt,dtype=float);g=np.asarray(gravity,dtype=float)
 if dt.ndim!=1 or not len(dt) or np.any(dt<=0) or g.shape!=(3,) or not np.isfinite(dt).all() or not np.isfinite(g).all():raise ValueError('Finite positive actual clock required')
 v=np.cumsum(dt[:,None]*g,axis=0);previous=np.vstack([np.zeros(3),v[:-1]])
 return v,np.cumsum(previous*dt[:,None]+.5*g*dt[:,None]**2,axis=0),np.cumsum(v*dt[:,None],axis=0)

def check_flags(enabled,raw,N,fixed):
 expected=np.zeros(N,bool);expected[fixed]=True
 e=np.asarray(enabled).reshape(-1);r=np.asarray(raw)
 if e.shape!=(N,) or r.shape!=(1,N,4):raise ValueError('Native target flag shape mismatch')
 np.testing.assert_array_equal(e,expected);np.testing.assert_array_equal(r[0,:,3],(~expected).astype(np.float32))
 return dict(enabled_fixed_count=int(e.sum()),raw_fourth_unique=np.unique(r[0,:,3]).tolist(),raw_semantics='0=enabled target,1=free; not invMass readback')

def native_geometry(rest,tets,X,T):
 np.testing.assert_array_equal(np.asarray(rest),X[None].astype(np.float32))
 canonical=lambda a:np.array(sorted(map(tuple,np.asarray(a).tolist())))
 np.testing.assert_array_equal(canonical(np.asarray(tets)[0]),canonical(T))
 return dict(points_bitwise_float32_match=True,oriented_tet_multiset_matches=True)

def validate_protocol(path,digest):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Frozen protocol changed')
 d=json.loads(p.read_text())
 if d['cases']!=cases() or d['gpu']!=0 or d['physics_hz']!=240 or d['fps']!=30 or d['global_wall_cap_s']!=600 or d['app_wall_cap_s']!=120:raise ValueError('Wrong frozen scope')
 if d['source_sha256']!={s:sha(ROOT/s) for s in SOURCE_NAMES}:raise ValueError('Frozen source closure changed')
 return d

def prepare(path=PROTOCOL):
 p=ROOT/path
 if p.exists():raise FileExistsError('Frozen protocol retained')
 d=dict(schema='isaac-block-gravity-drop-engineering-v0.1',status='FROZEN_BEFORE_NATIVE_CALLS',cases=cases(),gpu=0,CUDA_VISIBLE_DEVICES='0',cpu_threads=2,
 physics_hz=240,fps=30,solver='PGS',deform_iterations=32,gravity_m_s2=[0.,0.,-9.81],size_m=[.24,.08,.08],cells=[12,4,4],rho_kg_m3=625.,poisson=.35,
 linear_damping=0.,elasticity_damping=0.,self_collision=False,contact_offset_m=.001,rest_offset_m=0.,floor_top_z_m=0.,floor_half_extents_m=[1.,1.,.01],
 friction=.5,restitution=0.,drop_initial_bottom_gap_m=.05,global_wall_cap_s=600,app_wall_cap_s=120,isaac_python=ISAAC,output=OUT,
 source_sha256={s:sha(ROOT/s) for s in SOURCE_NAMES},
 ballistics=dict(velocity_max_abs_tolerance='max(2e-5m/s,.001*peak_speed)',maximum_node_nonuniformity_coordinate_rms_m=2e-5,maximum_abs_J_minus1=.001,position_references='Both exactheld andkickdrift reported, no fittedconvention chosen'),
 execution_failure_guards=['finite q/v','Jpositive','native material E/nu match','native rest/tet match','native flag match','initial q/rest andvzero','actual dt float32match','fixedpointdrift<=2um','nativegravitymatch','onlyGPU0compute','floor authoredstaticcollision'],
 qualifications=['Engineering demos, not continuum/time/contact qualification.','Nonzero drop penetration is measured, not repaired; no exact nonpenetration or contact-accuracy claim.','Freefall ballistic validation separate from drop deformation; no extra forces/handles/attachments.','All eight cases retained, no redraw/retry; stop first execution failure.','Native density/nodal masses not exposed; rho authored and total nominal solid mass reported.'])
 write(p,d);return dict(path=path,sha256=sha(p))

def gpu_status():
 return subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout

def run(path,digest):
 d=validate_protocol(path,digest);out=ROOT/d['output']
 if out.exists():raise FileExistsError('Fresh suite output required')
 out.mkdir(parents=True);start=time.monotonic();deadline=start+600
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
 gpu0=gpu_status().splitlines()[0].split(',')[1].strip()
 if any(r.split(',')[1].strip()==gpu0 for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU0 is not idle before suite')
 ledger=dict(schema='isaac-block-gravity-suite-execution-v0.1',controller_pid=os.getpid(),protocol=dict(path=path,sha256=digest),terminal=False,completed=False,cells=[],gpu_before=gpu_status())
 write(out/'execution.json',ledger)
 for c in d['cases']:
  remaining=deadline-time.monotonic()
  if remaining<=0:ledger['error']='Global600sdeadline';break
  validate_protocol(path,digest)
  command=[d['isaac_python'],'-u',str(ROOT/'experiments/isaac_block_gravity_worker.py'),'--protocol',path,'--protocol-sha256',digest,'--case',c['id']]
  row=dict(id=c['id'],command=command,state='running',output=str(Path(d['output'])/c['id']),started_unix_s=time.time())
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMNI_KIT_ACCEPT_EULA':'YES'}
  with (out/(c['id']+'.log')).open('x') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   row['pid']=child.pid;ledger['cells'].append(row);write(out/'execution.json',ledger)
   try:code=child.wait(timeout=min(120,remaining))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;row['timeout']=True
  result=ROOT/row['output']/'summary.json';summary=json.loads(result.read_text()) if result.exists() else {}
  row.update(exit_code=code,finished_unix_s=time.time(),state='completed' if code==0 and summary.get('completed') is True else 'failed')
  if result.exists():row['summary_sha256']=sha(result)
  trajectory=ROOT/row['output']/'trajectory.npz'
  if trajectory.exists():row['trajectory_sha256']=sha(trajectory)
  write(out/'execution.json',ledger)
  if row['state']!='completed':break
 ledger.update(terminal=True,completed=len(ledger['cells'])==8 and all(r['state']=='completed' for r in ledger['cells']),elapsed_s=time.monotonic()-start,gpu_after=gpu_status())
 write(out/'execution.json',ledger);return ledger

if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True);a=sub.add_parser('prepare');a.add_argument('--protocol',default=PROTOCOL);a=sub.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True);args=p.parse_args()
 r=prepare(args.protocol) if args.action=='prepare' else run(args.protocol,args.protocol_sha256);print(json.dumps(r))
 if args.action=='run':raise SystemExit(0 if r['completed'] else 1)
