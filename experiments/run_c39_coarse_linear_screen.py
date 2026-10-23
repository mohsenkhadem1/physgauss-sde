"""Prospective CPU rest-tangent action screen; no observations or nonlinear fields."""
import argparse,ast,hashlib,json,os,resource,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c39-coarse-linear-screen-v01';SOURCE='experiments/run_c39_coarse_linear_screen.py';TEST='tests/test_run_c39_coarse_linear_screen.py'
MESHES=((4,2,2),(8,4,4));SUPPORTS=('both_ends','separated_bottom_patches');CANDIDATES=('middle_down_2N','left_down_2N','middle_side_2N');ACTIONS=('gravity',)+CANDIDATES+('canonical_up_1N',);POINT_IDS=np.array([11,14,17,29,32,35,47,50,53]);RSS_CAP=2*1024**3

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def closure():
 seen=set();pending=[SOURCE,TEST]
 while pending:
  name=pending.pop()
  if name in seen:continue
  seen.add(name)
  for n in ast.walk(ast.parse((ROOT/name).read_text())):
   if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments.'):
    f='experiments/'+n.module.split('.')[1]+'.py'
    if (ROOT/f).exists():pending.append(f)
 return sorted(seen)
def prepare():
 p=dict(schema='C39-public-action-coarse-LINEAR-screen-v01',status='FROZEN_BEFORE_MODEL_RESPONSE_CALLS',source_sha256={f:sha(ROOT/f) for f in closure()},meshes=[list(n) for n in MESHES],supports=list(SUPPORTS),young_Pa=75000.,poisson=.35,density_kg_m3=625.,gravity_m_s2=9.81,actions=list(ACTIONS),candidate_actions=list(CANDIDATES),sensor_POINT_IDS=POINT_IDS.tolist(),sensor_sigma_m=.0002,method='Independent NumPy StableP2Block.elastic REST sparse tangent; exact free-DOF elimination, no stiffness repair. Original TorchP2.tangent residual checked separately for EVERY response.',gates=dict(original_free_residual_N=1e-10,exact_constraint_error_m=0.),fisher='Per added ideal nine-top-point acquisition: du/dlogE=-u, Fisher=||u9||^2/sigma^2; total gravity+action, not force-only response. Prefix gravity and canonical up1 reported separately.',promising='Different unique highest-Fisher candidate by support, SAME winners on BOTH meshes, winner/runnerup>=1.05 in all four model screens. Engineering screening only; no active-policy claim.',CPU_affinity_cores=[34,35],threads=2,CUDA_VISIBLE_DEVICES='',wall_cap_s=180,RSS_cap_bytes=RSS_CAP,no_actual_C38_prefix_labels_or_observations=True,no_nonlinear_or_dataset_admission=True)
 path=OP/'protocol.json'
 if path.exists():raise FileExistsError(path)
 write(path,p);return sha(path)
def validate(digest):
 f=OP/'protocol.json';p=json.loads(f.read_text())
 if sha(f)!=digest or any(sha(ROOT/k)!=h for k,h in p['source_sha256'].items()):raise ValueError('Changed frozen sources/recipe')
 return p
def constrained_solve(K,force,mask):
 from scipy.sparse.linalg import splu
 free=np.flatnonzero(~mask.ravel());u=np.zeros_like(force);u.ravel()[free]=splu(K[free][:,free].tocsc()).solve(force.ravel()[free]);r=(K@u.ravel()-force.ravel())[free]
 if not np.isfinite(u).all() or np.linalg.norm(r)>1e-10 or np.any(u[mask]!=0):raise ValueError('Original sparse K equilibrium or exact mask failed')
 return u,float(np.linalg.norm(r))
def fisher(u9,sigma=.0002):return float(np.sum(np.asarray(u9)**2)/sigma**2)
def work(digest):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={34,35} or any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('Exact CPU-only resources required')
 import torch,scipy
 from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
 from experiments.campaign_newton_stable_material import StableP2Block
 from experiments.campaign_p2_torch import TorchP2
 from experiments.campaign_c39_known_actions import compile_known_action
 from experiments.campaign_c38_table_validation import POINTS
 torch.set_num_threads(2);torch.set_num_interop_threads(2);start=time.monotonic();result=dict(completed=False,protocol_sha256=digest,models=[],no_observations_opened=True)
 def stopped(*_):raise TimeoutError('Frozen CPU wall/RSS interruption')
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGALRM,stopped);signal.alarm(180)
 try:
  for cells in MESHES:
   for support in SUPPORTS:
    m=SupportedBoundaryGradedGeometry(cells,75000.,support,0.,rule='keast24');dofs=(m.elements[:,:,None]*3+np.arange(3)).reshape(-1,30);m.rows=np.repeat(dofs,30,1).ravel().astype(np.int32);m.cols=np.tile(dofs,(1,30)).ravel().astype(np.int32)
    energy,g,K,J=StableP2Block.elastic(m,m.rest,True)
    if np.linalg.norm(g)>1e-12:raise ValueError('Nonzero rest stress')
    op=TorchP2(m,'cpu');cache=op.evaluate(torch.zeros_like(op.rest));row=dict(cells=list(cells),support=support,base_tets=len(m.tets),nodes=len(m.rest),responses=[]);result['models'].append(row)
    for action in ACTIONS:
     loads=compile_known_action(m.rest,m.tets,m.faces,m.elements,action);f=loads['total_N'];u,R=constrained_solve(K,f,m.mask);applied=TorchP2.tangent(op,cache,torch.tensor(u,dtype=torch.float64)).detach().numpy();Rt=float(np.linalg.norm((applied-f)[~m.mask]))
     if Rt>1e-10:raise ValueError('Independent ORIGINAL Torch HVP response residual failed')
     u63=m.query(u,POINTS);u9=u63[POINT_IDS];row['responses'].append(dict(action=action,original_sparse_free_R_N=R,original_Torch_HVP_free_R_N=Rt,exact_constraint_error_m=float(np.max(abs(u[m.mask]))),logE_Fisher_9top=fisher(u9),sensor_displacement9_m=u9.tolist(),canonical_query_displacement63_m=u63.tolist(),load_metadata=loads['metadata']))
     result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
     if result['peak_RSS_bytes']>RSS_CAP:raise MemoryError('Frozen 2GiB RSS exceeded')
     write(OP/'result.json',result)
    ranked=sorted([r for r in row['responses'] if r['action'] in CANDIDATES],key=lambda r:(-r['logE_Fisher_9top'],r['action']));row['candidate_ranking']=[r['action'] for r in ranked];row['winner_over_runnerup_Fisher']=ranked[0]['logE_Fisher_9top']/ranked[1]['logE_Fisher_9top'];del cache,op,K,m
  winners={s:[r['candidate_ranking'][0] for r in result['models'] if r['support']==s] for s in SUPPORTS};consistent=all(len(set(v))==1 for v in winners.values());different=consistent and winners[SUPPORTS[0]][0]!=winners[SUPPORTS[1]][0];margin=all(r['winner_over_runnerup_Fisher']>=1.05 for r in result['models']);result.update(completed=True,winners_by_support=winners,mesh_consistent_winners=consistent,support_dependent_winner=different,promising_for_fine_linear_screen=bool(different and margin),versions=dict(numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__))
 except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
 finally:signal.alarm(0);result['seconds']=time.monotonic()-start;write(OP/'result.json',result)
 return result

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('No retry')
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'};start=time.monotonic();log=(OP/'worker.log').open('x');child=None;state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_sha256=digest)
 def stopped(*_):raise InterruptedError('Owned CPU controller interruption')
 old={s:signal.signal(s,stopped) for s in [signal.SIGTERM,signal.SIGINT]}
 from experiments.campaign_owned_gpu_budget_v02 import terminate_owned
 try:
  blocked=signal.pthread_sigmask(signal.SIG_BLOCK,set(old))
  try:child=subprocess.Popen(['/usr/bin/taskset','-c','34,35',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_c39_coarse_linear_screen','work','--sha',digest],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['worker_pid']=child.pid
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
  write(OP/'execution.json',state)
  while child.poll() is None:
   if time.monotonic()-start>180:raise TimeoutError('Owned 180s cap')
   status=Path(f'/proc/{child.pid}/status')
   if status.exists():
    rss=next((int(x.split()[1])*1024 for x in status.read_text().splitlines() if x.startswith('VmRSS:')),0)
    if rss>RSS_CAP:raise MemoryError('Owned 2GiB RSS cap')
   time.sleep(.1)
  state['exit_code']=child.wait();state['completed']=state['exit_code']==0 and json.loads((OP/'result.json').read_text())['completed']
 except BaseException as e:state['error']=repr(e)
 finally:
  if child is not None and child.poll() is None:state['cleanup_exit_code']=terminate_owned(child)
  if child is not None:child.wait()
  log.close()
  for s,h in old.items():signal.signal(s,h)
  state.update(terminal=True,seconds=time.monotonic()-start);write(OP/'execution.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','work','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else work(a.sha) if a.action=='work' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','seconds','error','promising_for_fine_linear_screen']}));raise SystemExit(0 if isinstance(r,str) or r['completed'] else 1)
