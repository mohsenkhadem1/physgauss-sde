"""Warp-only resource continuation; import completed original-backend control."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,traceback
import numpy as np
from experiments.continue_p2_contact_dop853_v03 import ROOT,GPU,sha,write,manual_dop853,SOURCES as OLD_SOURCES
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
from experiments.benchmark_p2_warp_rhs import ENV_SITE
OP=ROOT/'runs/operator-dev/p2-warp-trajectory-v02';OUT=ROOT/'runs/gaussian-campaign-v01/p2-warp-trajectory-v02'
PRIOR=ROOT/'runs/gaussian-campaign-v01/p2-warp-trajectory-v01';PRIOR_OP=ROOT/'runs/operator-dev/p2-warp-trajectory-v01'
SOURCES=OLD_SOURCES+['experiments/run_p2_warp_trajectory_control.py','tests/test_run_p2_warp_trajectory_control.py']+['experiments/campaign_p2_warp_rhs.py','experiments/benchmark_p2_warp_rhs.py','experiments/continue_p2_warp_trajectory_v02.py','tests/test_continue_p2_warp_trajectory_v02.py']
CONFIG=dict(rtol=1e-9,displacement_atol_m=1e-12,velocity_atol_m_s=1e-10,impulse_atol_Ns=1e-12)
def parity(q0,v0,q1,v1):
 q=float(np.linalg.norm(q0-q1,axis=-1).max());v=float(np.linalg.norm(v0-v1,axis=-1).max());return dict(maximum_allnode_q_difference_m=q,maximum_allnode_v_difference_m_s=v,passed=q<=1e-8 and v<=1e-5)
def prepare():
 p=OP/'protocol.json'
 if p.exists():raise FileExistsError('Frozen phase')
 bench=ROOT/'runs/operator-dev/p2-warp-rhs-v01/gpu-result.json';be=ROOT/'runs/operator-dev/p2-warp-rhs-v01/gpu-execution.json';bp=ROOT/'runs/operator-dev/p2-warp-rhs-v01/gpu-protocol.json'
 if not json.loads(bench.read_text()).get('passed') or not json.loads(be.read_text()).get('terminal'):raise ValueError('Completed GPU same-state parity prerequisite')
 prior=json.loads(bp.read_text());tc=PRIOR/'torch.json';te=PRIOR_OP/'execution.json';tp=PRIOR_OP/'protocol.json';case=json.loads(tc.read_text());capture=ROOT/case['capture']['path']
 if not case.get('completed') or not case.get('passed') or not json.loads(te.read_text()).get('terminal') or sha(capture)!=case['capture']['sha256']:raise ValueError('ExactcompletedTorchrequired')
 for f,h in json.loads(tp.read_text())['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changedpriorphysicalsource')
 write(p,dict(schema='Warp-only6tet-resource-continuation-v0.2',status='FROZEN_BEFORE_PHYSICAL_CALLS',gpu_uuid=GPU,wall_cap_s=300,CPU_threads=2,cells=[1,1,1],backends=['warp'],config=CONFIG,maximum_step_s=1/30000,interval_s=[T0,END],outputs_s=(T0+np.arange(9)/120).tolist(),physics=dict(E_Pa=300000.,nu=.35,rho=625.,gravity=GRAVITY.tolist(),gap_m=.05,kappa_N_m3=1e9,surface_Q=8,friction=0.,damping=0.,floor_z_m=0.),source_sha256={f:sha(ROOT/f) for f in SOURCES},installed_warp_source_sha256=prior['installed_warp_source_sha256'],prerequisite_sha256={str(f.relative_to(ROOT)):sha(f) for f in [bench,be,bp,tc,te,tp,capture]},resource_continuation='Old300scappreserved; reusecompletedTorchonly; freshWarpfromexactinitializer. Priorprefixlacksf/nexth_abs so noexactresumeclaim; recomputed21116RHScommittedprefixcostexplicit.',gates=dict(allnode_q_m=1e-8,allnode_v_m_s=1e-5,masssolve_R_N=1e-8,minimum_J=.1,energy_J='max(1e-9,1e-6*mass*g*.05)',momentum_Ns=1e-8),nonclaims=['Paired same-discretization backend implementation only','No temporal/spatial/Q or hardcontact qualification','No existing trajectory/backend changed']))
 return sha(p)
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Changed protocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('installed_warp_source_sha256',ENV_SITE),('prerequisite_sha256',ROOT)]:
  if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changed source/input')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();result=dict(completed=False,passed=False,pid=os.getpid(),protocol_sha256=digest,cases=[],nonclaims=p['nonclaims'])
 try:
  import torch
  torch.set_num_threads(2);torch.ones(1,device='cuda:0').sum().item();own=[r for r in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True).splitlines() if r.split(',')[0].strip()==str(os.getpid())]
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('Wrong physicalGPU')
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0':raise ValueError('Wrong Warp')
  wp.config.kernel_cache_dir=str(OP/'warp-cache')
  from experiments.campaign_p2_warp_rhs import WarpP2Force
  from experiments.campaign_p2_torch_contact import PlaneContactP2
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.run_p2_contact_validation import build
  m=build([1,1,1],300000.,9.81);obj=PlaneContactP2(m,'cuda:0',kappa_A=1e9,surface_order=8);arrays={k:getattr(obj,k).cpu().numpy().copy() for k in WarpP2Force.REQUIRED};warp=WarpP2Force(arrays,young=m.young,poisson=m.poisson,kappa_A=1e9,device='cuda:0');M=torch.tensor(scalar_mass(m).toarray(),dtype=torch.float64,device='cuda:0');L=torch.linalg.cholesky(M);force=obj.force;n=len(m.rest);size=3*n;tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device='cuda:0');u0,v0,_,_=initial_vectors(m.rest,1/1920);y0=np.r_[u0.ravel(),v0.ravel(),np.zeros(3)];P0=(M@tensor(v0)).sum(0).cpu().numpy();torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device='cuda:0')*tensor(GRAVITY),force,atol=1e-13,rtol=1e-13);energytol=max(1e-9,1e-6*float(M.sum())*9.81*.05);fields={};versions=dict(torch=torch.__version__,warp=wp.__version__,numpy=np.__version__);result.update(versions=versions,own_GPU_records=own)
  def evaluate(name,u):return obj.evaluate(tensor(u)) if name=='torch' else warp.evaluate(u)
  # Warm the identical force routines; exclude cache/JIT from per-case integration.
  for backend in p['backends']:evaluate(backend,u0)
  prior_case=json.loads((PRIOR/'torch.json').read_text());saved=np.load(ROOT/prior_case['capture']['path'],allow_pickle=False);np.testing.assert_array_equal(saved['times_s'],p['outputs_s']);np.testing.assert_array_equal(saved['rest_m'],m.rest);np.testing.assert_array_equal(saved['tets'],m.tets);fields['torch']=(saved['positions_m'].copy(),saved['velocities_m_s'].copy());result['cases'].append(dict(**prior_case,imported=True));result['recomputed_prefix']=dict(prior_committed_rhs=21116,prior_committed_steps=1024,reason='Missingnext_h_abs/fprecludesexactaccepted-partitionresume; freshWarp-onlyfallbackauthorized')
  for backend in p['backends']:
   begin=time.monotonic();Et=[];Pt=[];Jmin=1.;massR=0.;calls=0;checked=0;OE=[];OPmom=[];initial=evaluate(backend,u0);E0=float(initial['energy']+.5*(tensor(v0)*(M@tensor(v0))).sum()-(force*tensor(u0)).sum())
   def rhs(t,y):
    nonlocal massR,calls,Jmin
    cache=evaluate(backend,y[:size].reshape(n,3));net=force-tensor(cache['gradient']);acc=torch.cholesky_solve(net,L);r=float(torch.linalg.vector_norm(M@acc-net));massR=max(massR,r);calls+=1;Jmin=min(Jmin,float(cache['minimum_J']))
    if not np.isfinite(r) or r>1e-8:raise ValueError('Masssolve guard')
    return np.r_[y[size:2*size],acc.cpu().numpy().ravel(),net.sum(0).cpu().numpy()]
   def balance(y):
    u=tensor(y[:size].reshape(n,3));v=tensor(y[size:2*size].reshape(n,3));cache=evaluate(backend,y[:size].reshape(n,3));E=float(cache['energy']+.5*(v*(M@v)).sum()-(force*u).sum());P=float(np.linalg.norm((M@v).sum(0).cpu().numpy()-P0-y[2*size:]));return E,P
   def accepted(info):
    nonlocal checked
    E,P=balance(info['accepted_states'][-1]);Et.append(E);Pt.append(P)
    while checked<len(info['output_states']):
     E,P=balance(info['output_states'][checked]);OE.append(E);OPmom.append(P);checked+=1
    count=len(info['accepted_times'])-1
    if count%512==0 or info['solver'].status=='finished':
     tmp=OUT/(backend+'-prefix.tmp.npz');np.savez(tmp,accepted_times_s=info['accepted_times'],accepted_states=np.asarray(info['accepted_states']).T,accepted_energy_J=Et,accepted_momentum_errors_Ns=Pt,common_times_s=info['output_times'],common_states=np.asarray(info['output_states']).T,protocol_sha256=digest,current_y=info['solver'].y,current_f=info['solver'].f,next_h_abs_s=info['solver'].h_abs);tmp.replace(OUT/(backend+'-prefix.npz'));write(OUT/'progress.json',dict(backend=backend,accepted_steps=count,time_s=info['accepted_times'][-1],rhs_calls=calls))
   c=p['config'];atol=np.r_[np.full(size,c['displacement_atol_m']),np.full(size,c['velocity_atol_m_s']),np.full(3,c['impulse_atol_Ns'])];answer=manual_dop853(rhs,y0.copy(),[T0,END],p['outputs_s'],c,atol,accepted);y=answer['output_states'];q=y[:size].T.reshape(-1,n,3)+m.rest;v=y[size:2*size].T.reshape(-1,n,3);fields[backend]=(q,v);file=OUT/(backend+'.npz');np.savez(file,times_s=answer['output_times'],positions_m=q,velocities_m_s=v,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,accepted_times_s=answer['accepted_times'],accepted_states=answer['accepted_states'],accepted_energy_J=Et,accepted_momentum_errors_Ns=Pt,output_energy_J=OE,output_momentum_errors_Ns=OPmom)
   energyerror=max(float(abs(np.asarray(Et)-E0).max()),float(abs(np.asarray(OE)-E0).max()));Perror=max(Pt+OPmom);row=dict(backend=backend,completed=True,passed=energyerror<=energytol and Perror<=1e-8 and Jmin>.1 and massR<=1e-8,elapsed_s=time.monotonic()-begin,maximum_energy_error_J=energyerror,energy_tolerance_J=energytol,maximum_momentum_error_Ns=Perror,minimum_J=Jmin,maximum_masssolve_R_N=massR,rhs_calls=calls,accepted_steps=answer['accepted_steps'],capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)));result['cases'].append(row);write(OUT/(backend+'.json'),row);write(OUT/'summary.json',result)
  pair=parity(*fields['torch'],*fields['warp']);result.update(completed=True,passed=all(r['passed'] for r in result['cases']) and pair['passed'],backend_pair=pair)
 except Exception as exc:result.update(error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
 finally:result['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',result)
 return result

def launch(digest):
 if sha(OP/'protocol.json')!=digest or (OP/'execution.json').exists():raise ValueError('Changed/reused protocol')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(s.split(',')[1].strip()==GPU for s in apps.splitlines() if ',' in s):raise ValueError('GPU1 busy')
 e=dict(terminal=False,completed=False,controller_pid=os.getpid(),protocol_sha256=digest,wall_cap_s=300);write(OP/'execution.json',e);begin=time.monotonic()
 with (OP/'worker.log').open('x') as f:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.continue_p2_warp_trajectory_v02','run','--sha',digest],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});e['worker_pid']=child.pid;write(OP/'execution.json',e)
  try:code=child.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;e['timeout']=True
 r=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=code==0 and r.get('completed') is True,passed=r.get('passed',False),exit_code=code,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e);return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
