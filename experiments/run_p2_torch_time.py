"""Seven predeclared available CPU trajectories: GPU full-mass time parity."""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
OP=ROOT/'runs/operator-dev/p2-torch-time-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-torch-time-v01';CPU=ROOT/'runs/gaussian-campaign-v01/p2-consistent-transient-v01'
JOBS=[f'tets6-{p}-hz{h}' for p in ['step','smooth_ramp'] for h in [240,480,960]]+['tets96-step-hz480']
SOURCES=['experiments/run_p2_torch_time.py','experiments/campaign_p2_torch.py','experiments/campaign_p2_transient.py','tests/test_campaign_p2_transient.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/boundary_block.py','experiments/run_p2_consistent_transient.py']
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenprotocol')
 refs={}
 for job in JOBS:
  row=json.loads((CPU/(job+'.json')).read_text())
  if row['completed'] is not True:raise ValueError('CompletedCPUcase required '+job)
  for ext in ['.npz','.json']:refs[str((CPU/(job+ext)).relative_to(ROOT))]=sha(CPU/(job+ext))
 p=dict(schema='P2-fullmass-GPU-time-parity-v0.1',status='FROZEN_BEFORE_GPU_TIME_CALLS',jobs=JOBS,source_sha256={s:sha(ROOT/s) for s in SOURCES},CPU_refs_sha256=refs,CPU_protocol=dict(path='runs/operator-dev/p2-consistent-transient-v01/protocol.json',sha256=sha(ROOT/'runs/operator-dev/p2-consistent-transient-v01/protocol.json')),CPU_threads=2,GPU=0,wall_cap_s=600,selection='FirstsevenavailablecompletedCPUcaseprefix; remainingCPUcasesnotinthisscreen,notselectedbyGPUerror',gates=dict(objective_R_N=1e-10,physical_R_N=1e-9,CPU_allstate_node_error_m=1e-8,CPU_velocity_error_m_s=1e-6,consistent_mass_action_error_kg=1e-13,fixed_displacement_m=0.),nonclaims=['DiscreteGPUCPUtimeparityonly;notspatial/futuregeneralization/contactqualification'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in {**p['source_sha256'],**p['CPU_refs_sha256']}.items():
  if sha(ROOT/f)!=h:raise ValueError('Changedsource/reference '+f)
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),pid=os.getpid(),cases=[])
 try:
  import torch
  from scipy import sparse
  from experiments.campaign_p2_torch import TorchP2
  from experiments.campaign_p2_transient import P2State,step_p2
  from experiments.campaign_newton_stable_material import StableP2Block
  torch.set_num_threads(2);torch.ones(1,device='cuda:0').sum().item()
  info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);allowed=next(s.split(',')[1].strip() for s in info.splitlines() if s.split(',')[0].strip()=='0');apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);own=[s for s in apps.splitlines() if s.split(',')[0].strip()==str(os.getpid())];state['own_compute_records']=own;write(OUT/'summary.json',state)
  if not own or any(s.split(',')[1].strip()!=allowed for s in own):raise ValueError('WrongGPU')
  for job in JOBS:
   begin=time.monotonic();meta=json.loads((CPU/(job+'.json')).read_text())
   with np.load(CPU/(job+'.npz')) as z:ref={k:z[k].copy() for k in z.files}
   m=StableP2Block(tuple(meta['cells']),young=5000,gravity=0,traction=1,order=3);t=TorchP2(m,'cuda:0');M=sparse.csr_matrix((ref['mass_csr_data'],ref['mass_csr_indices'],ref['mass_csr_indptr']),shape=tuple(ref['mass_shape']));v=np.random.default_rng(1321).normal(size=m.rest.shape)
   mass_error=float(abs(t.mass(torch.tensor(v,device=t.device)).cpu().numpy()-M@v).max())
   if mass_error>1e-13:raise ValueError('Independentconsistentmassactionfailed')
   old=None;now=P2State(torch.zeros_like(t.rest),torch.zeros_like(t.rest));U=[now.u.cpu().numpy()];V=[now.v.cpu().numpy()];ledger=[];h=1/meta['hz']
   for k,f in enumerate(ref['force_commands_N']):
    new,diag=step_p2(t,now,old,h,torch.tensor(f,device=t.device));old,now=now,new;U.append(now.u.cpu().numpy());V.append(now.v.cpu().numpy());ledger.append(diag)
   U=np.asarray(U);V=np.asarray(V);error=float(np.linalg.norm(U-ref['displacements_m'],axis=2).max());velocity=float(np.linalg.norm(V-ref['velocities_m_s'],axis=2).max());R=max(x['original_physical_free_R_N'] for x in ledger);file=OUT/(job+'.npz');np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,constraint_mask=m.mask,times_s=ref['times_s'],displacements_m=U,positions_m=m.rest+U,velocities_m_s=V,force_commands_N=ref['force_commands_N'])
   row=dict(id=job,completed=True,passed=error<=1e-8 and velocity<=1e-6 and R<=1e-9,maximum_CPU_allstate_node_error_m=error,maximum_CPU_velocity_error_m_s=velocity,maximum_physical_R_N=R,mass_action_error_kg=mass_error,elapsed_s=time.monotonic()-begin,ledger=ledger,capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)));state['cases'].append(row);write(OUT/(job+'.json'),row);write(OUT/'summary.json',state)
  state.update(completed=True,passed=all(x['passed'] for x in state['cases']))
 except Exception as e:state.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:state['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
  r=dict(pid=os.getpid(),terminal=False,hard_cap_s=600);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_torch_time','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
