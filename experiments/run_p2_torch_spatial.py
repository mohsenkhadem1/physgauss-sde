"""Two fine GPU P2 solves from rest against frozen independent CPU references."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/p2-torch-spatial-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-torch-spatial-v01';CPU=ROOT/'runs/gaussian-campaign-v01/newton-stable-p2-spatial-v01'
from experiments.run_p2_torch_benchmark import write,sha
SOURCES=['experiments/run_p2_torch_spatial.py','experiments/campaign_p2_torch.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/boundary_block.py']
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen stage')
 s=json.loads((CPU/'summary.json').read_text())
 refs={}
 for order in [3,4]:
  for f in [CPU/f'order{order}.json',CPU/f'stable-P2-tets12000-order{order}.npz']:refs[str(f.relative_to(ROOT))]=sha(f)
 p=dict(schema='GPU-P2-fine-spatial-v0.1',status='FROZEN_BEFORE_GPU_CALLS',CPU_threads=2,GPU=0,wall_cap_s=600,cells=[20,10,10],orders=[3,4],source_sha256={f:sha(ROOT/f) for f in SOURCES},CPU_refs_sha256=refs,CPU_summary=dict(path=str((CPU/'summary.json').relative_to(ROOT)),sha256=sha(CPU/'summary.json')),gates=dict(original_free_R_N=1e-10,CPU_node_error_m=1e-8,CPU_fixedquery_error_m=1e-8,constraint_m=0.,force_balance_N=1e-8,moment_balance_Nm=1e-8),physics=dict(E_Pa=5000.,nu=.35,rho=625.,traction_N=1.,gravity=0.,BC='fullleftface'),nonclaims=['GPUCPUparitydoesnotturnconditionalp>=1indicatorintorigorouscontinuumbound','Temporal/mass/heldload/contacttestsseparate'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Protocol/devicechanged')
 p=json.loads((OP/'protocol.json').read_text())
 for name,h in {**p['source_sha256'],**p['CPU_refs_sha256']}.items():
  if sha(ROOT/name)!=h:raise ValueError('Changedinput '+name)
 OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();state=dict(completed=False,passed=False,protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),cases=[],pid=os.getpid())
 try:
  import torch
  from experiments.campaign_p2_torch import TorchP2
  from experiments.campaign_newton_stable_material import StableP2Block
  torch.set_num_threads(2);device='cuda:0';torch.ones(1,device=device).sum().item()
  uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);allowed=next(r.split(',')[1].strip() for r in uuids.splitlines() if r.split(',')[0].strip()=='0')
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True);own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())];state['own_compute_records']=own;write(OUT/'summary.json',state)
  if not own or any(r.split(',')[1].strip()!=allowed for r in own):raise ValueError('WrongGPU')
  for order in p['orders']:
   start=time.monotonic();m=StableP2Block(tuple(p['cells']),young=5000,gravity=0,traction=1,order=order);t=TorchP2(m,device)
   with np.load(CPU/f'stable-P2-tets12000-order{order}.npz') as z:reference={k:z[k].copy() for k in z.files}
   answer=t.solve();q=answer['positions'].cpu().numpy();qp=m.query(q,reference['material_probes_m']);node=float(np.linalg.norm(q-reference['positions_m'],axis=1).max());query=float(np.linalg.norm(qp-reference['probe_positions_m'],axis=1).max());R=float(torch.linalg.vector_norm(answer['residual']));fixed=float(abs(answer['displacement'].cpu().numpy()[m.mask]).max())
   grad=answer['cache']['gradient'].cpu().numpy();reaction=np.where(m.mask,grad-m.load,0.);balance=float(np.linalg.norm((reaction+m.load).sum(0)));moment=float(np.linalg.norm(np.cross(q,reaction+m.load).sum(0)))
   f=OUT/f'order{order}.npz';np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,probe_positions_m=qp,material_probes_m=reference['material_probes_m'],constraint_mask=m.mask)
   row=dict(order=order,completed=True,passed=R<=1e-10 and node<=1e-8 and query<=1e-8 and fixed==0 and balance<=1e-8 and moment<=1e-8,original_free_R_N=R,maximum_CPU_node_error_m=node,maximum_CPU_fixedquery_error_m=query,fixed_drift_m=fixed,force_balance_N=balance,moment_balance_Nm=moment,minimum_J=float(answer['cache']['minimum_J']),elapsed_s=time.monotonic()-start,history=answer['history'],capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)));state['cases'].append(row);write(OUT/f'order{order}.json',row);write(OUT/'summary.json',state)
  state.update(completed=True,passed=all(r['passed'] for r in state['cases']),maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
 except Exception as e:state.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:state['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
  declaration=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=600);write(OP/'execution.json',declaration)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_torch_spatial','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});declaration['worker_pid']=child.pid;write(OP/'execution.json',declaration);start=time.monotonic()
   try:code=child.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;declaration['timeout']=True
  declaration.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',declaration);print(json.dumps(declaration));raise SystemExit(code)
