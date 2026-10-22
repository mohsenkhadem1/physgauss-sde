"""Prospective CPU/GPU P2 energy, tangent and equilibrium parity screen."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/p2-torch-benchmark-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-torch-benchmark-v01'
SOURCES=['experiments/campaign_p2_torch.py','experiments/run_p2_torch_benchmark.py','tests/test_campaign_p2_torch.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserve protocol')
 p=dict(schema='compatible-P2-float64-GPU-parity-v0.1',status='FROZEN_BEFORE_GPU_SCREEN',source_sha256={s:sha(ROOT/s) for s in SOURCES},cells=[[1,1,1],[4,2,2],[8,4,4]],GPU=0,CPU_threads=2,hard_cap_s=600,physics=dict(E_Pa=5000.,nu=.35,density=625.,traction_N=1.,gravity=0.,boundary='fullleftface',physical_damping=0.),gates=dict(energy_abs_J=1e-12,gradient_abs_N=1e-9,HVP_abs_N=1e-8,static_original_R_N=1e-10,same_mesh_CPU_node_error_m=1e-8,exact_fixed_displacement=0.),algorithm=dict(precision='float64',mass='exactconsistentT10notrowlumped',matrix_free_full_tangent=True,maximum_Newton=25,maximum_CG=1000,no_constitutive_changes=True),nonclaims=['Same-meshparityonly;continuumqualificationrequiresseparateP2finecapture','No data/training admission from benchmark alone'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed protocol')
 p=json.loads((OP/'protocol.json').read_text())
 if any(sha(ROOT/s)!=h for s,h in p['source_sha256'].items()) or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedsource/device')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),pid=os.getpid(),cases=[])
 try:
  import torch
  from experiments.campaign_p2_torch import TorchP2
  from experiments.campaign_newton_stable_material import StableP2Block
  torch.set_num_threads(2);device='cuda:0';probe=torch.ones(1,dtype=torch.float64,device=device);probe.sum().item()
  state['runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda,device_name=torch.cuda.get_device_name(0))
  uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);allowed=next(s.split(',')[1].strip() for s in uuids.splitlines() if s.split(',')[0].strip()=='0')
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True);own=[s for s in apps.splitlines() if s.split(',')[0].strip()==str(os.getpid())];state['own_compute_records']=own;write(OUT/'summary.json',state)
  if not own or any(s.split(',')[1].strip()!=allowed for s in own):raise ValueError('WrongGPU')
  for cells in p['cells']:
   begin=time.monotonic();m=StableP2Block(tuple(cells),young=5000,gravity=0,traction=1);t=TorchP2(m,device);rng=np.random.default_rng(1193);u=rng.normal(size=m.rest.shape)*1e-6;direction=rng.normal(size=m.rest.shape);E,g,H,J=m.elastic(m.rest+u,True)
   cache=t.evaluate(torch.as_tensor(u,device=device));hv=t.tangent(cache,torch.as_tensor(direction,device=device));ee=abs(float(cache['energy'])-E);ge=float(abs(cache['gradient'].cpu().numpy()-g).max());he=float(abs(hv.cpu().numpy().ravel()-H@direction.ravel()).max())
   if ee>1e-12 or ge>1e-9 or he>1e-8:raise ValueError('IndependentCPUconstitutive/tangentparityfailed')
   cpu=m.solve();gpu=t.solve();q=gpu['positions'].cpu().numpy();res=float(torch.linalg.vector_norm(gpu['residual']));error=float(np.linalg.norm(q-cpu['positions'],axis=1).max());fixed=float(abs(gpu['displacement'].cpu().numpy()[m.mask]).max());label='tets'+str(len(m.tets));file=OUT/(label+'.npz');np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,reference_CPU_positions_m=cpu['positions'],constraint_mask=m.mask)
   row=dict(cells=cells,tetrahedra=len(m.tets),nodes=len(m.rest),energy_error_J=ee,gradient_error_N=ge,HVP_error_N=he,original_free_R_N=res,maximum_CPU_node_error_m=error,fixed_drift_m=fixed,passed=res<=1e-10 and error<=1e-8 and fixed==0.,elapsed_s=time.monotonic()-begin,history=gpu['history'],capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)))
   state['cases'].append(row);write(OUT/'summary.json',state)
  state.update(completed=True,passed=all(r['passed'] for r in state['cases']),maximum_allocated_GPU_bytes=torch.cuda.max_memory_allocated())
 except Exception as e:state.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:state['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':r=run(a.sha);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
  declaration=dict(pid=os.getpid(),protocol_sha256=a.sha,terminal=False,hard_cap_s=600);write(OP/'execution.json',declaration)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_torch_benchmark','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
   declaration['worker_pid']=child.pid;write(OP/'execution.json',declaration);start=time.monotonic()
   try:code=child.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;declaration['timeout']=True
  declaration.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',declaration);print(json.dumps(declaration));raise SystemExit(code)
