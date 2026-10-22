"""Prospective GPU1 same-state root64 E/g parity and inclusive microbenchmark.

No dynamics, bulk/mass change, DuffyQ32, productionHessian or contact admission.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.benchmark_p2_warp_rhs import ENV_SITE
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-warp-root-contact-gpu-v02';OUT=ROOT/'runs/gaussian-campaign-v01/p2-warp-root-contact-gpu-v02';CPU=ROOT/'runs/operator-dev/p2-warp-root-contact-v01'
GPU='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb';CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz';EXPECTED='ac850acac4f8906c0b046b813952d07e198736569418f6edc04823b8cbe7f408'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 result=json.loads((CPU/'CPU-fixedstate-result.json').read_text())
 if not result['completed'] or not result['passed'] or sha(CAPTURE)!=EXPECTED:raise ValueError('CompletedCPUcontrols/exactfieldsrequired')
 sources=['experiments/campaign_p2_warp_root_contact.py','tests/test_campaign_p2_warp_root_contact.py','experiments/benchmark_p2_warp_root_contact_v02.py','tests/test_benchmark_p2_warp_root_contact.py','experiments/campaign_owned_gpu_budget_v02.py','tests/test_campaign_owned_gpu_budget_v02.py','experiments/benchmark_p2_warp_rhs.py']
 p=dict(schema='GPU1same-state-root64-forceenergy-parity-v1',status='FROZEN_PENDING_ROOT_SOURCE_REVIEW_BEFORE_GPU',source_sha256={f:sha(ROOT/f) for f in sources},CPU_evidence_sha256={str((CPU/f).relative_to(ROOT)):sha(CPU/f) for f in ['CPU-protocol.json','CPU-fixedstate-result.json']},installed_warp_source_sha256=json.loads((ROOT/'runs/operator-dev/p2-warp-rhs-v01/gpu-protocol.json').read_text())['installed_warp_source_sha256'],capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),selection=dict(common_output_indices=[2,3,6],accepted_nearest_time_s=.102),physics=dict(kappa_N_m3=1e9,floor_z_m=0.,outer_order=64,law='SAMEfinitecompliantreferenceareaT6plane;root-splitinnerquartic+endpointmappedouter64'),gates=dict(energy_J=1e-12,gradient_L2_N=1e-10),timing=dict(rounds=5,states_per_round=4,CPU_GPU_order_alternates=True,first_JIT_separate_excluded=True,included='NumPyinputtransfer,fullsamecontactE/gwork,E+gradienthostoutput; no bulk/fullM/integrator',guards_between_rounds_excluded_and_recorded=True,repeat_GPU_atomic_envelope=True),GPU_uuid=GPU,CPU_threads=2,CPU_affinity_cores=[30,31],hard_cap_s=180,resource_limits=LIMITS,taskset=dict(path='/usr/bin/taskset',sha256=sha(Path('/usr/bin/taskset'))),limits=['Force/EinterfaceONLY; noHessianproductionclaim','NoGPUtrajectory,temporal/spatial/surfaceadmission','CPU-vsGPUmicrobenchmarkofsameoperator,notfullRHS/solverperformance','OldDuffyQ8/Q16negativeand32to64Hnegativepreserved','FiniteouterE/gderivativeagreementcontrollednumericallyCPU;acceptedtrajectoryenergyguardstillrequiredfuture'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('CPU_evidence_sha256',ROOT),('installed_warp_source_sha256',ENV_SITE)]:
  for f,h in p[group].items():
   if sha(base/f)!=h:raise ValueError('Changedpin:'+f)
 if sha(CAPTURE)!=p['capture']['sha256']:raise ValueError('Changedcapture')
 return p

def parity(a,b,gates):
 row=dict(energy_difference_J=abs(a['contact_energy']-b['contact_energy']),gradient_L2_difference_N=float(np.linalg.norm(a['gradient']-b['gradient'])));row['passed']=row['energy_difference_J']<=gates['energy_J'] and row['gradient_L2_difference_N']<=gates['gradient_L2_N'];return row

def run(digest):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('GPU1UUIDrequired')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,pid=os.getpid(),protocol_sha256=digest,parity=[],timing=[])
 try:
  import torch
  sys.path.append(str(ENV_SITE));import warp as wp
  from experiments.campaign_p2_warp_root_contact import WarpRootContact
  if wp.__version__!='1.18.0':raise ValueError('PinnedWarp1.18required')
  torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,2,2);torch.cuda.set_per_process_memory_fraction(p['resource_limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.ones(1,device='cuda:0').sum().item();wp.config.kernel_cache_dir=str(OP/'warp-cache')
  def guard(tag):
   row=dict(tag=tag,unix_time_s=time.time(),**collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch));validate_budget(row,limits=p['resource_limits']);write(OP/'resource-latest.json',row)
   with (OP/'resource-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  guard('startup_ownGPU');s['versions']=dict(torch=torch.__version__,warp=wp.__version__,numpy=np.__version__)
  with np.load(CAPTURE,allow_pickle=False) as z:
   rest=z['rest_m'];faces=z['surface_elements'];q=z['positions_m'];times=z['times_s'];states=[('common_'+str(i),float(times[i]),np.ascontiguousarray(q[i]-rest)) for i in p['selection']['common_output_indices']];t=z['accepted_times_s'];i=int(np.argmin(abs(t-.102)));states.append(('accepted_nearest_.102',float(t[i]),np.ascontiguousarray(z['accepted_states'][:rest.size,i].reshape(rest.shape))))
  areas=.5*np.linalg.norm(np.cross(rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]]),axis=1);kw=dict(rest=rest,surface_elements=faces,reference_areas=areas,kappa_A=1e9,outer_order=64);cpu=WarpRootContact(**kw,device='cpu');gpu=WarpRootContact(**kw,device='cuda:0')
  clock=time.monotonic();cpu.evaluate(states[0][2]);s['first_CPU_call_including_JIT_s']=time.monotonic()-clock;clock=time.monotonic();gpu.evaluate(states[0][2]);wp.synchronize_device('cuda:0');s['first_GPU_call_including_JIT_s']=time.monotonic()-clock
  references=[]
  for name,t,u in states:
   a=cpu.evaluate(u);b=gpu.evaluate(u);row=dict(name=name,time_s=t,**parity(a,b,p['gates']));s['parity'].append(row);references.append(b)
  if not all(r['passed'] for r in s['parity']):raise ValueError('GPUCPUforce/Eparityfailed')
  guard('after_parity');maxrepeatE=0.;maxrepeatg=0.
  for roundid in range(p['timing']['rounds']):
   order=['cpu','gpu'] if roundid%2==0 else ['gpu','cpu'];times={}
   for label in order:
    obj=cpu if label=='cpu' else gpu;clock=time.monotonic();results=[obj.evaluate(u) for _,_,u in states]
    if label=='gpu':wp.synchronize_device('cuda:0')
    times[label]=time.monotonic()-clock
    if label=='gpu':
     for a,b in zip(results,references):r=parity(a,b,p['gates']);maxrepeatE=max(maxrepeatE,r['energy_difference_J']);maxrepeatg=max(maxrepeatg,r['gradient_L2_difference_N'])
   s['timing'].append(dict(round=roundid,order=order,CPU_total_s=times['cpu'],GPU_total_s=times['gpu']));guard('round_'+str(roundid));write(OUT/'summary.json',s)
  CPUmean=float(np.mean([r['CPU_total_s'] for r in s['timing']]))/4;GPUmean=float(np.mean([r['GPU_total_s'] for r in s['timing']]))/4;s.update(completed=True,passed=maxrepeatE<=p['gates']['energy_J'] and maxrepeatg<=p['gates']['gradient_L2_N'],maximum_repeat_GPU_energy_difference_J=maxrepeatE,maximum_repeat_GPU_gradient_L2_difference_N=maxrepeatg,CPU_mean_evaluate_s=CPUmean,GPU_mean_evaluate_s=GPUmean,CPU_to_GPU_ratio=CPUmean/GPUmean,limitations=p['limits']);guard('final')
 except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
 return s

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 if sha(Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 if any(GPU in r for r in apps):raise ValueError('GPU1notidle')
 e=dict(controller_pid=os.getpid(),terminal=False,completed=False,protocol_sha256=digest,hard_cap_s=180);write(OP/'execution.json',e);start=time.monotonic()
 def child(pid):e['worker_pid']=pid;write(OP/'execution.json',e)
 try:
  with (OP/'worker.log').open('x') as log:e.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.benchmark_p2_warp_root_contact_v02','run','--sha',digest],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=180,on_child=child))
 except BaseException as error:e['controller_error']=repr(error)
 finally:
  s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=e.get('exit_code')==0 and s.get('completed') is True,passed=s.get('passed',False),elapsed_s=time.monotonic()-start);write(OP/'execution.json',e)
 return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
