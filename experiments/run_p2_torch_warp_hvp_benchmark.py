"""Prospective paired fixed165888tet linear corrections; HVP backend only.

Both cases use Keast24, identical Q3 initial field and uniform4 coarse
preconditioner. No Newton/dynamic solve or force/material/contact changes.
"""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.benchmark_p2_warp_rhs import ENV_SITE
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
OP=ROOT/'runs/operator-dev/p2-torch-warp-hvp-benchmark-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-torch-warp-hvp-benchmark-v01'
GPU='GPU-571378aa-c44e-ffb8-b346-1e409f9821f9'
PREVIOUS=ROOT/'runs/operator-dev/p2-graded-coarse-benchmark-v01/protocol.json'
NEW_SOURCES=['experiments/campaign_p2_warp_bulk_hvp.py','tests/test_campaign_p2_warp_bulk_hvp.py','experiments/campaign_p2_torch_warp_hvp.py','tests/test_campaign_p2_torch_warp_hvp.py','experiments/run_p2_torch_warp_hvp_benchmark.py','tests/test_run_p2_torch_warp_hvp_benchmark.py','experiments/campaign_owned_gpu_budget_v02.py','tests/test_campaign_owned_gpu_budget_v02.py','experiments/benchmark_p2_warp_rhs.py']

def gates(row):
 return bool(row['old_HVP_true_R_N']<=1e-10 and row['original_candidate_free_R_N']<=1e-10 and row['maximum_constraint_m']==0 and row['minimum_J']>.1 and row['positive_Bernstein'] and row['maximum_candidate_vs_exact80_m']<=1e-8)

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 old=json.loads(PREVIOUS.read_text());pins=dict(old['source_sha256'])
 for f,h in pins.items():
  if sha(ROOT/f)!=h:raise ValueError('Oldsourcechanged:'+f)
 pins.update({f:sha(ROOT/f) for f in NEW_SOURCES});evidence={str(PREVIOUS.relative_to(ROOT)):sha(PREVIOUS)}
 for f in ['runs/operator-dev/p2-warp-bulk-hvp-v01/CPU-result.json','runs/operator-dev/p2-torch-warp-hvp-v01/CPU-result.json']:
  if json.loads((ROOT/f).read_text()).get('passed') is not True:raise ValueError('CPUcontrolsrequired')
  evidence[f]=sha(ROOT/f)
 for r in [old['reference_Q3'],old['reference_exact80']]:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Immutablefieldchanged')
 cores=sorted(os.sched_getaffinity(0))[-2:]
 p=dict(schema='fixed165888-sameoperator-Torch-vs-WarpHVP-v1',status='FROZEN_PRE_GPU_PENDING_ROOT_REVIEW_AND_GPU2_RELEASE',source_sha256=pins,evidence_sha256=evidence,installed_warp_API_sha256={f:sha(ENV_SITE/f) for f in ['warp/_src/torch.py','warp/_src/context.py']},reference_Q3=old['reference_Q3'],reference_exact80=old['reference_exact80'],cells=[48,24,24],fine_tets=165888,models=['original_TorchHVP_uniform4','WarpInteropHVP_uniform4'],same_operator=old['same_physical_operator'],changed='ONLYelasticHVP implementation; nohostarraycopies; WarpcurrentTorchCUDAstream; originalevaluate/force/J/nodeblocks/fullmass/preconditioner/CG/finalR unchanged',initialization=old['initialization'],linear_system=old['linear_system'],CG_rtol=old['CG_rtol'],CG_atol_N=1e-12,CG_cap=4000,linear_solves=2,Newton_solves=0,timing='Bothgeometry/preconditionerobjectsconstructedbeforetiming; oneHVPwarmup eachatSAMEfield/direction; WarpJITrecorded/excluded; syncbefore/after fullCG, sameuniform4 preconditioner and one linear correction each; originalfirst andnewsecond, no broadGPU speedclaim',gates=old['gates'],independent_acceptance='OldTorchHVP validates BOTHlinear directions; originalTorchgradient validates BOTHcandidatefields; wholeBernsteinJ/exactclamp/frozen80target required',GPU_uuid=GPU,CPU_threads=2,CPU_affinity_cores=cores,hard_cap_s=600,resource_limits=LIMITS,taskset=dict(path='/usr/bin/taskset',sha256=sha(Path('/usr/bin/taskset'))),limits=['Pairedfixed-discretizationCGcontrolonly','NoGPUcallsuntilrootreviewandGPU2previousphaseactualreaped','NofullNewton/trajectory/rerunreference/spatialadmission','Allnegativephasesandoldsourceunchanged'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('evidence_sha256',ROOT),('installed_warp_API_sha256',ENV_SITE)]:
  for f,h in p[group].items():
   if sha(base/f)!=h:raise ValueError('Changedpin:'+f)
 for r in [p['reference_Q3'],p['reference_exact80']]:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changedreference')
 return p

def run(digest):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('ExplicitGPU2UUIDrequired')
 OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,cases=[],linear_solves=0,Newton_solves=0,pid=os.getpid())
 def save():s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 try:
  import torch,sys
  sys.path.append(str(ENV_SITE));import warp as wp
  from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry
  from experiments.campaign_p2_streaming import StreamingTwoLevelP2
  from experiments.campaign_p2_torch_warp_hvp import TorchWarpBulkHVPMixin
  from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
  if wp.__version__!='1.18.0':raise ValueError('PinnedWarp1.18required')
  wp.config.kernel_cache_dir=str(OP/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());torch.cuda.set_per_process_memory_fraction(p['resource_limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.ones(1,device='cuda:0').sum().item()
  def check(tag):
   row=dict(tag=tag,unix_time_s=time.time(),**collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch));validate_budget(row,limits=p['resource_limits']);write(OP/'resource-latest.json',row)
   with (OP/'resource-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  check('startup_ownGPU');s['versions']=dict(torch=torch.__version__,warp=wp.__version__,numpy=np.__version__);save()
  with np.load(ROOT/p['reference_Q3']['path'],allow_pickle=False) as z:q3=z['positions_m'].copy();rest=z['rest_m'].copy();force=z['applied_force_N'].copy()
  with np.load(ROOT/p['reference_exact80']['path'],allow_pickle=False) as z:q80=z['positions_m'].copy()
  m=KeastBulkP2Geometry(tuple(p['cells']),mapping='clamp_graded')
  if not np.array_equal(m.rest,rest) or not np.array_equal(m.load,force):raise ValueError('Physicaldiscreteinputsnotidentical')
  class GuardMixin:
   def tangent(self,cache,v):
    result=super().tangent(cache,v);self.calls+=1
    if self.calls%16==0:check(self.label+'_HVP_'+str(self.calls))
    return result
  class Old(GuardMixin,StreamingTwoLevelP2):pass
  class New(GuardMixin,TorchWarpBulkHVPMixin,StreamingTwoLevelP2):pass
  old=Old(m,'cuda:0',4000,8);new=New(m,'cuda:0',4000,8)
  for t,name in [(old,'original'),(new,'interop')]:t.calls=0;t.label=name
  u=torch.tensor(q3-m.rest,dtype=torch.float64,device='cuda:0');oc=old.evaluate(u);nc=new.evaluate(u);R=(oc['gradient']-old.force).masked_fill(old.mask,0.);norm=float(torch.linalg.vector_norm(R));rtol=min(.01,max(1e-8,.1*norm**.5));torch.testing.assert_close(nc['gradient'],oc['gradient'],atol=0,rtol=0)
  d0=R/torch.linalg.vector_norm(R);torch.cuda.synchronize();clock=time.monotonic();ow=old.tangent(oc,d0);torch.cuda.synchronize();s['old_warm_HVP_s']=time.monotonic()-clock;clock=time.monotonic();nw=new.tangent(nc,d0);torch.cuda.synchronize();s['new_first_HVP_including_JIT_s']=time.monotonic()-clock;s['warm_HVP_L2_difference_N']=float(torch.linalg.vector_norm(ow-nw));s['initial_free_R_N']=norm;s['requested_CG_rtol']=rtol;check('both_warmed');save()
  directions=[]
  for t,cache,name in [(old,oc,p['models'][0]),(new,nc,p['models'][1])]:
   torch.cuda.synchronize();clock=time.monotonic();d,k=t.cg(cache,-R,0.,rtol=rtol,atol=1e-12);torch.cuda.synchronize();linear=time.monotonic()-clock;s['linear_solves']+=1
   true=(-R-old.tangent(oc,d)).masked_fill(old.mask,0.);trial=old.evaluate(u+d);q=(old.rest+u+d).cpu().numpy();cert=whole_p2_jacobian_bounds(m,q)
   row=dict(name=name,completed=True,CG_iterations=k,linear_s=linear,old_HVP_true_R_N=float(torch.linalg.vector_norm(true)),original_candidate_free_R_N=float(torch.linalg.vector_norm((trial['gradient']-old.force).masked_fill(old.mask,0.))),maximum_constraint_m=float(abs((u+d)[old.mask]).max()),minimum_J=float(trial['minimum_J']),positive_Bernstein=cert['all_elements_positive'],Bernstein_minimum_lower_bound=cert['minimum_lower_bound'],maximum_candidate_vs_exact80_m=float(np.linalg.norm(q-q80,axis=1).max()),coarse_free_dofs=int(t.coarse_free.numel()),linear_records=t.linear_records)
   row['passed']=gates(row);direction=d.cpu().numpy();directions.append(direction);f=OUT/(name+'-correction.npz');np.savez_compressed(f,correction_m=direction);row['capture']=dict(path=str(f.relative_to(ROOT)),sha256=sha(f));s['cases'].append(row);save();check(name+'_complete')
   if not row['passed']:raise RuntimeError('OriginalHVP/gradient/certificategatesfailed')
  delta=float(np.linalg.norm(directions[0]-directions[1],axis=1).max());s.update(completed=True,passed=delta<=1e-8 and all(x['passed'] for x in s['cases']),maximum_same_system_direction_difference_m=delta,new_to_old_linear_time_ratio=s['cases'][1]['linear_s']/s['cases'][0]['linear_s'],maximum_torch_GPU_allocated_bytes=torch.cuda.max_memory_allocated());check('final')
 except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:save()
 return s

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 if sha(Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 if any(r.split(',')[1].strip()==GPU for r in rows if ',' in r):raise ValueError('GPU2notidle')
 e=dict(controller_pid=os.getpid(),terminal=False,completed=False,protocol_sha256=digest,hard_cap_s=600);write(OP/'execution.json',e);begin=time.monotonic()
 def child(pid):e['worker_pid']=pid;write(OP/'execution.json',e)
 try:
  with (OP/'worker.log').open('x') as log:e.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_torch_warp_hvp_benchmark','run','--sha',digest],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=600,on_child=child))
 except BaseException as error:e['controller_error']=repr(error)
 finally:
  s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=e.get('exit_code')==0 and s.get('completed') is True,passed=s.get('passed',False),elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e)
 return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
