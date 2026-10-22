"""Prospective joint space/time check, fixed pure-gravity case, no admission.

Four fresh-rest trajectories with explicit independent per-case hard budgets.
No integration at import/prepare; source/protocol review is required beforehand.
"""
from pathlib import Path
import argparse,ast,json,os,sys,time,threading,traceback,subprocess
import numpy as np
from experiments.run_p2_supported_gravity_parity import ROOT,ENV_SITE,GPU,sha,write,record,runtime_pins
from experiments.campaign_c38_static_family import build_geometry,array_record
from experiments.campaign_supported_gravity_dop import WarpBulkP2Force,SupportedBulkDynamics,AcceptedDOPCheckpoint,streaming_dop853
from experiments.campaign_consistent_mass_pcg import ConsistentMassPCG
from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
from experiments.campaign_static_p2_block import shapes
from experiments.diagnose_nonlinear_physical_probes import interpolate
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
OP=ROOT/'runs/operator-dev/p2-supported-gravity-fine-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-fine-v01'
POLICY=dict(every_accepted_steps=32,on_initial=True,on_new_common_output=True,on_final=True,max_uncommitted_accepted_steps_bound=32)
CONFIGS=[dict(id='coarse',rtol=1e-7,u_atol_m=1e-11,v_atol_m_s=1e-9,max_step_s=1e-4),dict(id='refined',rtol=1e-9,u_atol_m=1e-13,v_atol_m_s=1e-11,max_step_s=5e-5)]

def probes():return np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
def probe_map(model):
 _,ids,bary=interpolate(model.rest[:model.tets.max()+1],model.tets,model.rest[None,:model.tets.max()+1],probes());return model.elements[ids],shapes(bary)
def query_field(field,ids,weights):return np.einsum('pa,tpai->tpi',weights,np.asarray(field)[:,ids])
def conditional_indicator(delta,hcoarse,hfine):
 if not np.isfinite([delta,hcoarse,hfine]).all() or delta<0 or not hcoarse>hfine>0:raise ValueError('Positive actual mesh ratio and finite nonnegative discrepancy required')
 return float(delta/(hcoarse/hfine-1))

def source_closure():
 todo=[Path(__file__),ROOT/'tests/test_run_p2_supported_gravity_fine.py'];seen={}
 while todo:
  f=todo.pop();key=str(f.relative_to(ROOT))
  if key in seen:continue
  seen[key]=sha(f)
  for n in ast.walk(ast.parse(f.read_text())):
   if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments.'):
    p=ROOT/(n.module.replace('.','/')+'.py')
    if p.exists():todo.append(p)
 return dict(sorted(seen.items()))

def torch_runtime():
 import torch
 return dict(version=str(torch.__version__),CUDA_version=torch.version.cuda)

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserve prospective four-case plan')
 evidence=[]
 for folder in ['p2-supported-gravity-parity-v01','p2-supported-gravity-capacity-v01','p2-supported-mass-pcg-v01']:
  a=ROOT/'runs/gaussian-campaign-v01'/folder/'summary.json';b=ROOT/'runs/operator-dev'/folder/'execution.json';s=json.loads(a.read_text());e=json.loads(b.read_text())
  if not s['completed'] or not (s.get('passed') or s.get('backend_parity_passed')) or not e['terminal'] or e['exit_code']!=0:raise ValueError('Completed prerequisite required '+folder)
  evidence += [record(a),record(b)]
 cap=json.loads((ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-capacity-v01/summary.json').read_text());cases=[]
 for cells,budgets in [([20,10,10],[1200,2400]),([24,12,12],[1800,3600])]:
  row=next(x for x in cap['meshes'] if x['cells']==cells)
  for config,budget in zip(CONFIGS,budgets):cases.append(dict(id=str(row['tetrahedra'])+'tets-'+config['id'],cells=cells,tetrahedra=row['tetrahedra'],nodes=row['nodes'],config=config,wall_cap_s=budget,expected_rest=row['rest'],expected_elements=row['elements'],expected_body_force=row['force'],expected_hmax_m=row['hmax_m']))
 p=dict(schema='supported-pure-gravity-fine-joint-space-time-v0.1',status='FROZEN_BEFORE_FINE_TRAJECTORIES_REQUIRES_ROOT_REVIEW',gpu_uuid=GPU,CPU_threads=2,CPU_affinity_cores=sorted(os.sched_getaffinity(0))[:2],resource_limits=LIMITS,source_sha256=source_closure(),runtime_source_sha256=runtime_pins(),torch_runtime=torch_runtime(),taskset=dict(path='/usr/bin/taskset',sha256=sha('/usr/bin/taskset')),prerequisite_evidence=evidence,cases=cases,total_per_case_caps_s=9000,physics=dict(E_Pa=100000.,nu=.35,rho_kg_m3=625.,gravity_m_s2=9.81,support='both_ends',surface_force_N=0.,body_force_only=True,initial_u_v='zero fresh REST everycase',law='unchanged polynomialstableNH, Keast24degree6exactbulk, original80+10Jguards',mass='unchangedfullanalyticT10consistentCSR; strictGPUFP64PCG, diagonalPRECONDITIONERonly',no_contact_or_damping=True),mass_solver=dict(absolute_tolerance_N=1e-12,relative_tolerance=1e-11,max_iterations=256,true_residual_every=8),outputs_s=np.linspace(0.,.1,25).tolist(),material_probes_m=probes().tolist(),checkpoint_policy=POLICY,gates=dict(mass_physical_R_N=1e-9,energy_error_J=1e-8,sampled_J_strictly_above=.1,whole_Bernstein_at_common_strictly_above=0.,fixed_u_v_a=0,temporal_allnode_q_m=20e-6,temporal_allnode_v_m_s=1e-4,accepted_partitions_must_differ=True,spatial_63query_q_indicator_m=20e-6,spatial_63query_v_indicator_m_s=1e-4),spatial_metric='Max vector discrepancy at all25x63materialqueries, dividedby(actualhcoarse/hfine-1); conditionalp>=1engineeringindicator NOT rigorousbound',temporal_metric='Same-mesh ALLnodes/25clocks; JOINTtolerance+step-caprefinement, no isolatedorderclaim; byteidenticalacceptedpartitions mean agreementONLY',cost_estimate=dict(measured_mass_PCG_ms=dict(tets12000=[25,29],tets20736=[32,36]),measured_bulk_force_ms=dict(tets12000=[6,7],tets20736=[11,12]),minimum_combined_RHS_calls_approx=72000,minimum_full_stage_s_approx=3000,adaptive_partition_unknown=True,cap_reason='Per-casecaps allow~2xadaptiveheadroom/output/checkpoint/wholeJ; totalmaximum9000s, noretries'),durability='Ownednewprocesspercase; stopnewcasesonfirstfailure, reap/drain; currentacceptedstate/f/h pluscomplete scalarprefix/commonstates every32accepted orcommon/final; rawallacceptedfullfieldsnotstored',nonclaims=['No inheritedstaticspaceorcontinuumdynamicqualification','No contact/materialpopulation/image/inference/policy admission','No claim of isolatedtime-stepconvergenceorder','No actualobservation/truth/modelselection inputs; allfouroutcomesplannedbeforecalls'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed protocol')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 if runtime_pins()!=p['runtime_source_sha256'] or torch_runtime()!=p['torch_runtime']:raise ValueError('Changed runtime')
 for r in p['prerequisite_evidence']:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed prerequisite')
 return p

class ReportingPCG(ConsistentMassPCG):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.calls=0;self.maximum_iterations=0;self.maximum_true_R_N=0.;self.total_iterations=0;self.restarts=0
 def solve(self,rhs):
  answer=super().solve(rhs);r=self.last_record;self.calls+=1;self.maximum_iterations=max(self.maximum_iterations,r['iterations']);self.total_iterations+=r['iterations'];self.maximum_true_R_N=max(self.maximum_true_R_N,r['true_residual_N']);self.restarts+=r['restarts'];return answer
 def statistics(self):return dict(calls=self.calls,maximum_iterations=self.maximum_iterations,total_iterations=self.total_iterations,maximum_true_R_N=self.maximum_true_R_N,restarts=self.restarts)

def run_case(digest,case_id):
 p=validate(digest);case=next(x for x in p['cases'] if x['id']==case_id)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('GPU0 UUID only')
 directory=OUT/case_id;directory.mkdir(parents=True,exist_ok=False);start=time.monotonic();result=dict(id=case_id,completed=False,passed=False,protocol=record(OP/'protocol.json'),case=case);write(directory/'result.json',result)
 try:
  import torch
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0' or not Path(wp.__file__).resolve().is_relative_to(ENV_SITE):raise ValueError('Pinned Warp required')
  wp.config.kernel_cache_dir=str(OP/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());torch.cuda.set_per_process_memory_fraction(min(.95,LIMITS['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory));torch.ones(1,device='cuda:0').sum().item()
  stop=threading.Event();lock=threading.Lock();measurements=[]
  def guard(tag,reserve_bytes=256*1024**2):
   with lock:
    a=collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch);validate_budget(a,reserve_write_bytes=reserve_bytes,reserve_memory_bytes=reserve_bytes,limits=p['resource_limits']);measurements.append(a);write(directory/'resource-latest.json',dict(tag=tag,**a))
  def monitor():
   while not stop.wait(1):
    try:guard('watchdog')
    except BaseException as error:write(directory/'resource-failure.json',dict(error=repr(error)));os.kill(os.getpid(),__import__('signal').SIGTERM);return
  guard('before_geometry');thread=threading.Thread(target=monitor,daemon=True);thread.start();m=build_geometry(tuple(case['cells']),100000.,'both_ends',Fz=0.,rule='keast24')
  for a,key in [(m.rest,'expected_rest'),(m.elements,'expected_elements'),(m.load,'expected_body_force')]:
   if array_record(a)!=case[key]:raise ValueError('Changed actual geometry/knownforce')
  if m.maximum_reference_tet_edge_m!=case['expected_hmax_m']:raise ValueError('Changed actualh')
  backend=WarpBulkP2Force(m,device='cuda:0');d=SupportedBulkDynamics(m,backend)
  # Instance-only numerical solver replacement, old helper/source untouched.
  d.mass_lu=ReportingPCG(d.mass_ff.tocsr(),device='cuda:0',absolute_tolerance_N=p['mass_solver']['absolute_tolerance_N'],relative_tolerance=p['mass_solver']['relative_tolerance'],max_iterations=p['mass_solver']['max_iterations'],true_residual_every=p['mass_solver']['true_residual_every'])
  ids,weights=probe_map(m);metadata=dict(protocol_sha256=digest,case=case,source_sha256=p['source_sha256'],checkpoint_policy=POLICY,actual_model=dict(rest=array_record(m.rest),elements=array_record(m.elements),force=array_record(m.load),mask=array_record(m.mask)))
  class GuardedCheckpoint(AcceptedDOPCheckpoint):
   def __call__(self,*args):guard('before_checkpoint');return super().__call__(*args)
  checkpoint=GuardedCheckpoint(directory/'checkpoint',metadata);zero=np.zeros_like(m.rest);initial=d.pack(zero,zero);c=case['config'];atol=np.r_[np.full(d.size,c['u_atol_m']),np.full(d.size,c['v_atol_m_s'])]
  def rhs(t,y):
   try:return d.rhs(t,y)
   except BaseException as error:
    np.savez(directory/'failed-RHS-state.npz',time_s=t,y=y,error=repr(error),last_mass_solve_json=json.dumps(d.mass_lu.last_record));raise
  answer=streaming_dop853(rhs,d.telemetry,initial,p['outputs_s'],rtol=c['rtol'],atol=atol,max_step_s=c['max_step_s'],energy_tolerance_J=p['gates']['energy_error_J'],checkpoint=checkpoint,checkpoint_every_accepted_steps=32);u,v=map(np.asarray,zip(*[d.unpack(y) for y in answer['common_states']]));q=u+m.rest;Q=query_field(q,ids,weights);V=query_field(v,ids,weights);bounds=np.array([whole_p2_jacobian_bounds(m,x)['minimum_lower_bound'] for x in q]);reactions=np.array([d.mechanics(y)['reaction_N'] for y in answer['common_states']]);guard('before_capture');M=d.mass
  if bounds.min()<=0:raise ValueError('Whole-element common-output positivity certificate failed')
  file=directory/'trajectory.npz';np.savez(file,times_s=answer['times_s'],positions_m=q,displacements_m=u,velocities_m_s=v,reactions_N=reactions,rest_m=m.rest,tets=m.tets,elements=m.elements,fixed_mask=m.mask,known_body_force_N=m.load,mass_data_kg=M.data,mass_indices=M.indices,mass_indptr=M.indptr,mass_shape=M.shape,material_probes_m=probes(),probe_positions_m=Q,probe_velocities_m_s=V,accepted_times_s=np.array([r['time_s'] for r in answer['accepted_scalar_ledger']]),common_whole_J_lower=bounds)
  write(directory/'accepted-ledger.json',answer['accepted_scalar_ledger']);result.update(completed=True,passed=True,capture=record(file),accepted_scalar_ledger=record(directory/'accepted-ledger.json'),accepted_steps=answer['accepted_steps'],nfev=answer['nfev'],maximum_RHS_mass_residual_N=d.maximum_RHS_mass_residual_N,minimum_RHS_sampled_J=d.minimum_RHS_J,minimum_common_whole_J_lower=float(bounds.min()),maximum_energy_error_J=max(abs(r['mechanical_energy_J']-answer['initial_mechanical_energy_J']) for r in answer['accepted_scalar_ledger']+answer['common_telemetry']),mass_PCG=d.mass_lu.statistics(),checkpoint_commits=answer['checkpoint_commits'],maximum_observed_uncommitted_accepted_steps=answer['maximum_observed_uncommitted_accepted_steps'])
 except BaseException as error:
  result.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
  if 'd' in locals() and isinstance(d.mass_lu,ReportingPCG):result['last_mass_solve']=d.mass_lu.last_record
 finally:
  if 'stop' in locals():stop.set()
  if 'thread' in locals():thread.join(timeout=2)
  if 'measurements' in locals() and measurements:result['maximum_observed_resources']={k:max(a[k] for a in measurements) for k in ['gpu_bytes','rss_bytes','disk_bytes']}
  result['elapsed_s']=time.monotonic()-start;write(directory/'result.json',result)
 return result

def compare(p,results):
 for r in results:
  if sha(ROOT/r['capture']['path'])!=r['capture']['sha256']:raise ValueError('Changed completed trajectory evidence')
 temporal=[];spatial=[]
 for n in [12000,20736]:
  a,b=[next(r for r in results if r['id']==str(n)+'tets-'+name) for name in ['coarse','refined']]
  with np.load(ROOT/a['capture']['path']) as x,np.load(ROOT/b['capture']['path']) as y:
   np.testing.assert_array_equal(x['times_s'],y['times_s']);q=float(np.linalg.norm(x['positions_m']-y['positions_m'],axis=-1).max());v=float(np.linalg.norm(x['velocities_m_s']-y['velocities_m_s'],axis=-1).max());distinct=not np.array_equal(x['accepted_times_s'],y['accepted_times_s'])
  temporal.append(dict(tetrahedra=n,allnode_position_difference_m=q,allnode_velocity_difference_m_s=v,accepted_partitions_differ=distinct,passed=bool(q<=20e-6 and v<=1e-4 and distinct),agreement_only_if_partitions_identical=not distinct))
 for label in ['coarse','refined']:
  a,b=[next(r for r in results if r['id']==str(n)+'tets-'+label) for n in [12000,20736]]
  with np.load(ROOT/a['capture']['path']) as x,np.load(ROOT/b['capture']['path']) as y:
   np.testing.assert_array_equal(x['times_s'],y['times_s']);np.testing.assert_array_equal(x['material_probes_m'],y['material_probes_m']);q=float(np.linalg.norm(x['probe_positions_m']-y['probe_positions_m'],axis=-1).max());v=float(np.linalg.norm(x['probe_velocities_m_s']-y['probe_velocities_m_s'],axis=-1).max())
  qi=conditional_indicator(q,a['case']['expected_hmax_m'],b['case']['expected_hmax_m']);vi=conditional_indicator(v,a['case']['expected_hmax_m'],b['case']['expected_hmax_m']);spatial.append(dict(configuration=label,maximum_63query_q_difference_m=q,maximum_63query_v_difference_m_s=v,conditional_p1_q_indicator_m=qi,conditional_p1_v_indicator_m_s=vi,passed=qi<=20e-6 and vi<=1e-4,scope='Conditionalengineeringindicatorunderassumedorder>=1; notrigorousbound'))
 return temporal,spatial

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('No automatic retries')
 if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 if GPU in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):raise RuntimeError('GPU0 not idle')
 OUT.mkdir(parents=True,exist_ok=False);e=dict(controller_pid=os.getpid(),terminal=False,completed=False,qualified=False,protocol_sha256=digest,cases=[dict(id=c['id'],status='QUEUED') for c in p['cases']]);write(OP/'execution.json',e);start=time.monotonic();results=[]
 try:
  for case,row in zip(p['cases'],e['cases']):
   validate(digest);row['status']='LIVE';write(OP/'execution.json',e)
   def on_child(pid):row['worker_pid']=pid;write(OP/'execution.json',e)
   with (OP/(case['id']+'.log')).open('x') as log:row.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_supported_gravity_fine','case','--sha',digest,'--case-id',case['id']],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=case['wall_cap_s'],on_child=on_child))
   f=OUT/case['id']/'result.json';r=json.loads(f.read_text()) if f.exists() else {};row['result']=record(f) if f.exists() else None;row['status']='COMPLETE' if row.get('exit_code')==0 and r.get('completed') and r.get('passed') else 'FAILED';write(OP/'execution.json',e)
   if row['status']!='COMPLETE':raise RuntimeError('Stopnewcases/drainafterfirstfailure; remainingQUEUED retained')
   results.append(r)
  temporal,spatial=compare(p,results);e.update(completed=True,qualified=all(r['passed'] for r in temporal+spatial),temporal=temporal,spatial=spatial)
 except BaseException as error:e['error']=repr(error)
 finally:e.update(terminal=True,elapsed_s=time.monotonic()-start);write(OP/'execution.json',e);write(OUT/'summary.json',dict(**e,nonclaims=p['nonclaims']))
 return e

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','launch','case']);a.add_argument('--sha');a.add_argument('--case-id');args=a.parse_args();r=prepare() if args.action=='prepare' else launch(args.sha) if args.action=='launch' else run_case(args.sha,args.case_id);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','qualified','elapsed_s','error']}));raise SystemExit(0 if args.action=='prepare' or r.get('completed') else 1)
