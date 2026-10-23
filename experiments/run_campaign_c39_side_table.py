"""NEW typed-public-action nonlinear P2 worker; original solver/law/gates unchanged.

No global monkeypatching. Bulk, preconditioner, checkpoints and acceptance are
copied unchanged from the immutable C38 worker; only explicit force compilation
and a new exact ten-state catalog/controller are added. No observation draws.
"""
import argparse,ast,json,multiprocessing,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_campaign_c38_graded_family_warp import ROOT,ENV_SITE,GATES,write,sha,record,guard_resources
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry,MAPPING
from experiments.campaign_c39_known_actions import compile_known_action
from experiments.campaign_c38_static_family import wrench
from experiments.campaign_owned_gpu_budget_v02 import validate_cpu
SOURCE='experiments/run_campaign_c39_side_table.py';TEST='tests/test_run_campaign_c39_side_table.py'
OP=ROOT/'runs/operator-dev/c39-side-nonlinear-table-v01';OUT=ROOT/'runs/gaussian-campaign-v01/c39-side-nonlinear-table-v01'
GPUS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b','GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','GPU-571378aa-c44e-ffb8-b346-1e409f9821f9')
MESHES=((32,16,16),(40,20,20))

def catalog():
 rows=[]
 for material_id,fraction,use in [('Elo',0.,'TABLE_NODE'),('ElogQ1',.25,'WITHHELD_INTERPOLATION_VALIDATION'),('Emid',.5,'TABLE_NODE'),('ElogQ3',.75,'WITHHELD_INTERPOLATION_VALIDATION'),('Ehi',1.,'TABLE_NODE')]:
  E=float(np.sqrt(75000.*150000.)) if fraction==.5 else float(75000.*2**fraction)
  for support in ('both_ends','separated_bottom_patches'):
   rows.append(dict(id='C39__'+material_id+'__'+support+'__side2',young_Pa=E,support=support,Fz_N=0.,action_id='middle_side_2N',force_vector_N=[0.,2.,0.],kind='NEW_C39_nonlinear_side_table',material_id=material_id,log_material_fraction=fraction,material_use=use,force_id='side2',reused_original_case=False))
 return rows

def build_model(cells,case,*,rule):
 if case.get('action_id')!='middle_side_2N' or case.get('force_vector_N')!=[0.,2.,0.] or case.get('Fz_N')!=0.:raise ValueError('Explicit typed public SIDE2 action required; Fz alone cannot specify action')
 m=SupportedBoundaryGradedGeometry(cells,case['young_Pa'],case['support'],0.,rule=rule);compiled=compile_known_action(m.rest,m.tets,m.faces,m.elements,case['action_id'])
 np.testing.assert_array_equal(compiled['body_N'],m.public_force_components['body_N']);m.load=compiled['total_N'];m.force_metadata=compiled['metadata'];m.public_force_components=compiled
 return m

def worker(p,protocol,case_id):
    rows=[r for r in p['cases'] if r['id']==case_id]
    if len(rows)!=1:raise ValueError('Exactly declared case id required before output/physics')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=p['resources']['physical_GPU_UUID']:raise ValueError('Exactly declared physical GPU UUID visibility required')
    case=rows[0];directory=ROOT/p['output']/case_id;directory.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(case=case,protocol=record(protocol),pid=os.getpid(),completed=False,passed=False,meshes=[],raw_local=True,domain_or_dataset_admitted=False)
    def save():state['seconds']=time.monotonic()-start;write(directory/'result.json',state)
    def terminated(*_):raise TimeoutError('Owned per-case wall/RSS guard; acceptediterates retained')
    signal.signal(signal.SIGTERM,terminated)
    from experiments.run_static_p2_axial_continuation import watchdog
    watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+p['resources']['per_case_wall_cap_s'],p['resources']['RSS_cap_bytes'],directory/'resource-guard.json'),daemon=True);watcher.start();save()
    try:
        import torch
        sys.path.append(str(ENV_SITE))
        import warp as wp
        wp.config.kernel_cache_dir=str(directory/'warp-cache')
        from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
        from experiments.campaign_c38_verified_warp_solver import IndependentlyCheckedWarpCGMixin
        from experiments.campaign_p2_checkpointed import CheckpointedStaticSolveMixin,AcceptedStaticCheckpoint,checkpoint_identity
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        from experiments.campaign_newton_stable_material import StableP2Block
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads())
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactlyone visibleCUDAdevice required')
        torch.cuda.set_per_process_memory_fraction(min(1.,p['resources']['GPU_allocated_cap_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
        torch.zeros(1,device='cuda:0').sum().item();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);own=[line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip()==str(os.getpid())];state['GPU_provenance']=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],torch_device='cuda:0',own_physical_GPU_UUIDs=own);save()
        if set(own)!={p['resources']['physical_GPU_UUID']}:raise ValueError('OwncomputePID must occupy only the declared physical GPU')
        state['startup_resources']=guard_resources(p,gpu=True);save()
        class Solver(CheckpointedStaticSolveMixin,IndependentlyCheckedWarpCGMixin,StreamingSupportedP2):
            def evaluate(self,u):
                value=super().evaluate(u);guard_resources(p,gpu=True);return value
            def tangent(self,c,v):
                value=super().tangent(c,v);self.budget_HVP_calls=getattr(self,'budget_HVP_calls',0)+1
                if self.budget_HVP_calls%16==0:guard_resources(p,gpu=True)
                return value
        def geometry(cells,rule):return build_model(tuple(cells),case,rule=rule)
        def solver(model):return Solver(model,'cuda:0',p['solver']['CG_cap'],p['solver']['quadrature_chunk_size'],support_kind=case['support'])
        def solve(model,name):
            t=solver(model);identity=checkpoint_identity(model,t,source_sha256=p['source_sha256'],protocol_sha256=sha(protocol));checkpoint=AcceptedStaticCheckpoint(directory/name/'accepted',identity)
            def guarded_checkpoint(value):
                guard_resources(p,gpu=True);checkpoint(value);guard_resources(p,gpu=True)
            answer=t.solve(force_tolerance=p['solver']['force_tolerance_N'],maximum_iterations=p['solver']['Newton_cap'],on_accepted=guarded_checkpoint);return t,answer
        points=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
        fields=[];state['new_resolution_solves']=0;state['historical_bulk_controls_reused']=True;save()
        for cells in p['meshes']:
            begin=time.monotonic();name='mesh'+str(cells[0]);state['stage']=name;save();m=geometry(cells,'keast24');t,a=solve(m,name);u=a['displacement'].detach().clone();q=a['positions'].detach().cpu().numpy();Q=m.query(q,points);r24=float(torch.linalg.vector_norm(a['residual']));energy24=float(a['cache']['energy']);gradient24=a['cache']['gradient'].detach().clone();history=a['history'];iterations=a['iterations'];linear_records=t.linear_records;independent_linear_checks=t.independent_linear_checks;del a,t;torch.cuda.empty_cache()
            m80=geometry(cells,'exact80')
            for key in ('rest','tets','elements','faces','mask','load'):
                if not np.array_equal(getattr(m,key),getattr(m80,key)):raise ValueError('Sameu graded exact24/80 geometry/constraints/force mismatch')
            t80=solver(m80);c80=t80.evaluate(u);full_residual=(c80['gradient']-t80.force).detach().cpu().numpy();r80=float(np.linalg.norm(full_residual[~m80.mask]));gradient_difference=float(torch.linalg.vector_norm(c80['gradient']-gradient24));energy_difference=abs(float(c80['energy'])-energy24);J=float(c80['minimum_J']);diag=m80.diagnostics(dict(positions=q,residual=full_residual,J=np.array([J]),tolerance_N=1e-10,iterations=iterations));certificate=whole_p2_jacobian_bounds(m80,q);components=m.public_force_components;actual_wrench={key:dict(zip(('resultant_N','moment_Nm'),wrench(q,components[key]))) for key in ('body_N','surface_N','total_N')}
            passed=bool(r24<=GATES['free_residual24_N'] and r80<=GATES['free_residual80_N'] and diag['force_balance_defect_N']<=GATES['force_balance_N'] and diag['actual_moment_balance_defect_Nm']<=GATES['actual_moment_balance_Nm'] and diag['maximum_constraint_error_m']==0 and J>GATES['minimum_sampled_J'] and certificate['all_elements_positive'] and gradient_difference<=GATES['gradient24_80_difference_N'] and energy_difference<=GATES['energy24_80_difference_J'])
            guard_resources(p,gpu=True);target=directory/name/'capture.npz';np.savez_compressed(target,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,displacement_m=u.detach().cpu().numpy(),constraint_mask=m.mask,known_body_force_N=components['body_N'],known_surface_force_N=components['surface_N'],known_total_force_N=m.load,material_probes_m=points,probe_positions_m=Q)
            row=dict(cells=cells,grading=m.grading_metadata,completed=True,physics_passed=passed,original_free_R24_N=r24,original_free_R80_N=r80,gradient24_80_difference_N=gradient_difference,energy24_80_difference_J=energy_difference,diagnostics=diag,whole_P2_J=dict(all_elements_positive=certificate['all_elements_positive'],minimum_lower_bound=certificate['minimum_lower_bound']),mask_trace=m.mask_trace,known_force_metadata=m.force_metadata,actual_deformed_wrenches=actual_wrench,hmax_m=m.maximum_reference_tet_edge_m,capture=record(target),history=history,linear_records=linear_records,independent_linear_checks=independent_linear_checks,all_independent_original_Torch_linear_checks_passed=all(v['passed'] for v in independent_linear_checks),seconds=time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());state['meshes'].append(row);state['new_resolution_solves']+=1;fields.append(Q);save()
            if not passed:raise RuntimeError('Original24/80physical/trace/quadrature gate failed; capturedfailure retained')
            del m,m80,t80,c80,u,gradient24;torch.cuda.empty_cache();print(case_id,name,'physicalPASS',round(row['seconds'],2),flush=True)
        deltas=[float(np.linalg.norm(b-a,axis=1).max()) for a,b in zip(fields[:-1],fields[1:])];factors=[1/(a['hmax_m']/b['hmax_m']-1) for a,b in zip(state['meshes'][:-1],state['meshes'][1:])];indicators=[a*b for a,b in zip(deltas,factors)];state.update(completed=True,physics_passed=True,passed=indicators[-1]<=GATES['conditional_actual_h_spatial_indicator_m'],adjacent_max63_query_differences_m=deltas,actual_h_indicator_factors=factors,conditional_spatial_indicators_m=indicators,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except BaseException as error:
        state.update(failure=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc());save();raise
    finally:save();watcher.terminate();watcher.join(timeout=2)
    return state

def sources():
 seen=set();pending=[SOURCE,TEST]
 while pending:
  name=pending.pop()
  if name in seen:continue
  seen.add(name)
  for node in ast.walk(ast.parse((ROOT/name).read_text())):
   if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
    dep='experiments/'+node.module.split('.')[1]+'.py'
    if (ROOT/dep).exists():pending.append(dep)
 return sorted(seen)
def phase_cases(index):return catalog()[index::3]
def prepare():
 fine=ROOT/'runs/operator-dev/c39-fine-linear-screen-v01/compact-summary.json';fine_result=json.loads(fine.read_text())
 if not all(fine_result[k] for k in ['completed','all_numeric_passed','all_q63_spatial_passed','fine_opposite_winner_passed']):raise ValueError('ALL declared fine linear prerequisite gates required')
 old=ROOT/'runs/operator-dev/c38-graded-family-warp-v01/protocol-v01.json';base=json.loads(old.read_text());oldsummary=ROOT/'runs/operator-dev/c38-nonlinear-table-v01/completed-summary.json';summary=json.loads(oldsummary.read_text())
 if not summary['completed'] or not summary['all40_declared_q63_static_gates_passed']:raise ValueError('Original qualified C38 fields required')
 reused=[]
 for r in summary['rows']:
  if r['case']['Fz_N'] in (-2.,0.,1.):reused.append({k:r[k] for k in ['case','source_result','external_cache','completed','physics_passed','passed','meshes']})
 if len(reused)!=30:raise ValueError('Exact unchanged30 gravity/down/canonical reuse records required')
 pins={f:sha(ROOT/f) for f in sources()};evidence={str(f.relative_to(ROOT)):sha(f) for f in [fine,fine.with_name('protocol.json'),fine.with_name('terminal-proof.json'),old,oldsummary]};paths=[]
 for index in range(3):
  cases=phase_cases(index);directory=OP/('gpu'+str(index));directory.mkdir(parents=True,exist_ok=True);path=directory/'protocol-v02.json'
  if path.exists():raise FileExistsError('Preserve prospective phase')
  p=dict(schema='C39-typed-public-SIDE-nonlinear-phase-v02',status='FROZEN_FOR_ROOT_REVIEW_BEFORE_NEW_NONLINEAR_SOLVES',phase_index=index,cases=cases,all_new_cases=catalog(),meshes=[list(x) for x in MESHES],source_sha256=pins,evidence_sha256=evidence,installed_warp_API_sha256=base['installed_warp_API_sha256'],output=str((OUT/('gpu'+str(index))).relative_to(ROOT)),owned_metadata_directory=str(directory.relative_to(ROOT)),CPU_affinity_cores=[2*index,2*index+1],taskset=base['taskset'],mapping=MAPPING,gates=GATES,solver=base['solver'],resources={**base['resources'],'physical_GPU_UUID':GPUS[index],'physical_GPU_index':index,'global_wall_cap_s':4500,'per_case_wall_cap_s':900,'absolute_phase_bound_including_cleanup_s':4503,'projected_remaining_writes_reserve_bytes':9*1024**3},counts=dict(new_cases=len(cases),new_fine_fields=2*len(cases),all_new_cases=10,all_new_fields=20,reused_C38_states=30,reused_C38_fields=60,composite_states=40,composite_fields=80),reused_C38_records=reused,initialization='Each NEW side32/40 FROMREST, exact same stableNH/Keast24 and independent80 residual/J/wrench/CG/trace gates. No C38 field reruns or force/material retuning.',CPU_controls='NEW typed-force geometry/force independence and exact24/80 inputs; manufactured finite-u NumPy/Torch parity receipt required before any nonlinear launch.',metadata_scope='TABLE_NODE and WITHHELD_INTERPOLATION_VALIDATION denote numerical material-table uses only, not sampled population roles.',failure_policy='No retries; stop new launches on incomplete/resource/physical failure, drain inflight and preserve all queued records. Completed spatial-negative cases retained and remaining cases may proceed.',scope='Only SIDE public action table. No observation/noise/inverse/learned/contact/dynamic calls. No continuous-E/wholefield/active-policy admission. Root reviews this new worker before launch.',raw_local=True)
  write(path,p);paths.append(record(path))
 return paths

def validate(protocol,digest):
 path=ROOT/protocol
 if sha(path)!=digest:raise ValueError('Changed phase protocol')
 p=json.loads(path.read_text());i=p['phase_index']
 if i not in range(3) or p['cases']!=phase_cases(i) or p['all_new_cases']!=catalog() or p['meshes']!=[list(x) for x in MESHES] or p['gates']!=GATES or p['mapping']!=MAPPING or p['solver']!=dict(Newton_cap=25,CG_cap=4000,quadrature_chunk_size=8,force_tolerance_N=1e-10):raise ValueError('Exact typed cases/original numerical gates required')
 if p['source_sha256']!={f:sha(ROOT/f) for f in sources()}:raise ValueError('Complete exact source closure required')
 if p['CPU_affinity_cores']!=[2*i,2*i+1] or p['resources']['physical_GPU_UUID']!=GPUS[i] or p['resources']['global_wall_cap_s']!=4500 or p['resources']['per_case_wall_cap_s']!=900 or p['resources']['GPU_allocated_cap_bytes']!=8*1024**3 or p['resources']['RSS_cap_bytes']!=12*1024**3 or p['resources']['disk_free_reserve_bytes']!=30*1024**3 or p['resources']['projected_remaining_writes_reserve_bytes']!=9*1024**3:raise ValueError('Exact reviewed resource partition required')
 for group in ['source_sha256','evidence_sha256']:
  if any(sha(ROOT/f)!=h for f,h in p[group].items()):raise ValueError('Changed '+group)
 if any(sha(ENV_SITE/f)!=h for f,h in p['installed_warp_API_sha256'].items()):raise ValueError('Changed Warp API')
 if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 return p,path

def launch():
 from experiments.campaign_owned_gpu_budget_v02 import terminate_owned
 if (OP/'execution.json').exists():raise FileExistsError('No retry')
 protocols=[OP/('gpu'+str(i))/'protocol-v02.json' for i in range(3)];phases=[validate(str(f.relative_to(ROOT)),sha(f))[0] for f in protocols];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(g in apps for g in GPUS):raise RuntimeError('All three declared GPUs must be idle')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,terminal=False,controller_pid=os.getpid(),records=[dict(phase_index=i,case=c,status='QUEUED') for i in range(3) for c in phases[i]['cases']],stop_new_launches=False);active={};logs=[]
 def stop(*_):raise InterruptedError('Owned global4500 controller stop')
 previous={x:signal.signal(x,stop) for x in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]};signal.alarm(4500)
 try:
  while True:
   if not s['stop_new_launches']:
    for i,p in enumerate(phases):
     if i in active:continue
     row=next((r for r in s['records'] if r['phase_index']==i and r['status']=='QUEUED'),None)
     if row is None:continue
     f=(OP/('gpu'+str(i))/(row['case']['id']+'.log')).open('x');logs.append(f);blocked=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
     try:
      child=subprocess.Popen([p['taskset']['path'],'-c',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_campaign_c39_side_table','worker','--protocol',str(protocols[i].relative_to(ROOT)),'--sha',sha(protocols[i]),'--case',row['case']['id']],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':p['resources']['physical_GPU_UUID'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});row.update(status='RUNNING',pid=child.pid);active[i]=(child,row,time.monotonic())
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
     write(OP/'execution.json',s)
   if not active:break
   for i,(child,row,t) in list(active.items()):
    if time.monotonic()-t>900 and child.poll() is None:row['timeout']=True;terminate_owned(child)
    if child.poll() is None:continue
    row.update(status='TERMINAL',exit_code=child.wait());result=ROOT/phases[i]['output']/row['case']['id']/'result.json';v=json.loads(result.read_text()) if result.exists() else {};row.update(completed=v.get('completed',False),physics_passed=v.get('physics_passed',False),passed=v.get('passed',False));row['source_result']=record(result) if result.exists() else None
    if result.exists():
     cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(result)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
     if cache.exists():assert cache.read_bytes()==result.read_bytes()
     else:cache.write_bytes(result.read_bytes())
     row['external_cache']=record(cache)
    if row['exit_code']!=0 or not row['completed'] or not row['physics_passed']:s['stop_new_launches']=True
    del active[i];write(OP/'execution.json',s)
   time.sleep(.2)
  s.update(completed=all(r['status']=='TERMINAL' and r['completed'] for r in s['records']),all_physics_passed=all(r.get('physics_passed',False) for r in s['records']),all_spatial_passed=all(r.get('passed',False) for r in s['records']))
 except BaseException as e:s['error']=repr(e)
 finally:
  signal.alarm(0)
  for child,row,t in active.values():row.update(status='CONTROLLER_CLEANUP',exit_code=terminate_owned(child))
  for f in logs:f.close()
  for x,h in previous.items():signal.signal(x,h)
  s.update(terminal=True,seconds=time.monotonic()-start,no_observations_drawn=True);write(OP/'execution.json',s)
  rows=[]
  for row in s['records']:
   if row.get('source_result'):
    raw=json.loads((ROOT/row['source_result']['path']).read_text());rows.append({**row,'meshes':raw.get('meshes',[])})
  write(OP/'completed-summary.json',dict(schema='C39-SIDE10-static-completed-v01',completed=s['completed'],all10_physics_and_q63_passed=s.get('all_physics_passed',False) and s.get('all_spatial_passed',False),rows=rows,missing_case_ids=[r['case']['id'] for r in s['records'] if not r.get('completed')],observations_noise_training_or_population_roles_drawn=False,raw_arrays_loaded=False))
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','launch']);p.add_argument('--protocol');p.add_argument('--sha');p.add_argument('--case');a=p.parse_args()
 if a.action=='prepare':r=prepare();print(json.dumps(r))
 elif a.action=='worker':
  d,path=validate(a.protocol,a.sha);r=worker(d,path,a.case);print(json.dumps(dict(completed=r['completed'],passed=r['passed'])))
 else:r=launch();print(json.dumps(dict(completed=r['completed'],seconds=r['seconds'])));raise SystemExit(0 if r['completed'] else 1)
