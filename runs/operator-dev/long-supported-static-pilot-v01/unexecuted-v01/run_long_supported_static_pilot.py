"""NEW .48m E300k supported-gravity static prerequisite; no dynamics admitted."""
import argparse,ast,json,multiprocessing,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_campaign_c38_graded_family_warp import ROOT,ENV_SITE,GATES,write,sha,record,guard_resources
from experiments.campaign_owned_gpu_budget_v02 import validate_cpu,terminate_owned
from experiments.campaign_c38_static_family import wrench
SOURCE='experiments/run_long_supported_static_pilot.py';TEST='tests/test_run_long_supported_static_pilot.py'
OP=ROOT/'runs/operator-dev/long-supported-static-pilot-v01';OUT=ROOT/'runs/gaussian-campaign-v01/long-supported-static-pilot-v01'
GPUS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b','GPU-c49707f6-24fd-38e4-3973-3543ec644ceb')
MESHES=((32,16,16),(40,20,20))

def restore_worker_signal_mask():
    watched={signal.SIGTERM,signal.SIGINT,signal.SIGALRM};before=signal.pthread_sigmask(signal.SIG_BLOCK,set());signal.pthread_sigmask(signal.SIG_UNBLOCK,watched);after=signal.pthread_sigmask(signal.SIG_BLOCK,set())
    if watched&after:raise RuntimeError('Owned child signal mask was not restored')
    return dict(before=sorted(map(int,before)),after=sorted(map(int,after)),required_unblocked=sorted(map(int,watched)))

def cases():return [dict(id='long_gravity__'+s,young_Pa=300000.,support=s,action_id='gravity',world_surface_force_N=[0.,0.,0.],size_m=[.48,.08,.08]) for s in ['both_ends','separated_bottom_patches']]

def build_model(cells,case,*,rule):
    from experiments.campaign_long_supported_block import LongSupportedGeometry
    if case not in cases():raise ValueError('Only two prospective body-only long-domain cases')
    return LongSupportedGeometry(cells,support=case['support'],rule=rule,world_surface_force_N=(0.,0.,0.))

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
        from experiments.campaign_long_supported_block import StreamingLongSupportedP2 as StreamingSupportedP2
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
        points=np.stack(np.meshgrid(np.linspace(0,.48,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
        from experiments.campaign_long_supported_block import SparseLongStatic
        control=SparseLongStatic((4,2,2),support=case['support']);cpu_control=control.solve(force_tolerance_N=1e-10);gpu_control_solver=solver(control);gpu_control=gpu_control_solver.solve(force_tolerance=1e-10,maximum_iterations=25);control_delta=float(np.linalg.norm(cpu_control['positions']-gpu_control['positions'].detach().cpu().numpy(),axis=1).max())
        if control_delta>1e-8 or float(torch.linalg.vector_norm(gpu_control['residual']))>1e-10:raise RuntimeError('NEW domain independent CPU/GPU coarse equilibrium gate')
        state['new_domain_CPU_GPU_control']=dict(passed=True,cells=[4,2,2],allnode_position_difference_m=control_delta,CPU_original_free_R_N=float(np.linalg.norm(cpu_control['residual'][~control.mask])),GPU_original_free_R_N=float(torch.linalg.vector_norm(gpu_control['residual'])))
        del cpu_control,gpu_control,gpu_control_solver,control;torch.cuda.empty_cache()
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
        f=pending.pop()
        if f in seen:continue
        seen.add(f)
        for node in ast.walk(ast.parse((ROOT/f).read_text())):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
                name='experiments/'+node.module.split('.')[1]+'.py'
                if (ROOT/name).exists():pending.append(name)
    return sorted(seen)

def prepare():
    base=json.loads((ROOT/'runs/operator-dev/c38-graded-family-warp-v01/protocol-v01.json').read_text());p=dict(schema='NEWlong-supported-gravity-static-prerequisite-v01',status='SOURCE_ONLY_FROZEN_PENDING_ROOT_REVIEW_NO_GPU_CALLS',source_sha256={f:sha(ROOT/f) for f in sources()},cases=cases(),meshes=[list(x) for x in MESHES],output=str(OUT.relative_to(ROOT)),owned_metadata_directory=str(OP.relative_to(ROOT)),CPU_affinity_cores=None,solver=base['solver'],gates=GATES,taskset=base['taskset'],installed_warp_API_sha256=base['installed_warp_API_sha256'],resources={**base['resources'],'global_wall_cap_s':700,'per_case_wall_cap_s':600,'physical_GPU_UUID':None,'physical_GPU_index':None,'absolute_phase_bound_including_cleanup_s':703,'projected_remaining_writes_reserve_bytes':6*1024**3},jobs=[dict(case_id=c['id'],GPU_uuid=GPUS[i],CPU_affinity_cores=[2*i,2*i+1]) for i,c in enumerate(cases())],physics=dict(size_m=[.48,.08,.08],E_Pa=300000.,nu=.35,rho_kg_m3=625.,gravity_m_s2=9.81,surface_force_N=[0.,0.,0.],contact=False,damping=0.,law='Unchanged stable NeoHookean/consistentT10; new domain needs own reference',support_patch_edges_x_m=[0.,.12,.36,.48]),controls='Two NEW-domain independent sparseCPU/GPU96 equilibrium controls before four fine FROMREST fields. Matching .48m coarse preconditioner; no .24m extrapolation.',excitation_diagnostics=dict(bulk_nonrigid_coordinate_RMS_goal_m=.001,original_peak_principal_strain_goal=[.02,.08],local_fixed_edge_peak_not_converged=True,no_force_or_material_retuning=True),cost_estimate=dict(measured_C39_similar_field_seconds=[80,126],new_domain_effect='Same49k/96ktet counts but2xlength/aspect and1.92kg totalmass; CG/Newton cost uncertain',predicted_case_seconds=[220,450],new_coarse_controls=2,new_fine_equilibrium_solves=4,exact80_checks=4,parallel_GPUs=2),scope='Static fixed63 relative material queries only; actual-h conditional p>=1 indicator20um, not continuum/fullfield/stress/dynamic qualification. No trajectories or observation/noise draws. Own static gate before any moving-material pilot.')
    f=OP/'protocol.json'
    if f.exists():raise FileExistsError(f)
    OP.mkdir(parents=True,exist_ok=True)
    write(f,p);return sha(f)

def validate(digest):
    f=OP/'protocol.json';p=json.loads(f.read_text())
    if sha(f)!=digest or p['cases']!=cases() or p['source_sha256']!={n:sha(ROOT/n) for n in sources()} or p['gates']!=GATES:raise ValueError('Changed prospective static scope/source/gates')
    if p['meshes']!=[list(x) for x in MESHES] or p['solver']!=dict(Newton_cap=25,CG_cap=4000,quadrature_chunk_size=8,force_tolerance_N=1e-10):raise ValueError('Unchanged declared mesh/solver only')
    if p['output']!=str(OUT.relative_to(ROOT)) or p['jobs']!=[dict(case_id=c['id'],GPU_uuid=GPUS[i],CPU_affinity_cores=[2*i,2*i+1]) for i,c in enumerate(cases())]:raise ValueError('Exact prospective outputs/leases only')
    r=p['resources']
    if r['global_wall_cap_s']!=700 or r['per_case_wall_cap_s']!=600 or r['GPU_allocated_cap_bytes']!=8*1024**3 or r['RSS_cap_bytes']!=12*1024**3 or r['disk_free_reserve_bytes']!=30*1024**3 or r['projected_remaining_writes_reserve_bytes']!=6*1024**3:raise ValueError('Exact prospective resource bounds')
    if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
    if any(sha(ENV_SITE/n)!=h for n,h in p['installed_warp_API_sha256'].items()):raise ValueError('Changed installed Warp API')
    return p

def launch(digest):
    p=validate(digest)
    if (OP/'execution.json').exists() or OUT.exists():raise FileExistsError('No rerun')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    if any(g in apps for g in GPUS):raise RuntimeError('Declared GPUs must be idle')
    OUT.mkdir(parents=True);start=time.monotonic();s=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_sha256=digest,records=[]);active=[];logs=[]
    def stopped(*_):raise InterruptedError('Owned700s controller interrupt')
    previous={x:signal.signal(x,stopped) for x in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]};signal.alarm(700)
    try:
        for job in p['jobs']:
            log=(OP/(job['case_id']+'.log')).open('x');logs.append(log);oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
            try:
                child=subprocess.Popen([p['taskset']['path'],'-c',','.join(map(str,job['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_long_supported_static_pilot','worker','--sha',digest,'--case',job['case_id']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':job['GPU_uuid'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});row=dict(job=job,pid=child.pid);s['records'].append(row);active.append((child,row,time.monotonic()))
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
            write(OP/'execution.json',s)
        while active:
            for child,row,t in active.copy():
                if time.monotonic()-t>600 and child.poll() is None:row['timeout']=True;terminate_owned(child)
                if child.poll() is None:continue
                row['exit_code']=child.wait();f=OUT/row['job']['case_id']/'result.json';r=json.loads(f.read_text()) if f.exists() else {};row.update(completed=r.get('completed',False),physics_passed=r.get('physics_passed',False),passed=r.get('passed',False));row['source_result']=record(f) if f.exists() else None
                if f.exists():
                    cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(f)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
                    if cache.exists():assert cache.read_bytes()==f.read_bytes()
                    else:cache.write_bytes(f.read_bytes())
                    row['external_cache']=record(cache)
                active.remove((child,row,t));write(OP/'execution.json',s)
            if active:time.sleep(.2)
        s.update(completed=all(r['completed'] and r['exit_code']==0 for r in s['records']),all_static_q63_gates_passed=all(r['passed'] for r in s['records']))
    except BaseException as e:s['error']=repr(e)
    finally:
        signal.alarm(0)
        for child,row,t in active:row['cleanup_exit_code']=terminate_owned(child)
        for f in logs:f.close()
        for sig,old in previous.items():signal.signal(sig,old)
        s.update(terminal=True,seconds=time.monotonic()-start,no_dynamic_or_dataset_admission=True);write(OP/'execution.json',s)
    return s
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','launch','signal-control']);p.add_argument('--sha');p.add_argument('--case');a=p.parse_args()
    if a.action=='signal-control':
        import tempfile
        from experiments.run_static_p2_axial_continuation import watchdog
        r=restore_worker_signal_mask()
        with tempfile.TemporaryDirectory() as temporary:
            mark=Path(temporary)/'guard.json';watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),time.monotonic()+30,12*1024**3,mark),daemon=True);watcher.start();deadline=time.monotonic()+2
            while not mark.exists() and time.monotonic()<deadline:time.sleep(.01)
            if not mark.exists():raise RuntimeError('Actual watchdog did not initialize')
            watcher.terminate();watcher.join(timeout=2);r.update(watchdog_exit_code=watcher.exitcode,watchdog_terminated=not watcher.is_alive());print(json.dumps(r));assert not watcher.is_alive()
    elif a.action=='prepare':print(json.dumps(prepare()))
    elif a.action=='launch':r=launch(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','seconds','error','all_static_q63_gates_passed']}));raise SystemExit(0 if r['completed'] else 1)
    else:
        mask=restore_worker_signal_mask();d=validate(a.sha);job=next(x for x in d['jobs'] if x['case_id']==a.case);d={**d,'CPU_affinity_cores':job['CPU_affinity_cores'],'resources':{**d['resources'],'physical_GPU_UUID':job['GPU_uuid']}};write(OP/(a.case+'-signal-mask.json'),mask);r=worker(d,OP/'protocol.json',a.case);print(json.dumps(dict(completed=r['completed'],passed=r['passed'])))
