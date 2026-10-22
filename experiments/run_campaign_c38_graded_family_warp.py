"""Prospective remaining12 graded32/40 Warp-HVP continuation; no dataset admission.

Fresh serial case processes, REST initialization at every resolution, accepted
Newton snapshots, original exact24+80 residuals and actual-h spatial indicator.
All failed/queued cells persist; no automatic resume/retry or material retuning.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,multiprocessing,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.campaign_c38_static_family import compile_known_forces,wrench,E_BOUNDS,SUPPORTS
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry, MAPPING
from experiments.campaign_owned_gpu_budget_v02 import collect_resources,validate_budget,validate_cpu,LIMITS

def build_geometry(cells,young,support,Fz,*,rule):
    return SupportedBoundaryGradedGeometry(cells,young,support,Fz,rule=rule)

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c38_graded_family_warp.py'
ENV_SITE=ROOT/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'
MESHES=((32,16,16),(40,20,20))
GPU_UUID='GPU-571378aa-c44e-ffb8-b346-1e409f9821f9'
GATES=dict(free_residual24_N=1e-10,free_residual80_N=1e-10,force_balance_N=1e-8,actual_moment_balance_Nm=1e-8,exact_constraint_error_m=0.,minimum_sampled_J=.1,positive_whole_element=True,coarse_CPU_GPU_allnode_m=1e-8,gradient24_80_difference_N=1e-10,energy24_80_difference_J=1e-11,conditional_actual_h_spatial_indicator_m=20e-6)

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return x.item()
    return x
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n');tmp.replace(p)
def record(p):
    p=Path(p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Repository-local artifact required')
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))

REUSED_CASE_IDS=('Elo__separated_bottom_patches__down2','Elo__both_ends__down2')

def all_cases():
    from experiments.run_campaign_c38_static_prequalification import cases as original_cases
    return original_cases()

def cases():return [c for c in all_cases() if c['id'] not in REUSED_CASE_IDS]

def source_closure():
    seeds=[SOURCE,'experiments/run_campaign_c38_static_graded_pilot.py','experiments/campaign_c38_static_family.py','experiments/campaign_c38_supported_grading.py','experiments/campaign_p2_supported_preconditioner.py','experiments/campaign_p2_checkpointed.py','experiments/campaign_p2_jacobian_certificate.py','experiments/campaign_newton_stable_material.py','experiments/run_static_p2_axial_continuation.py'];seen=set();pending=list(seeds)
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise FileNotFoundError('Local source dependency missing:'+name)
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.ImportFrom):
                if node.level==1:names=[node.module.split('.')[0]] if node.module else[a.name for a in node.names]
                elif node.module and node.module.startswith('experiments.'):names=[node.module.split('.')[1]]
            for module in names:
                dep='experiments/'+module+'.py'
                if (ROOT/dep).is_file():pending.append(dep)
    return sorted(seen)

METADATA_INPUTS={
    'base_family_protocol':'runs/operator-dev/c38-static-prequalification-v01/protocol-v02.json',
    'original14_summary':'runs/operator-dev/c38-static-prequalification-v01/completed-summary.json',
    'original14_terminal':'runs/operator-dev/c38-static-prequalification-v01/terminal-proof.json',
    'coarse_identity':'runs/operator-dev/c38-coarse-identity-v01/receipt.json',
    'selected_summary':'runs/operator-dev/c38-static-graded-refinement-v03/completed-summary.json',
    'selected_terminal':'runs/operator-dev/c38-static-graded-refinement-v03/terminal-proof.json',
    'selected_cache':'runs/operator-dev/c38-static-graded-refinement-v03/completed-case-cache-receipt.json',
    'HVP_protocol':'runs/operator-dev/p2-torch-warp-hvp-benchmark-v02/protocol.json',
    'HVP_result':'runs/operator-dev/p2-torch-warp-hvp-benchmark-v02/result-summary.json',
    'HVP_terminal':'runs/operator-dev/p2-torch-warp-hvp-benchmark-v02/terminal-verification.json',
}


def physical_key(c):return (c['id'],c['young_Pa'],c['support'],c['Fz_N'])

def references():
    metadata={k:json.loads((ROOT/n).read_text()) for k,n in METADATA_INPUTS.items()}
    original=metadata['original14_summary'];identity=metadata['coarse_identity'];selected=metadata['selected_summary'];terminal=metadata['selected_terminal'];benchmark=metadata['HVP_result']
    if original.get('completed') is not True or original.get('cases_completed')!=14 or identity.get('passed') is not True or len(identity.get('rows',[]))!=14 or not all(r['passed'] for r in identity['rows']):raise ValueError('All14 original controls and exact CPU geometry/force/mass proof required')
    if selected.get('completed') is not True or selected.get('selected_two_case_finest_displacement_indicators_passed') is not True or terminal.get('all_owned_PIDs_absent') is not True or terminal.get('all_source_pins_unchanged') is not True:raise ValueError('Both completed selected cases required')
    if benchmark.get('completed') is not True or benchmark.get('passed') is not True:raise ValueError('Completed paired GPU HVP acceptance required')
    hvp_sources=metadata['HVP_protocol']['source_sha256'];required={'experiments/campaign_p2_torch_warp_hvp.py','experiments/campaign_p2_warp_bulk_hvp.py'}
    if not required.issubset(hvp_sources):raise ValueError('Exact accepted HVP source pins required')
    for name,digest in hvp_sources.items():
        if sha(ROOT/name)!=digest:raise ValueError('Accepted HVP dependency changed')
    hvp_terminal=metadata['HVP_terminal']
    if hvp_terminal.get('all_source_pins_match') is not True or not hvp_terminal.get('owned_PIDs_absent') or not all(hvp_terminal['owned_PIDs_absent'].values()) or hvp_terminal.get('protocol_sha256')!=sha(ROOT/METADATA_INPUTS['HVP_protocol']):raise ValueError('Completed reaped HVP proof required')
    expected={c['id']:c for c in all_cases()};identity_cases={r['case']['id']:r for r in identity['rows']}
    if set(identity_cases)!=set(expected):raise ValueError('Exact all14 coarse controls required')
    reused=[];case_refs=[];cache={r['original_path']:r for r in metadata['selected_cache']['completed_cases_cached']}
    for r in selected['rows']:
        case=r['case'];cell=case['id']
        if cell not in REUSED_CASE_IDS or physical_key(case)!=physical_key(expected[cell]):raise ValueError('Exact selected physical cases required')
        ref=r['case_result'];cached=cache.get(ref['path'])
        if sha(ROOT/ref['path'])!=ref['sha256'] or cached is None or cached['sha256']!=ref['sha256'] or sha(ROOT/cached['cache_path'])!=ref['sha256']:raise ValueError('External exact selected case bytes required')
        case_refs.append(ref)
        for mesh in r['mesh_records']:
            if mesh['cells'][0] not in (32,40):continue
            capture=mesh['capture']
            if sha(ROOT/capture['path'])!=capture['sha256']:raise ValueError('Exact reused fine capture changed')
            reused.append(dict(case_id=cell,physical_case=expected[cell],cells=mesh['cells'],capture=capture,source_case=ref,source_cache={'path':cached['cache_path'],'sha256':cached['sha256']},reused_exact_field=True,not_recomputed=True))
    if len(reused)!=4 or {(r['case_id'],r['cells'][0]) for r in reused}!={(c,n) for c in REUSED_CASE_IDS for n in (32,40)}:raise ValueError('Exactly4 selected32/40 fields required')
    # Preserve the prior incident/revalidation lineage without reading its lost
    # full report. Selected completed-summary metadata explicitly records it.
    for r in selected['metadata_lineage'].values():
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Prior incident/revalidation metadata changed')
    return metadata,reused,case_refs


def prepare(protocol_path,output):
    destination=Path(protocol_path).resolve();out=Path(output).resolve()
    if destination.exists() or out.exists() or not out.is_relative_to(ROOT):raise ValueError('Fresh exclusive protocol/output required')
    metadata,reused,case_refs=references();p=metadata['base_family_protocol'].copy()
    p.update(schema='C38-remaining12-graded-WarpHVP-v01',status='FROZEN_BEFORE_NEW_PHYSICS',output=str(out.relative_to(ROOT)),source_sha256={n:sha(ROOT/n) for n in source_closure()},all_original_cases=all_cases(),cases=cases(),meshes=MESHES,mapping=MAPPING,metadata_inputs={k:record(ROOT/n) for k,n in METADATA_INPUTS.items()},reused_fine_fields=reused,reused_completed_case_results=case_refs,first_full_nonlinear_case_id=cases()[0]['id'],first_case_gate='First case is part of declared12, NEW32/40 FROMREST and allphysical/independentlinear gates before later launches; no extra pilot/rerun',fine_field_counts=dict(new=24,reused=4,total=28),new_coarse_solve_count=0,coarse_control_reuse='Original14 sparseCPU/GPU controls plus all14 exact coarse geometry/forces/fullM identity; auxiliary storage differences explicit; no whole-object identity claim',backend='Exact frozen TorchWarpBulkHVPMixin + original CheckpointedStaticSolveMixin/StreamingSupportedP2; no physical operator/tolerance change',independent_linear_gate='Each returned CG direction checked once with ORIGINAL Torch HVP; boundmax(2*requestedCGtol,1e-11N)+64eps*max(normRHS,normApplied,1e-30); early inexactNewton NOT forced to finalstatic1e-10N',initialization='FROM REST for every NEWcase/32/40; no24 or coarse solve; no warmstarts',full14_family_or_dataset_admitted=False,original_prior_full_case_JSON_recovered=False,peak_warning='Selected bottom peak61.27% and end18.49%; growing fixed-edge peak not converged stress/moderate local strain. All14 finite displacement screen only.')
    p['CPU_affinity_cores']=metadata['HVP_protocol']['CPU_affinity_cores']
    p['taskset']={'path':'/usr/bin/taskset','sha256':sha('/usr/bin/taskset')}
    p['owned_metadata_directory']=str(destination.parent.relative_to(ROOT))
    p['resource_sampling']='Full resident+Torch/RSS/disk at startup, each energy evaluation, every16HVPs and before/after checkpoint/capture writes; ownprocess only'
    p['installed_warp_API_sha256']=metadata['HVP_protocol']['installed_warp_API_sha256']
    p['runtime_bridge']='Append same isolated existing Warp prefix as accepted benchmark in fresh worker ONLY; installed packages/venv unchanged'
    p['resources']={**p['resources'],'global_wall_cap_s':10800,'per_case_wall_cap_s':900,'absolute_phase_bound_including_cleanup_s':10803,'projected_remaining_writes_reserve_bytes':3*1024**3}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:json.dump(safe(p),stream,indent=2,allow_nan=False);stream.write('\n')
    return sha(destination)


def validate(protocol_path,digest):
    if sha(protocol_path)!=digest:raise ValueError('Frozen remaining12 protocol changed')
    p=json.loads(Path(protocol_path).read_text())
    if p.get('schema')!='C38-remaining12-graded-WarpHVP-v01' or p.get('cases')!=cases() or p.get('all_original_cases')!=all_cases() or p.get('meshes')!=[list(v) for v in MESHES] or p.get('gates')!=GATES or p.get('mapping')!=MAPPING or set(p.get('source_sha256',{}))!=set(source_closure()):raise ValueError('Exact12-case24-new-fields design required')
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed')
    if p.get('metadata_inputs')!={k:record(ROOT/n) for k,n in METADATA_INPUTS.items()}:raise ValueError('Control/reuse metadata changed')
    for name,digest in p['installed_warp_API_sha256'].items():
        if sha(ENV_SITE/name)!=digest:raise ValueError('Installed accepted Warp API changed')
    if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Pinned taskset changed')
    _,reused,case_refs=references()
    if p.get('reused_fine_fields')!=reused or p.get('reused_completed_case_results')!=case_refs or p.get('first_full_nonlinear_case_id')!=cases()[0]['id']:raise ValueError('Exact reused fields/first case required')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='2':raise ValueError('Explicit CPU2 required')
    return p

def guard_resources(p,*,gpu=False):
    limits=p['resources'];reserve=limits['projected_remaining_writes_reserve_bytes']
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limits['RSS_cap_bytes']:raise MemoryError('Frozen12GiB processRSS cap')
    if shutil.disk_usage(ROOT).free-reserve<limits['disk_free_reserve_bytes']:raise OSError('Frozen30GiB free +3GiB projectedwrites reserve')
    if gpu:
        import torch
        validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads())
        roots=[ROOT/p['output'],ROOT/p['owned_metadata_directory']]
        metrics=collect_resources(gpu_uuid=p['resources']['physical_GPU_UUID'],pid=os.getpid(),roots=roots,torch=torch)
        validate_budget(metrics,reserve_write_bytes=reserve,limits=LIMITS)
        return metrics

def worker(p,protocol,case_id):
    rows=[r for r in p['cases'] if r['id']==case_id]
    if len(rows)!=1:raise ValueError('Exactly declared case id required before output/physics')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=p['resources']['physical_GPU_UUID']:raise ValueError('Exactly physicalGPU2UUID visibility required')
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
        if set(own)!={p['resources']['physical_GPU_UUID']}:raise ValueError('OwncomputePID must occupy only physicalGPU2')
        state['startup_resources']=guard_resources(p,gpu=True);save()
        class Solver(CheckpointedStaticSolveMixin,IndependentlyCheckedWarpCGMixin,StreamingSupportedP2):
            def evaluate(self,u):
                value=super().evaluate(u);guard_resources(p,gpu=True);return value
            def tangent(self,c,v):
                value=super().tangent(c,v);self.budget_HVP_calls=getattr(self,'budget_HVP_calls',0)+1
                if self.budget_HVP_calls%16==0:guard_resources(p,gpu=True)
                return value
        def geometry(cells,rule):return build_geometry(tuple(cells),case['young_Pa'],case['support'],case['Fz_N'],rule=rule)
        def solver(model):return Solver(model,'cuda:0',p['solver']['CG_cap'],p['solver']['quadrature_chunk_size'],support_kind=case['support'])
        def solve(model,name):
            t=solver(model);identity=checkpoint_identity(model,t,source_sha256=p['source_sha256'],protocol_sha256=sha(protocol));checkpoint=AcceptedStaticCheckpoint(directory/name/'accepted',identity)
            def guarded_checkpoint(value):
                guard_resources(p,gpu=True);checkpoint(value);guard_resources(p,gpu=True)
            answer=t.solve(force_tolerance=p['solver']['force_tolerance_N'],maximum_iterations=p['solver']['Newton_cap'],on_accepted=guarded_checkpoint);return t,answer
        points=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
        fields=[];state['new_resolution_solves']=0;state['historical_coarse_controls_reused']=True;save()
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

def stop_owned(child):
    if child.poll() is not None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def run(p,protocol):
    if set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('Controller must be pinned to declaredCPU2')
    out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ledger=dict(completed=False,passed=False,protocol=record(protocol),pid=os.getpid(),records=[],queued_case_ids=[r['id'] for r in p['cases']],max_concurrent_cases=1,retries=0,reused_fine_fields=p['reused_fine_fields'],all_original_case_ids=[r['id'] for r in p['all_original_cases']],first_case_gate_passed=False,domain_or_image_cohort_admitted=False,raw_local=True);write(out/'execution.json',ledger);active=None
    def stopped(*_):raise InterruptedError('Ownedcontroller stop requested; drain activeworker, retainqueue')
    oldterm=signal.signal(signal.SIGTERM,stopped);oldint=signal.signal(signal.SIGINT,stopped)
    try:
        for case in p['cases']:
            remaining=p['resources']['global_wall_cap_s']-(time.monotonic()-start)
            if remaining<=0:ledger['failure']='Globalhardcap before nextcase';break
            guard_resources(p);log=out/(case['id']+'.log');command=[p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),sys.executable,'-u','-m','experiments.run_campaign_c38_graded_family_warp','worker','--protocol',str(Path(protocol).resolve()),'--protocol-sha256',sha(protocol),'--case-id',case['id']]
            with log.open('x') as stream:
                active=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':p['resources']['physical_GPU_UUID'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r=dict(case=case,pid=active.pid,command=command,terminal=False);ledger['records'].append(r);ledger['queued_case_ids'].remove(case['id']);write(out/'execution.json',ledger)
                try:code=active.wait(timeout=min(p['resources']['per_case_wall_cap_s'],remaining))
                except BaseException as error:
                    stop_owned(active);code=active.returncode;r['controller_exception']=dict(type=type(error).__name__,message=str(error))
                r.update(terminal=True,exit_code=code);active=None
            s={}
            result=out/case['id']/'result.json'
            if result.exists():
                r['result']=record(result);s=json.loads(result.read_text());r.update(completed=s.get('completed',False),passed=s.get('passed',False))
            if case['id']==p['first_full_nonlinear_case_id']:
                ledger['first_case_gate_passed']=bool(code==0 and r.get('completed') and s.get('physics_passed') and all(m.get('all_independent_original_Torch_linear_checks_passed') for m in s.get('meshes',[])))
                if not ledger['first_case_gate_passed']:ledger['failure']='Initial declared case physical/independentlinear gate failed'
            write(out/'execution.json',ledger)
            if code!=0 or r.get('completed') is not True or (case['id']==p['first_full_nonlinear_case_id'] and not ledger['first_case_gate_passed']):ledger['failure']='Stopnewlaunches after technical/physicalfailure; remainingqueue preserved';break
    except BaseException as error:
        ledger['controller_failure']=dict(type=type(error).__name__,message=str(error))
        if active is not None:
            stop_owned(active)
            if ledger['records'] and ledger['records'][-1]['pid']==active.pid:ledger['records'][-1].update(terminal=True,exit_code=active.returncode)
    finally:
        ledger.update(completed=len(ledger['records'])==len(p['cases']) and all(r.get('completed') for r in ledger['records']),passed=len(ledger['records'])==len(p['cases']) and all(r.get('passed') for r in ledger['records']),seconds=time.monotonic()-start,owned_worker_live_at_exit=False);write(out/'execution.json',ledger);signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGINT,oldint)
    return ledger

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--case-id');args=a.parse_args()
    if args.mode=='prepare':print(prepare(args.protocol,args.output));return
    p=validate(args.protocol,args.protocol_sha256)
    if args.mode=='worker':s=worker(p,args.protocol,args.case_id)
    else:s=run(p,args.protocol)
    raise SystemExit(0 if s['completed'] else 1)
if __name__=='__main__':main()
