"""Prospective supplied-physics P2 rest-tangent predictions; no target scoring.

Four FROM-REST basis models (two known supports x graded32/40), eight linear
RHS solves. All fourteen declared engineering configurations are predicted from
public force definitions and supplied E/support, not image inference. No private
nonlinear capture is read, no observation/role/admission is generated.
"""
from __future__ import annotations
import argparse,json,multiprocessing,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_graded_family_warp as base
from experiments.campaign_c38_static_family import SUPPORTS

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c38_static_linear.py'
LINEAR_SOURCE='experiments/campaign_c38_static_linear.py'
BASE_PROTOCOL='runs/operator-dev/c38-graded-family-warp-v01/protocol-v01.json'
E0=100000.
MESHES=((32,16,16),(40,20,20))
QUERY_POINTS=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3)


def source_closure():return sorted(set(base.source_closure())|{SOURCE,LINEAR_SOURCE})
def jobs():return [dict(id=s+'__mesh'+str(c[0]),support=s,cells=list(c)) for s in SUPPORTS for c in MESHES]
def sha(p):return base.sha(p)
def write(p,x):base.write(p,x)
def record(p):
    path=Path(p).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Repository-local record required')
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def prepare(protocol_path,output):
    destination=Path(protocol_path).resolve();out=Path(output).resolve()
    if destination.exists() or out.exists() or not destination.is_relative_to(ROOT) or not out.is_relative_to(ROOT):
        raise ValueError('Fresh exclusive repository-local protocol/output required')
    old=json.loads((ROOT/BASE_PROTOCOL).read_text())
    sources={n:sha(ROOT/n) for n in source_closure()}
    # Only frozen metadata is inspected. Neither active case outputs nor any
    # nonlinear capture is loaded or hashed by this preparation/worker.
    p=dict(schema='C38-static-linear-supplied-physics-v01',status='FROZEN_BEFORE_BASIS_SOLVES',
           source_sha256=sources,metadata_inputs={'accepted_solver_protocol':record(ROOT/BASE_PROTOCOL)},
           output=str(out.relative_to(ROOT)),owned_metadata_directory=str(destination.parent.relative_to(ROOT)),
           jobs=jobs(),all_cases=base.all_cases(),reference_young_Pa=E0,meshes=MESHES,
           point_queries_m=QUERY_POINTS,task_kind='static_equilibrium_load_prediction',
           information_role='SUPPLIED_PHYSICS_ORACLE_A_ENGINEERING_MATRIX',
           case_rows_are_not_population_roles=True,nonlinear_target_access=False,scoring_authorized=False,
           independent_basis_residual_tolerance_N=1e-10,conditional_actual_h_query_indicator_m=20e-6,
           interpretation='Linear reference baseline; no nonlinear equilibrium, RGB-D, inference, temporal or population qualification',
           prediction_J_policy='Retain all predicted fields and explicit sampled/whole-element J failure flags; never repair/project/clamp positions',
           prediction_seal_policy='All28 predictions and capture hashes frozen before any separate nonlinear target scoring',
           source_operator='Unchanged rest tangent with frozen verified Warp CG; each response checked with original Torch tangent',
           solver=old['solver'],CPU_affinity_cores=old['CPU_affinity_cores'],taskset=old['taskset'],
           installed_warp_API_sha256=old['installed_warp_API_sha256'],resources={**old['resources'],
               'global_wall_cap_s':600,'per_case_wall_cap_s':180,'absolute_phase_bound_including_cleanup_s':603,
               'projected_remaining_writes_reserve_bytes':3*1024**3},
           cost_forecast='4 fresh models/8 basis CG solves: estimated200-450s total and50-150s/model from pairedHVP controls, not measured baseline runtime; hard600/180s retained',
           raw_local=True,retries=0,max_concurrent_workers=1)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(base.safe(p),f,indent=2,allow_nan=False);f.write('\n')
    return sha(destination)


def validate(protocol,digest):
    if sha(protocol)!=digest:raise ValueError('Frozen linear protocol changed')
    p=json.loads(Path(protocol).read_text())
    if p.get('schema')!='C38-static-linear-supplied-physics-v01' or p.get('status')!='FROZEN_BEFORE_BASIS_SOLVES' or p.get('jobs')!=jobs() or p.get('all_cases')!=base.all_cases() or p.get('meshes')!=[list(m) for m in MESHES] or p.get('reference_young_Pa')!=E0 or not np.array_equal(p.get('point_queries_m'),QUERY_POINTS):raise ValueError('Exact prospective4basis/14case design required')
    if p.get('nonlinear_target_access') is not False or p.get('scoring_authorized') is not False or p.get('independent_basis_residual_tolerance_N')!=1e-10 or p.get('conditional_actual_h_query_indicator_m')!=20e-6:raise ValueError('Unchanged residual/indicator and no-target barrier required')
    for key in ('output','owned_metadata_directory'):
        value=p.get(key)
        if not isinstance(value,str) or Path(value).is_absolute() or '..' in Path(value).parts or Path(value).as_posix()!=value:raise ValueError('Canonical repository-local output metadata required')
    expected={n:sha(ROOT/n) for n in source_closure()}
    if p.get('source_sha256')!=expected:raise ValueError('Full frozen source closure changed')
    if p.get('metadata_inputs')!={'accepted_solver_protocol':record(ROOT/BASE_PROTOCOL)}:raise ValueError('Accepted solver metadata changed')
    old=json.loads((ROOT/BASE_PROTOCOL).read_text())
    for key in ('CPU_affinity_cores','taskset','installed_warp_API_sha256','solver'):
        if p.get(key)!=old[key]:raise ValueError('Exact accepted solver/runtime setting required: '+key)
    required={**old['resources'],'global_wall_cap_s':600,'per_case_wall_cap_s':180,'absolute_phase_bound_including_cleanup_s':603,'projected_remaining_writes_reserve_bytes':3*1024**3}
    if p.get('resources')!=required:raise ValueError('Exact600s/180s resource design required')
    for n,h in p['installed_warp_API_sha256'].items():
        if sha(base.ENV_SITE/n)!=h:raise ValueError('Installed Warp API changed')
    if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Taskset changed')
    return p


def sampled_minimum_J(op,displacement,*,resource_guard=None):
    """Geometric diagnostic without the nonlinear solver's rejecting J guard.

    An inverted linear prediction remains a saved, explicitly flagged baseline
    failure; it must not be repaired or discarded by elastic-energy evaluation.
    """
    import torch
    minimum=float('inf');local=displacement[op.elements]
    for checking in (False,True):
        for grad,_ in op.chunks(checking=checking):
            J=torch.linalg.det(op.deformation(local,grad))
            if not bool(torch.isfinite(J).all()):raise FloatingPointError('Nonfinite predicted deformation Jacobian')
            minimum=min(minimum,float(J.min()))
            if resource_guard is not None:resource_guard()
    return minimum


def worker(p,protocol,job_id):
    found=[j for j in p['jobs'] if j['id']==job_id]
    if len(found)!=1:raise ValueError('Exactly declared basis job required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=p['resources']['physical_GPU_UUID']:raise ValueError('ExactlyGPU2UUID visibility required')
    job=found[0];directory=ROOT/p['output']/job_id;directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();state=dict(job=job,pid=os.getpid(),protocol=record(protocol),completed=False,passed=False,predictions=[],raw_local=True,supplied_physics_oracle_A=True,nonlinear_truth_read=False)
    def save():state['seconds']=time.monotonic()-start;write(directory/'result.json',state)
    def terminated(*_):raise TimeoutError('Owned baseline worker wall/RSS cap')
    signal.signal(signal.SIGTERM,terminated)
    from experiments.run_static_p2_axial_continuation import watchdog
    watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+180,p['resources']['RSS_cap_bytes'],directory/'resource-guard.json'),daemon=True);watcher.start();save()
    try:
        import torch
        sys.path.append(str(base.ENV_SITE));import warp as wp
        wp.config.kernel_cache_dir=str(directory/'warp-cache')
        from experiments.campaign_c38_static_linear import fit_reference_basis
        from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
        from experiments.campaign_c38_verified_warp_solver import IndependentlyCheckedWarpCGMixin
        from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactlyone visibleCUDAdevice required')
        torch.cuda.set_per_process_memory_fraction(min(1.,p['resources']['GPU_allocated_cap_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
        torch.zeros(1,device='cuda:0').sum().item()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        own=[line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
        state['GPU_provenance']=dict(own_physical_GPU_UUIDs=own,CVD=os.environ['CUDA_VISIBLE_DEVICES']);save()
        if set(own)!={p['resources']['physical_GPU_UUID']}:raise ValueError('OwnPID must use onlyGPU2')
        state['startup_resources']=base.guard_resources(p,gpu=True);save()
        class Solver(IndependentlyCheckedWarpCGMixin,StreamingSupportedP2):
            def evaluate(self,u):
                result=super().evaluate(u);base.guard_resources(p,gpu=True);return result
            def tangent(self,c,v):
                result=super().tangent(c,v);self.budget_hvp=getattr(self,'budget_hvp',0)+1
                if self.budget_hvp%16==0:base.guard_resources(p,gpu=True)
                return result
        model=SupportedBoundaryGradedGeometry(tuple(job['cells']),E0,job['support'],0.,rule='keast24')
        op=Solver(model,'cuda:0',p['solver']['CG_cap'],p['solver']['quadrature_chunk_size'],support_kind=job['support'])
        basis=fit_reference_basis(model,job['support'],operator=op,residual_tolerance_N=1e-10)
        state['basis_checks']=basis.independent_checks;state['CG_original_HVP_checks']=op.independent_linear_checks;save()
        base.guard_resources(p,gpu=True)
        path=directory/'basis.npz';np.savez_compressed(path,rest_m=model.rest,tets=model.tets,elements=model.elements,faces=model.faces,constraint_mask=model.mask,gravity_response_m=basis.gravity_response_m,unit_surface_response_m_per_N=basis.unit_surface_response_m_per_N)
        state['basis_capture']=record(path);save()
        for case in p['all_cases']:
            if case['support']!=job['support']:continue
            q=basis.positions(case['young_Pa'],case['Fz_N']);u=torch.as_tensor(q-model.rest,dtype=torch.float64,device='cuda:0')
            # These are the linear field's geometric diagnostics, not nonlinear
            # equilibrium residuals and not truth data. J failures are retained.
            minimum_J=sampled_minimum_J(op,u,resource_guard=lambda:base.guard_resources(p,gpu=True));certificate=whole_p2_jacobian_bounds(model,q)
            probes=model.query(q,QUERY_POINTS);finite=bool(np.isfinite(q).all())
            jpass=bool(finite and minimum_J>.1 and certificate['all_elements_positive'])
            base.guard_resources(p,gpu=True);target=directory/(case['id']+'.npz')
            np.savez_compressed(target,positions_m=q,probe_positions_m=probes,material_probes_m=QUERY_POINTS)
            state['predictions'].append(dict(case=case,cells=job['cells'],capture=record(target),hmax_m=model.maximum_reference_tet_edge_m,finite=finite,minimum_sampled_J=minimum_J,whole_element_J_positive=certificate['all_elements_positive'],minimum_whole_element_J_lower_bound=certificate['minimum_lower_bound'],prediction_J_passed=jpass,nonlinear_equilibrium_claim=False));save()
            del u;torch.cuda.empty_cache();base.guard_resources(p,gpu=True)
        state.update(completed=True,passed=True,all_basis_residual_checks_passed=all(r['passed'] for r in basis.independent_checks),all_prediction_J_passed=all(r['prediction_J_passed'] for r in state['predictions']),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        state.update(failure=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc());save();raise
    finally:save();watcher.terminate();watcher.join(timeout=2)
    return state


def seal_predictions(p,results):
    if len(results)!=4 or {r['job']['id'] for r in results}!={j['id'] for j in p['jobs']} or not all(r.get('completed') and r.get('passed') and len(r['predictions'])==7 for r in results):raise ValueError('All4basis jobs and28predictions required')
    rows=[]
    for case in p['all_cases']:
        pair=[]
        for nx in (32,40):
            matches=[r for result in results for r in result['predictions'] if r['case']==case and r['cells'][0]==nx]
            if len(matches)!=1:raise ValueError('Exact case/resolution prediction pair required')
            row=matches[0]
            if sha(ROOT/row['capture']['path'])!=row['capture']['sha256']:raise ValueError('Prediction capture changed')
            with np.load(ROOT/row['capture']['path'],allow_pickle=False) as z:
                if set(z.files)!={'positions_m','probe_positions_m','material_probes_m'} or not np.array_equal(z['material_probes_m'],QUERY_POINTS):raise ValueError('Prediction query/member mismatch')
                pair.append((row,z['probe_positions_m'].copy()))
        delta=float(np.linalg.norm(pair[1][1]-pair[0][1],axis=1).max());factor=1/(pair[0][0]['hmax_m']/pair[1][0]['hmax_m']-1);indicator=delta*factor
        rows.append(dict(case=case,prediction_records=[x[0] for x in pair],maximum63_query_difference_m=delta,actual_h_factor=factor,conditional_linear_spatial_indicator_m=indicator,linear_spatial_indicator_passed=indicator<=20e-6,all_prediction_J_passed=all(x[0]['prediction_J_passed'] for x in pair)))
    return dict(completed=True,all28_predictions_sealed_before_nonlinear_target_access=True,nonlinear_target_access=False,nonlinear_comparison_or_dataset_admitted=False,rows=rows,all_linear_spatial_indicators_passed=all(r['linear_spatial_indicator_passed'] for r in rows),all_prediction_J_passed=all(r['all_prediction_J_passed'] for r in rows),raw_local=True)


def run(p,protocol):
    if set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('ControllerCPU2 affinity required')
    out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();active=None
    state=dict(completed=False,protocol=record(protocol),pid=os.getpid(),records=[],queued_job_ids=[j['id'] for j in p['jobs']],retries=0,nonlinear_target_access=False,raw_local=True)
    write(out/'execution.json',state)
    def stopped(*_):raise InterruptedError('Owned baseline controller stopped')
    oldterm=signal.signal(signal.SIGTERM,stopped);oldint=signal.signal(signal.SIGINT,stopped)
    def expired(*_):raise TimeoutError('Global600s baseline wall cap')
    oldalarm=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,600)
    try:
        for job in p['jobs']:
            remaining=600-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Global600s cap before next basis model')
            base.guard_resources(p)
            command=[p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),sys.executable,'-u','-m','experiments.run_campaign_c38_static_linear','worker','--protocol',str(Path(protocol).resolve()),'--protocol-sha256',sha(protocol),'--job-id',job['id']]
            with (out/(job['id']+'.log')).open('x') as log:
                active=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':p['resources']['physical_GPU_UUID'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
                row=dict(job=job,pid=active.pid,command=command,terminal=False);state['records'].append(row);state['queued_job_ids'].remove(job['id']);write(out/'execution.json',state)
                try:code=active.wait(timeout=min(180,remaining))
                except BaseException:
                    base.stop_owned(active);row.update(terminal=True,exit_code=active.returncode);raise
                row.update(terminal=True,exit_code=code);active=None
            result=out/job['id']/'result.json'
            if result.exists():row['result']=record(result);s=json.loads(result.read_text());row.update(completed=s.get('completed',False))
            write(out/'execution.json',state)
            if code!=0 or row.get('completed') is not True:raise RuntimeError('Basis failure; queue retained, no retry')
        results=[json.loads((ROOT/r['result']['path']).read_text()) for r in state['records']]
        seal=seal_predictions(p,results);seal['protocol']=record(protocol);write(out/'predictions-seal.json',seal)
        state.update(completed=True,predictions_seal=record(out/'predictions-seal.json'))
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        if active is not None:
            base.stop_owned(active)
            if state['records']:state['records'][-1].update(terminal=True,exit_code=active.returncode)
        state.update(seconds=time.monotonic()-start,owned_workers_live_at_exit=False);write(out/'execution.json',state);signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldalarm);signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGINT,oldint)
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run','worker'));parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256');parser.add_argument('--output');parser.add_argument('--job-id');a=parser.parse_args()
    if a.mode=='prepare':print(prepare(a.protocol,a.output));return
    p=validate(a.protocol,a.protocol_sha256)
    s=worker(p,a.protocol,a.job_id) if a.mode=='worker' else run(p,a.protocol)
    raise SystemExit(0 if s['completed'] else 1)
if __name__=='__main__':main()
