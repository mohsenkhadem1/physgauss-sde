"""Prospective two-case boundary-graded comparison; no dataset/image admission.

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

def build_geometry(cells,young,support,Fz,*,rule):
    return SupportedBoundaryGradedGeometry(cells,young,support,Fz,rule=rule)

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c38_static_graded_pilot.py'
MESHES=((12,6,6),(20,10,10),(24,12,12))
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

def cases():
    return [dict(id='Elo__separated_bottom_patches__down2',young_Pa=75000.,support='separated_bottom_patches',Fz_N=-2.,kind='worst_completed_uniform_corner'),
            dict(id='Elo__both_ends__down2',young_Pa=75000.,support='both_ends',Fz_N=-2.,kind='same_E_force_both_end_control')]

def source_closure():
    seeds=[SOURCE,'experiments/campaign_c38_static_family.py','experiments/campaign_c38_supported_grading.py','experiments/campaign_p2_supported_preconditioner.py','experiments/campaign_p2_checkpointed.py','experiments/campaign_p2_jacobian_certificate.py','experiments/campaign_newton_stable_material.py','experiments/run_static_p2_axial_continuation.py'];seen=set();pending=list(seeds)
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

def prepare(path,output):
    if Path(path).exists() or Path(output).exists():raise FileExistsError('Fresh protocol/output; retain every earlier candidate')
    evidence=ROOT/'runs/operator-dev/c38-static-prequalification-v01/completed-summary.json';e=json.loads(evidence.read_text())
    proof=ROOT/'runs/operator-dev/c38-static-prequalification-v01/terminal-proof.json'
    uniformproof=json.loads(proof.read_text())
    if e.get('completed') is not True or e.get('cases_completed')!=14 or e.get('spatial_cases_failed')!=7 or uniformproof.get('all_source_pins_unchanged') is not True or uniformproof.get('GPU2_released') is not True:raise ValueError('Original complete14-case matrix including7 spatial failures must be preserved')
    prior={r['case']['id']:r for r in e['rows']}
    if any(prior[c['id']]['case']['young_Pa']!=c['young_Pa'] or prior[c['id']]['case']['Fz_N']!=c['Fz_N'] or prior[c['id']]['case']['support']!=c['support'] for c in cases()):raise ValueError('Same physical cases required')
    out=Path(output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Repository-local fresh output required')
    p=dict(schema='C38-static-graded-pilot-v01',status='FROZEN_BEFORE_NEW_PHYSICS',output=str(out.relative_to(ROOT)),source_sha256={n:sha(ROOT/n) for n in source_closure()},existing_uniform_matrix=record(evidence),uniform_terminal_proof=record(proof),historical_candidate='All14 uniform results including7 failures preserved; only mesh changes, no physics or threshold retuning',mapping=MAPPING,grading='Fixed symmetric quadratic corner-axis map in xsegments[0,.06,.18,.24] and y/z surfaces; rebuild true T10 physical edge midpoints, gradients and volumes; identical geometry for every hypothesis',cases=cases(),meshes=MESHES,CPU_GPU_control_cells=[4,2,2],physics=dict(size_m=[.24,.08,.08],E_Pa=E_BOUNDS,poisson=.35,density_kg_m3=625.,gravity_vector_m_s2=[0.,0.,-9.81],surface_force_Fz_N=[-2.,0.,2.],surface_patch='Known full-y TOP reference band0.06<=x<=0.18; top_middle_down basis scaledFz/−.12, world dead force',support_kinds=SUPPORTS,body_force='FULL consistent signedT10 shape integrals, including constrained-node loads',force_independent_of_E_and_support=True,constitutive_law='Polynomial stable-NH with kappa=lambda+mu; NOT logarithmic-NH bar'),quadrature='PositiveKeast24 degree6 exact affine-reference constant-material polynomial bulk; original80+10J guards retained; original exact80 residual/energy/gradient at SAME acceptedu',initialization='FROM REST for everycase/resolution/control; NO crossmesh/case warmstart',preconditioner='Original StreamingSupportedP2, explicit declared support, exactnodeblock+coarse4x2x2 correction; no stiffness repair',solver=dict(Newton_cap=25,CG_cap=4000,quadrature_chunk_size=8,force_tolerance_N=1e-10),checkpoint='Every accepted GPU Newton displacement before nextCG; source/model/force/quadrature/protocol hashes; accepted is not converged; no automatic physicalresume',physical_probes='Same fixed7x3x3 material probes acrossmeshes/cases',spatial_indicator='max63physicalquerydelta/(actual_h_coarse/actual_h_fine−1), assumedorder>=1 engineeringindicator NOT rigorousbound',gates=GATES,resources=dict(physical_GPU_UUID=GPU_UUID,physical_GPU_index=2,CPU_threads=2,global_wall_cap_s=1200,per_case_wall_cap_s=600,owned_cleanup_grace_s=3,absolute_phase_bound_including_cleanup_s=1203,GPU_allocated_cap_bytes=8*1024**3,RSS_cap_bytes=12*1024**3,disk_free_reserve_bytes=30*1024**3,max_concurrent_cases=1),failure_policy='Stopnewcases onphysical/solver/resourcefailure; preservequeue. Complete spatialindicatorfailure is retained and otherdeclaredcasescontinue. No retries/redraw/drop',domain_or_image_cohort_admitted=False,object_roles_or_observations_drawn=False,dynamic_contact_inverse_training_qualified=False,raw_local=True)
    Path(path).parent.mkdir(parents=True,exist_ok=True);write(path,p);return sha(path)

def validate(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen family protocol changed')
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='C38-static-graded-pilot-v01' or p.get('mapping')!=MAPPING or p.get('cases')!=cases() or p.get('meshes')!=[list(v) for v in MESHES] or p.get('gates')!=GATES or set(p.get('source_sha256',{}))!=set(source_closure()):raise ValueError('Exact two-case graded source/physics design required')
    for n,d in p['source_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError('Frozen dependency changed:'+n)
    for name in ('existing_uniform_matrix','uniform_terminal_proof'):
        evidence=p[name]
        if sha(ROOT/evidence['path'])!=evidence['sha256']:raise ValueError('Preserved uniform evidence changed')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='2':raise ValueError('ExplicitCPU2 required')
    return p

def guard_resources(p,*,gpu=False):
    limits=p['resources'];rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss>limits['RSS_cap_bytes']:raise MemoryError('Frozen12GiB processRSS cap')
    if shutil.disk_usage(ROOT).free<limits['disk_free_reserve_bytes']:raise OSError('Frozen30GiB diskreserve')
    if gpu:
        import torch
        if torch.cuda.max_memory_allocated()>limits['GPU_allocated_cap_bytes']:raise MemoryError('Frozen8GiB allocatedGPU cap')

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
        from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
        from experiments.campaign_p2_checkpointed import CheckpointedStaticSolveMixin,AcceptedStaticCheckpoint,checkpoint_identity
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        from experiments.campaign_newton_stable_material import StableP2Block
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactlyone visibleCUDAdevice required')
        torch.cuda.set_per_process_memory_fraction(min(1.,p['resources']['GPU_allocated_cap_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
        torch.zeros(1,device='cuda:0').sum().item();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);own=[line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip()==str(os.getpid())];state['GPU_provenance']=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],torch_device='cuda:0',own_physical_GPU_UUIDs=own);save()
        if set(own)!={p['resources']['physical_GPU_UUID']}:raise ValueError('OwncomputePID must occupy only physicalGPU2')
        class Solver(CheckpointedStaticSolveMixin,StreamingSupportedP2):
            def evaluate(self,u):
                value=super().evaluate(u);guard_resources(p,gpu=True);return value
            def tangent(self,c,v):
                value=super().tangent(c,v);guard_resources(p,gpu=True);return value
        def geometry(cells,rule):return build_geometry(tuple(cells),case['young_Pa'],case['support'],case['Fz_N'],rule=rule)
        def solver(model):return Solver(model,'cuda:0',p['solver']['CG_cap'],p['solver']['quadrature_chunk_size'],support_kind=case['support'])
        def solve(model,name):
            t=solver(model);identity=checkpoint_identity(model,t,source_sha256=p['source_sha256'],protocol_sha256=sha(protocol));checkpoint=AcceptedStaticCheckpoint(directory/name/'accepted',identity);answer=t.solve(force_tolerance=p['solver']['force_tolerance_N'],maximum_iterations=p['solver']['Newton_cap'],on_accepted=checkpoint);return t,answer
        # Original independent sparse CPU operator with the SAMEexact80 rule,
        # geometry, material, mask and full body/surface force.
        state['stage']='independent_coarse_CPU_GPU';save();m=geometry(p['CPU_GPU_control_cells'],'exact80');cpu=StableP2Block(tuple(p['CPU_GPU_control_cells']),young=case['young_Pa'],poisson=.35,density=625.,gravity=0.,traction=0.,order=3)
        for key in ('rest','tets','elements','faces'):
            if not np.array_equal(getattr(cpu,key),getattr(m,key)):raise ValueError('Independentcoarsegeometry mismatch')
        cpu.bary=m.bary.copy();cpu.weights=m.weights.copy();cpu.N=m.N.copy();cpu.check_bary=np.concatenate([m.check_bary,m.bary]);cpu.load=m.load.copy();cpu.mask=m.mask.copy();cpu.free=np.flatnonzero(~cpu.mask.ravel());a=cpu.solve(maximum_iterations=p['solver']['Newton_cap'],force_tolerance_N=1e-10)
        mc=geometry(p['CPU_GPU_control_cells'],'keast24');tc,b=solve(mc,'coarse-control');coarse_delta=float(np.linalg.norm(b['positions'].detach().cpu().numpy()-a['positions'],axis=1).max());state['coarse_CPU_GPU']=dict(allnode_difference_m=coarse_delta,passed=coarse_delta<=GATES['coarse_CPU_GPU_allnode_m'],CPU_original_free_R_N=float(np.linalg.norm(a['residual'][~m.mask])),GPU_original_free_R_N=float(torch.linalg.vector_norm(b['residual'])),CPU_history=a['history'],CPU_uses_independent_sparse_exact80=True);save()
        if not state['coarse_CPU_GPU']['passed']:raise RuntimeError('Independentexact80CPU/KeastGPUcontrol failed')
        del cpu,tc,a,b,m,mc;torch.cuda.empty_cache();points=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3);fields=[]
        for cells in p['meshes']:
            begin=time.monotonic();name='mesh'+str(cells[0]);state['stage']=name;save();m=geometry(cells,'keast24');t,a=solve(m,name);u=a['displacement'].detach().clone();q=a['positions'].detach().cpu().numpy();Q=m.query(q,points);r24=float(torch.linalg.vector_norm(a['residual']));energy24=float(a['cache']['energy']);gradient24=a['cache']['gradient'].detach().clone();history=a['history'];iterations=a['iterations'];linear_records=t.linear_records;del a,t;torch.cuda.empty_cache()
            m80=geometry(cells,'exact80')
            for key in ('rest','tets','elements','faces','mask','load'):
                if not np.array_equal(getattr(m,key),getattr(m80,key)):raise ValueError('Sameu graded exact24/80 geometry/constraints/force mismatch')
            t80=solver(m80);c80=t80.evaluate(u);full_residual=(c80['gradient']-t80.force).detach().cpu().numpy();r80=float(np.linalg.norm(full_residual[~m80.mask]));gradient_difference=float(torch.linalg.vector_norm(c80['gradient']-gradient24));energy_difference=abs(float(c80['energy'])-energy24);J=float(c80['minimum_J']);diag=m80.diagnostics(dict(positions=q,residual=full_residual,J=np.array([J]),tolerance_N=1e-10,iterations=iterations));certificate=whole_p2_jacobian_bounds(m80,q);components=m.public_force_components;actual_wrench={key:dict(zip(('resultant_N','moment_Nm'),wrench(q,components[key]))) for key in ('body_N','surface_N','total_N')}
            passed=bool(r24<=GATES['free_residual24_N'] and r80<=GATES['free_residual80_N'] and diag['force_balance_defect_N']<=GATES['force_balance_N'] and diag['actual_moment_balance_defect_Nm']<=GATES['actual_moment_balance_Nm'] and diag['maximum_constraint_error_m']==0 and J>GATES['minimum_sampled_J'] and certificate['all_elements_positive'] and gradient_difference<=GATES['gradient24_80_difference_N'] and energy_difference<=GATES['energy24_80_difference_J'])
            target=directory/name/'capture.npz';np.savez_compressed(target,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,displacement_m=u.detach().cpu().numpy(),constraint_mask=m.mask,known_body_force_N=components['body_N'],known_surface_force_N=components['surface_N'],known_total_force_N=m.load,material_probes_m=points,probe_positions_m=Q)
            row=dict(cells=cells,grading=m.grading_metadata,completed=True,physics_passed=passed,original_free_R24_N=r24,original_free_R80_N=r80,gradient24_80_difference_N=gradient_difference,energy24_80_difference_J=energy_difference,diagnostics=diag,whole_P2_J=dict(all_elements_positive=certificate['all_elements_positive'],minimum_lower_bound=certificate['minimum_lower_bound']),mask_trace=m.mask_trace,known_force_metadata=m.force_metadata,actual_deformed_wrenches=actual_wrench,hmax_m=m.maximum_reference_tet_edge_m,capture=record(target),history=history,linear_records=linear_records,seconds=time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());state['meshes'].append(row);fields.append(Q);save()
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
    out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ledger=dict(completed=False,passed=False,protocol=record(protocol),pid=os.getpid(),records=[],queued_case_ids=[r['id'] for r in p['cases']],max_concurrent_cases=1,retries=0,domain_or_image_cohort_admitted=False,raw_local=True);write(out/'execution.json',ledger);active=None
    def stopped(*_):raise InterruptedError('Ownedcontroller stop requested; drain activeworker, retainqueue')
    oldterm=signal.signal(signal.SIGTERM,stopped);oldint=signal.signal(signal.SIGINT,stopped)
    try:
        for case in p['cases']:
            remaining=p['resources']['global_wall_cap_s']-(time.monotonic()-start)
            if remaining<=0:ledger['failure']='Globalhardcap before nextcase';break
            guard_resources(p);log=out/(case['id']+'.log');command=[sys.executable,'-u','-m','experiments.run_campaign_c38_static_graded_pilot','worker','--protocol',str(Path(protocol).resolve()),'--protocol-sha256',sha(protocol),'--case-id',case['id']]
            with log.open('x') as stream:
                active=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':p['resources']['physical_GPU_UUID'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r=dict(case=case,pid=active.pid,command=command,terminal=False);ledger['records'].append(r);ledger['queued_case_ids'].remove(case['id']);write(out/'execution.json',ledger)
                try:code=active.wait(timeout=min(p['resources']['per_case_wall_cap_s'],remaining))
                except BaseException as error:
                    stop_owned(active);code=active.returncode;r['controller_exception']=dict(type=type(error).__name__,message=str(error))
                r.update(terminal=True,exit_code=code);active=None
            result=out/case['id']/'result.json'
            if result.exists():
                r['result']=record(result);s=json.loads(result.read_text());r.update(completed=s.get('completed',False),passed=s.get('passed',False))
            write(out/'execution.json',ledger)
            if code!=0 or r.get('completed') is not True:ledger['failure']='Stopnewlaunches after technical/physicalfailure; remainingqueue preserved';break
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
