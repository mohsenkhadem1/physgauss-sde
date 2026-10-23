"""NEW one-step midpoint algebra/cost control; not a physical trajectory."""
import argparse,ast,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_campaign_c38_graded_family_warp import ROOT,ENV_SITE,write,sha,record,guard_resources
from experiments.campaign_owned_gpu_budget_v02 import validate_cpu,run_owned_command
from experiments.run_long_supported_static_pilot import restore_worker_signal_mask
SOURCE='experiments/run_long_midpoint_capacity.py'
OP=ROOT/'runs/operator-dev/long-midpoint-capacity-v01';OUT=ROOT/'runs/gaussian-campaign-v01/long-midpoint-capacity-v01'
STATIC=ROOT/'runs/operator-dev/long-supported-static-pilot-v01'
GPU='GPU-5b432ce5-a24a-68ba-150f-682265260a9b'


def sources():
    seen=set();pending=[SOURCE,'tests/test_run_long_midpoint_capacity.py','tests/test_campaign_midpoint_capacity_controls.py']
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
    summary=json.loads((STATIC/'compact-summary.json').read_text());base=json.loads((STATIC/'protocol-v02.json').read_text());assert summary['completed'] and summary['all_static_displacement_query_gates_passed']
    both=next(r for r in summary['rows'] if r['case']['support']=='both_ends');OP.mkdir(parents=True,exist_ok=True)
    p=dict(schema='NEWlong-midpoint-capacity-only-v01',status='FROZEN_SOURCE_PENDING_ROOT_REVIEW_NO_PHYSICS',source_sha256={f:sha(ROOT/f) for f in sources()},static_prerequisites=[record(STATIC/n) for n in ['compact-summary.json','terminal-proof.json','protocol-v02.json']],both_BCs_retained_as_prerequisites=[r['case'] for r in summary['rows']],selected_support='both_ends',selection_reason='Prospective visible1.374mm static bulk; original maximum-local-strain2to8percent goal failed9.74percent; no model admission',input_fields=[dict(cells=r['cells'],capture=r['capture']) for r in both['meshes']],physics={**base['physics'],'surface_force_N':[0.,0.,-2.],'initial_equilibrium_surface_force_N':[0.,0.,0.]},benchmark=dict(h_s=.02,current_algebra_clock_s=.49,midpoint_algebra_clock_s=.5,endpoint_algebra_clock_s=.51,initial_velocity='EXACT_ZERO',static_equilibrium_at_nominal_clock_is_not_physical_prehistory=True,explicit_public_surface_force_N=[0.,0.,-2.],C3_pulse_duration_s=1.,C3_pulse_midpoint_value=1.,actual_force='CONSTANT_KNOWN_BODY_GRAVITY_PLUS_EXPLICIT_DOWN2N_AT_THIS_SINGLE_MIDPOINT',damping=0.,contact=False,filter=False),solver=base['solver'],CPU_affinity_cores=[0,1],resources={**base['resources'],'physical_GPU_UUID':GPU,'physical_GPU_index':0,'max_concurrent_cases':1,'projected_remaining_writes_reserve_bytes':3*1024**3},taskset=base['taskset'],installed_warp_API_sha256=base['installed_warp_API_sha256'],output=str(OUT.relative_to(ROOT)),owned_metadata_directory=str(OP.relative_to(ROOT)),gates=dict(original_objective_free_R_N=1e-10,physical_midpoint_free_R_N=1e-9,minimum_sampled_J=.1,whole_midpoint_endpoint_J_positive=True,exact_constraints=True,gradient24_80_difference_N=1e-10,energy24_80_difference_J=1e-11,midpoint_force_balance_N=1e-8,midpoint_moment_balance_Nm=1e-8,energy_work_identity_J=1e-10,coarse_CPU_GPU_allnode_q_m=1e-8,coarse_CPU_GPU_allnode_v_m_s=1e-6),scope='Algebra/cost controls only: one coarse CPU static equilibrium, one coarse CPU/GPU midpoint pair, one midpoint each on hashbound32/40 equilibria. No temporal/spatial convergence, no trajectory, no forecast, no observation/inference/dataset admission.',cost_estimate='Coarse CPU direct96 expected<2s; fine one-step Newton/CG mass-conditioned expected10to90s each plus exact80/certificate. Hard600s worker/700s controller; preserve all failures; no retries.')
    f=OP/'protocol.json'
    if f.exists():raise FileExistsError(f)
    write(f,p);return sha(f)


def validate(digest):
    f=OP/'protocol.json';p=json.loads(f.read_text())
    if sha(f)!=digest or p['source_sha256']!={n:sha(ROOT/n) for n in sources()}:raise ValueError('Exact frozen prospective sources/protocol required')
    for r in p['static_prerequisites']+[x['capture'] for x in p['input_fields']]:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed hash-bound static prerequisite/field')
    if [r['cells'] for r in p['input_fields']]!=[[32,16,16],[40,20,20]] or p['selected_support']!='both_ends' or p['benchmark']['h_s']!=.02 or p['benchmark']['explicit_public_surface_force_N']!=[0.,0.,-2.]:raise ValueError('Exact single algebraic benchmark scope')
    if sha(p['taskset']['path'])!=p['taskset']['sha256'] or any(sha(ENV_SITE/n)!=h for n,h in p['installed_warp_API_sha256'].items()):raise ValueError('Changed installed runtime')
    return p


def worker(p,protocol):
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,pid=os.getpid(),protocol=record(protocol),records=[],not_a_physical_trajectory=True,no_temporal_or_spatial_qualification=True)
    def save():state['seconds']=time.monotonic()-start;write(OUT/'result.json',state)
    def stopped(*_):raise TimeoutError('Owned600s capacity/RSS guard')
    signal.signal(signal.SIGTERM,stopped)
    import multiprocessing
    from experiments.run_static_p2_axial_continuation import watchdog
    watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+600,12*1024**3,OUT/'resource-guard.json'),daemon=True);watcher.start()
    try:
        import torch
        sys.path.append(str(ENV_SITE));import warp as wp
        wp.config.kernel_cache_dir=str(OUT/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2)
        validate_cpu([0,1],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads())
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or torch.cuda.device_count()!=1:raise ValueError('Exactly leased GPU0 visibility')
        torch.cuda.set_per_process_memory_fraction(min(1.,8*1024**3/torch.cuda.get_device_properties(0).total_memory),0);torch.zeros(1,device='cuda:0').sum().item();state['startup_resources']=guard_resources(p,gpu=True);save()
        from experiments.campaign_long_supported_block import LongSupportedGeometry,SparseLongStatic,StreamingLongSupportedP2,compile_forces
        from experiments.campaign_c38_verified_warp_solver import IndependentlyCheckedWarpCGMixin
        from experiments.campaign_p2_smooth_midpoint import midpoint_step,smooth_bump
        from experiments.campaign_midpoint_capacity_controls import cpu_midpoint
        from experiments.campaign_p2_transient import P2State
        from experiments.campaign_p2_consistent_transient import scalar_mass
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        from experiments.campaign_c38_static_family import wrench
        class Solver(IndependentlyCheckedWarpCGMixin,StreamingLongSupportedP2):
            def evaluate(self,u):
                value=super().evaluate(u);guard_resources(p,gpu=True);return value
            def tangent(self,c,v):
                value=super().tangent(c,v);self.calls=getattr(self,'calls',0)+1
                if self.calls%16==0:guard_resources(p,gpu=True)
                return value
        def model(cells,rule='keast24'):return LongSupportedGeometry(tuple(cells),support='both_ends',rule=rule)
        def solver(m):return Solver(m,'cuda:0',4000,8,support_kind='both_ends')
        h=p['benchmark']['h_s'];surface=np.asarray(p['benchmark']['explicit_public_surface_force_N']);assert smooth_bump(.5,1.)==1.
        coarse=SparseLongStatic((4,2,2),support='both_ends');equilibrium=coarse.solve(force_tolerance_N=1e-10);u0=equilibrium['displacement_local_m'];v0=np.zeros_like(u0);coarseF=compile_forces(coarse,surface)['total_N'];mass=scalar_mass(coarse)
        def elastic(u,tangent=False):return coarse.elastic(coarse.rest+u,tangent)
        CPU=cpu_midpoint(elastic,mass,coarse.mask,u0,v0,h,coarseF,coarse.young*coarse.volumes.sum());op=solver(coarse);gpu,ledger=midpoint_step(op,P2State(torch.tensor(u0,device='cuda:0'),torch.zeros_like(op.rest),.49),h,torch.tensor(coarseF,device='cuda:0'));dq=float(np.linalg.norm(CPU['u']-gpu.u.cpu().numpy(),axis=1).max());dv=float(np.linalg.norm(CPU['v']-gpu.v.cpu().numpy(),axis=1).max())
        coarse80=model((4,2,2),'exact80');op80=solver(coarse80);tu0=torch.tensor(u0,device='cuda:0');tum=(tu0+gpu.u)/2;cu=[op80.evaluate(z) for z in [tu0,tum,gpu.u]];cm24=op.evaluate(tum);mass_acc=op.mass(gpu.v/h);coarse_r80=mass_acc+cu[1]['gradient']-torch.tensor(coarseF,device='cuda:0');coarse_R80=float(torch.linalg.vector_norm(coarse_r80[~op.mask]));coarse_gradient_delta=float(torch.linalg.vector_norm(cu[1]['gradient']-cm24['gradient']));coarse_energy_delta=max(abs(float(c['energy'])-float(op.evaluate(z)['energy'])) for c,z in zip(cu,[tu0,tum,gpu.u]));coarse_certs=[whole_p2_jacobian_bounds(coarse80,coarse.rest+z.cpu().numpy()) for z in [tum,gpu.u]];coarse_balance=torch.tensor(coarseF,device='cuda:0')+coarse_r80.masked_fill(~op.mask,0.)-mass_acc;coarse_force_defect=float(torch.linalg.vector_norm(coarse_balance.sum(0)));coarse_moment_defect=float(torch.linalg.vector_norm(torch.linalg.cross(op.rest+tum,coarse_balance,dim=1).sum(0)))
        coarse_pass=dq<=1e-8 and dv<=1e-6 and coarse_R80<=1e-9 and coarse_gradient_delta<=1e-10 and coarse_energy_delta<=1e-11 and coarse_force_defect<=1e-8 and coarse_moment_defect<=1e-8 and abs(ledger['ledger_identity_defect_J'])<=1e-10 and abs(CPU['ledger']['ledger_identity_defect_J'])<=1e-10 and all(c['all_elements_positive'] for c in coarse_certs)
        state['coarse_control']=dict(CPU_ledger=CPU['ledger'],GPU_ledger=ledger,allnode_q_difference_m=dq,allnode_v_difference_m_s=dv,original_midpoint_free_R80_N=coarse_R80,gradient24_80_difference_N=coarse_gradient_delta,maximum_energy24_80_difference_J=coarse_energy_delta,midpoint_force_balance_N=coarse_force_defect,midpoint_actual_moment_balance_Nm=coarse_moment_defect,midpoint_endpoint_whole_P2_J=[dict(all_elements_positive=c['all_elements_positive'],minimum_lower_bound=c['minimum_lower_bound']) for c in coarse_certs],passed=bool(coarse_pass))
        save()
        if not state['coarse_control']['passed']:raise RuntimeError('Independent coarse midpoint parity failed')
        del op,gpu,coarse,CPU,coarse80,op80,tu0,tum,cu,cm24,mass_acc,coarse_r80,coarse_balance;torch.cuda.empty_cache()
        points=np.stack(np.meshgrid(np.linspace(0,.48,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3)
        for source in p['input_fields']:
            begin=time.monotonic();cells=source['cells'];m=model(cells);initial=ROOT/source['capture']['path'];assert sha(initial)==source['capture']['sha256']
            with np.load(initial,allow_pickle=False) as f:
                for name,key in [('rest','rest_m'),('tets','tets'),('elements','elements'),('faces','faces'),('mask','constraint_mask'),('load','known_total_force_N')]:
                    if not np.array_equal(getattr(m,name),f[key]):raise ValueError('Initial static field belongs to exact geometry/constraint/gravity')
                u0=f['displacement_m'].copy()
            force=compile_forces(m,surface);op=solver(m);initial_tensor=torch.tensor(u0,device='cuda:0');state0=P2State(initial_tensor,torch.zeros_like(initial_tensor),.49);F=torch.tensor(force['total_N'],device='cuda:0');answer,ledger=midpoint_step(op,state0,h,F);torch.cuda.synchronize();solve_seconds=time.monotonic()-begin
            u1=answer.u.detach().clone();v1=answer.v.detach().clone();um=(initial_tensor+u1)/2;a=v1/h;Ma=op.mass(a);qmid=m.rest+um.cpu().numpy();qend=m.rest+u1.cpu().numpy();K1=.5*float((v1*op.mass(v1)).sum());linear_checks=op.independent_linear_checks
            c24=[op.evaluate(z) for z in [initial_tensor,um,u1]];g24=c24[1]['gradient'].detach().clone();E24=[float(c['energy']) for c in c24];del c24,op;torch.cuda.empty_cache();m80=model(cells,'exact80');op80=solver(m80);c80=[op80.evaluate(z) for z in [initial_tensor,um,u1]];g80=c80[1]['gradient'];r80=Ma+g80-F;R80=float(torch.linalg.vector_norm(r80[~op80.mask]));gradient_delta=float(torch.linalg.vector_norm(g80-g24));energy_delta=max(abs(float(c['energy'])-e) for c,e in zip(c80,E24));reaction=r80.masked_fill(~op80.mask,0.);balance=F+reaction-Ma;force_defect=float(torch.linalg.vector_norm(balance.sum(0)));moment_defect=float(torch.linalg.vector_norm(torch.linalg.cross(op80.rest+um,balance,dim=1).sum(0)))
            du=u1-initial_tensor;work=float((F*du).sum());dU=float(c80[2]['energy']-c80[0]['energy']);potential_defect=dU-float((g80*du).sum());residual_work=float((r80*du).sum());identity=dU+K1-work-potential_defect-residual_work
            certificates=[whole_p2_jacobian_bounds(m80,q) for q in [qmid,qend]];certs=[dict(all_elements_positive=c['all_elements_positive'],minimum_lower_bound=c['minimum_lower_bound']) for c in certificates]
            passed=R80<=1e-9 and gradient_delta<=1e-10 and energy_delta<=1e-11 and force_defect<=1e-8 and moment_defect<=1e-8 and abs(identity)<=1e-10 and abs(ledger['ledger_identity_defect_J'])<=1e-10 and all(c['all_elements_positive'] for c in certs) and all(c['passed'] for c in linear_checks)
            target=OUT/('mesh'+str(cells[0])+'.npz');np.savez_compressed(target,rest_m=m.rest,elements=m.elements,tets=m.tets,midpoint_positions_m=qmid,positions_m=qend,velocity_m_s=v1.cpu().numpy(),displacement_m=u1.cpu().numpy(),material_probes_m=points,probe_positions_m=m.query(qend,points),probe_velocities_m_s=m.query(v1.cpu().numpy(),points),known_force_N=force['total_N'])
            row=dict(cells=cells,completed=True,passed=bool(passed),input_field=source['capture'],capture=record(target),midpoint_ledger24=ledger,original_midpoint_free_R80_N=R80,gradient24_80_difference_N=gradient_delta,maximum_energy24_80_difference_J=energy_delta,midpoint_force_balance_N=force_defect,midpoint_actual_moment_balance_Nm=moment_defect,energy_work_ledger80=dict(work_J=work,potential_defect_J=potential_defect,stationarity_work_J=residual_work,identity_defect_J=identity),midpoint_endpoint_whole_P2_J=certs,external_wrench_start_mid_end=[dict(zip(['resultant_N','moment_Nm'],wrench(q,force['total_N']))) for q in [m.rest+u0,qmid,qend]],all_independent_CG_checks_passed=all(c['passed'] for c in linear_checks),linear_records=linear_checks,solve_and_initialization_seconds=solve_seconds,total_check_seconds=time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());state['records'].append(row);save()
            if not passed:raise RuntimeError('Original midpoint physical/quadrature/certificate/work gates failed')
            del m,m80,op80,c80,u1,v1,um,Ma,initial_tensor,F,answer;torch.cuda.empty_cache()
        state.update(completed=True,passed=True);save()
    except BaseException as e:state['failure']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc());save();raise
    finally:save();watcher.terminate();watcher.join(timeout=2)
    return state


def launch(digest):
    p=validate(digest)
    if OUT.exists() or (OP/'execution.json').exists():raise FileExistsError('No retry')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    if GPU in apps:raise RuntimeError('GPU0 must be idle')
    with (OP/'worker.log').open('x') as log:
        def child(pid):write(OP/'execution.json',dict(terminal=False,worker_pid=pid,controller_pid=os.getpid(),protocol_sha256=digest))
        result=run_owned_command([p['taskset']['path'],'-c','0,1',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_long_midpoint_capacity','worker','--sha',digest],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=700,on_child=child)
    result.update(terminal=True,protocol=record(OP/'protocol.json'))
    f=OUT/'result.json'
    if f.exists():
        result['source_result']=record(f);cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(f)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():assert cache.read_bytes()==f.read_bytes()
        else:cache.write_bytes(f.read_bytes())
        result['external_cache']=record(cache);a=json.loads(f.read_text());result.update(completed=a['completed'],passed=a['passed'])
    write(OP/'execution.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','launch']);parser.add_argument('--sha');args=parser.parse_args()
    if args.action=='prepare':print(prepare())
    elif args.action=='worker':restore_worker_signal_mask();r=worker(validate(args.sha),OP/'protocol.json');print(json.dumps(dict(completed=r['completed'],passed=r['passed'])))
    else:r=launch(args.sha);print(json.dumps(r));raise SystemExit(0 if r.get('completed') and r.get('passed') and r['exit_code']==0 else 1)
