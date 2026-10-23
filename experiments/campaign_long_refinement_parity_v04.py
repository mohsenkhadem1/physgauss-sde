"""New exclusive tiny actual CPU/GPU driver parity control only.

Unchanged original96tet static equilibrium and one half-stepBE/midpoint pair;
no fine states and no physical trajectory. Preserves old capacity negative/data.
"""
import json,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_campaign_c38_graded_family_warp import ROOT,ENV_SITE,write,sha,record,guard_resources
from experiments.campaign_owned_gpu_budget_v02 import validate_cpu
OUT=ROOT/'runs/gaussian-campaign-v01/long-smooth-refinement-v04/parity'
GPU='GPU-5b432ce5-a24a-68ba-150f-682265260a9b'

def worker(p,protocol):
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,pid=os.getpid(),protocol=record(protocol),driver_provenance=p['driver_provenance'],records=[],not_a_physical_trajectory=True,no_temporal_or_spatial_qualification=True)
    def save():state['seconds']=time.monotonic()-start;write(OUT/'result.json',state)
    def stopped(*_):raise TimeoutError('Owned600s capacity/RSS guard')
    signal.signal(signal.SIGTERM,stopped)
    import multiprocessing
    from experiments.run_static_p2_axial_continuation import watchdog
    watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+600,12*1024**3,OUT/'resource-guard.json'),daemon=True);watcher.start();state['watchdog_pid']=watcher.pid
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
        coarse_pass=all(c['passed'] for c in op.independent_linear_checks) and dq<=1e-8 and dv<=1e-6 and coarse_R80<=1e-9 and coarse_gradient_delta<=1e-10 and coarse_energy_delta<=1e-11 and coarse_force_defect<=1e-8 and coarse_moment_defect<=1e-8 and abs(ledger['ledger_identity_defect_J'])<=1e-10 and abs(CPU['ledger']['ledger_identity_defect_J'])<=1e-10 and all(c['all_elements_positive'] for c in coarse_certs)
        state['coarse_control']=dict(CPU_ledger=CPU['ledger'],GPU_ledger=ledger,allnode_q_difference_m=dq,allnode_v_difference_m_s=dv,original_midpoint_free_R80_N=coarse_R80,gradient24_80_difference_N=coarse_gradient_delta,maximum_energy24_80_difference_J=coarse_energy_delta,midpoint_force_balance_N=coarse_force_defect,midpoint_actual_moment_balance_Nm=coarse_moment_defect,midpoint_endpoint_whole_P2_J=[dict(all_elements_positive=c['all_elements_positive'],minimum_lower_bound=c['minimum_lower_bound']) for c in coarse_certs],passed=bool(coarse_pass))
        save()
        if not state['coarse_control']['passed']:raise RuntimeError('Independent coarse midpoint parity failed')
        state.update(completed=True,passed=True);save()
    except BaseException as e:state['failure']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc());save();raise
    finally:save();watcher.terminate();watcher.join(timeout=2)
    return state
