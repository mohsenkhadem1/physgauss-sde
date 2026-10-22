"""Q80 quadrature continuation from one immutable Q3 equilibrium, not rest."""
import argparse,json,multiprocessing,os,shutil,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-warm-exact80-48-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-warm-exact80-48-v01'

def solve_warm(model,reference_positions,device='cpu'):
    import torch
    from experiments.campaign_p2_streaming import StreamingTwoLevelP2
    q=np.asarray(reference_positions)
    if q.dtype!=np.float64 or q.shape!=model.rest.shape or not np.isfinite(q).all():raise ValueError('Matching finite FP64 Q3 field required')
    solver=StreamingTwoLevelP2(model,device,4000,8);initial=torch.tensor(q-model.rest,dtype=torch.float64,device=solver.device)
    before=solver.evaluate(initial);initial_R=float(torch.linalg.vector_norm((before['gradient']-solver.force).masked_fill(solver.mask,0.)))
    del before
    answer=solver.solve(initial=initial,force_tolerance=1e-10,maximum_iterations=25)
    return solver,answer,initial_R

def prepare():
    summary=ROOT/'runs/gaussian-campaign-v01/p2-graded-static-v02/summary.json';old=json.loads(summary.read_text());row=old['meshes'][-1]
    assessment=ROOT/'runs/gaussian-campaign-v01/p2-exact80-operator-v01/summary.json';a=json.loads(assessment.read_text())
    if not old['completed'] or row['cells']!=[48,24,24] or not row['physics_passed'] or not a['completed'] or a['nonlinear_solves']!=0:raise ValueError('Immutable completed Q3 field and operator diagnostic required')
    files=list(json.loads((ROOT/'runs/operator-dev/p2-exact80-operator-v01/protocol.json').read_text())['source_sha256'])+['experiments/run_p2_warm_exact80.py','experiments/campaign_p2_jacobian_certificate.py','tests/test_p2_warm_exact80.py']
    p=dict(schema='C38-Q80-warm-equilibrium-continuation-v0.1',status='FROZEN_BEFORE_EQUILIBRIUM_CORRECTION',source_sha256={f:sha(ROOT/f) for f in files},evidence_sha256={str(f.relative_to(ROOT)):sha(f) for f in [summary,assessment,ROOT/'runs/operator-dev/p2-exact80-operator-v01/terminal-receipt.json',OP/'CPU-control.txt']},reference=row,operator_assessment=a,physics=dict(E_Pa=4500,nu=.35,rho=625,gravity=0,force_N=[0,0,-.12],pattern='transverse_end',support='left_face',mapping='clamp_graded'),initialization='HashboundQ3q minusidenticalpublicrest. WARMSTARTquadraturecontinuation, noindependentbasinclaim; notfromREST',actual_quadrature='positiveDuffy5x4x4 degree6exact, all80points',chunk_size=8,CG_cap=4000,Newton_cap=25,GPU=2,CPU_threads=2,hard_cap_s=900,GPU_allocated_guard_bytes=8*1024**3,RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,gates=dict(original_free_R_N=1e-10,global_force_N=1e-8,actual_moment_Nm=1e-8,clamp_m=0,minimum_J=.1,whole_element_J_positive=True,same63query_Q3toQ80_m=20e-6),whole_J='Positive Bernsteinlowerbound oroldnormsufficientbound, bothreported. NonpositiveBernstein isINCONCLUSIVE notinversion.',limitations=['Onefixed48meshquadraturecorrection; nojointspatialqualification','NoQ3rerun/changedphysicalparameters/contact/time/observations','25Newtoncap andalloriginalFP64operator/acceptance unchanged','Alloldfailedspatial/excitationgates retained; reportpeakstrainwithouttrimming'])
    if (OP/'protocol.json').exists():raise FileExistsError('Preserve frozen protocol')
    write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
    if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Changed protocol/GPU')
    p=json.loads((OP/'protocol.json').read_text())
    for f,h in {**p['source_sha256'],**p['evidence_sha256']}.items():
        if sha(ROOT/f)!=h:raise ValueError('Changedpin '+f)
    r=p['reference']['capture']
    if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('ChangedQ3field')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,warmstart=True,independent_basin_claim=False,joint_spatial_or_dynamic_or_inference_qualified=False)
    def save():s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
    def interrupt(*_):raise TimeoutError('Frozen900s/RSSguard')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+900,p['RSS_guard_bytes'],OUT/'resource-guard.json'),daemon=True);guard.start()
    try:
        import torch
        from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(p['GPU_allocated_guard_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_bytes']:raise RuntimeError('Frozen30GiBdiskreserve')
        m=ExactBulkP2Geometry((48,24,24),mapping='clamp_graded')
        with np.load(ROOT/r['path'],allow_pickle=False) as z:
            q3=z['positions_m'].copy();points=z['material_probes_m'].copy();oldQ=z['probe_positions_m'].copy()
            if not np.array_equal(z['rest_m'],m.rest) or not np.array_equal(z['elements'],m.elements) or not np.array_equal(z['constraint_mask'],m.mask) or not np.array_equal(z['applied_force_N'],m.load):raise ValueError('Identicaldiscreterest/patch/control/clamprequired')
        t,answer,initial_R=solve_warm(m,q3,'cuda:0');q=answer['positions'].cpu().numpy();res=(answer['cache']['gradient']-t.force).cpu().numpy();Q=m.query(q,points);f=OUT/'tets165888-exact80.npz'
        np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=Q)
        s.update(initial_Q80_free_residual_N=initial_R,capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),history=answer['history'],linear_records=t.linear_records,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());save()
        if torch.cuda.max_memory_allocated()>p['GPU_allocated_guard_bytes']:raise MemoryError('Frozen8GiBGPUallocatedguard')
        diag=m.diagnostics(dict(positions=q,residual=res,J=answer['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=answer['iterations']));cert=whole_p2_jacobian_bounds(m,q);norm=diag['global_affine_F_positive_J_sufficient_lower_bound'];positive=bool((norm or 0)>0 or cert['all_elements_positive']);physical=bool(diag['original_free_residual_N']<=1e-10 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and positive)
        querydelta=float(np.linalg.norm(Q-oldQ,axis=-1).max());s.update(completed=True,passed=physical and querydelta<=20e-6,physics_passed=physical,diagnostics=diag,whole_J_Bernstein=dict(minimum_lower_bound=cert['minimum_lower_bound'],all_elements_positive=cert['all_elements_positive'],scope=cert['scope']),allnode_Q3toQ80_max_delta_m=float(np.linalg.norm(q-q3,axis=-1).max()),max63query_Q3toQ80_delta_m=querydelta,quadrature_query_gate_passed=querydelta<=20e-6)
    except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:save();guard.terminate();guard.join(timeout=2)
    return s

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='run':
        r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','max63query_Q3toQ80_delta_m','error']}));raise SystemExit(0 if r['completed'] else 1)
    else:
        if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
        if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changedprotocol')
        r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=900);write(OP/'execution.json',r)
        with (OP/'worker.log').open('x') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_warm_exact80','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
            try:code=child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode;r['timeout']=True
        r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
