"""Prospective same-physics Q3 graded64 continuation; immutable48 reference."""
import argparse,json,multiprocessing,os,shutil,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-streaming-graded64-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-streaming-graded64-v01'

def prepare():
    parent=ROOT/'runs/gaussian-campaign-v01/p2-graded-static-v02/summary.json';old=json.loads(parent.read_text());execution=ROOT/'runs/operator-dev/p2-graded-static-v02/execution.json';parity=ROOT/'runs/gaussian-campaign-v01/p2-streaming-gpu-parity-v01/summary.json'
    if not old['completed'] or not json.loads(execution.read_text())['terminal'] or not json.loads(parity.read_text())['passed']:raise ValueError('Terminal old references/parity required')
    reference=old['meshes'][-1]
    if reference['cells']!=[48,24,24] or not reference['physics_passed']:raise ValueError('Completed48 reference required')
    source=list(json.loads((ROOT/'runs/operator-dev/p2-streaming-gpu-parity-v01/protocol.json').read_text())['source_sha256'])+['experiments/run_p2_streaming_graded64.py','experiments/campaign_p2_jacobian_certificate.py']
    p=dict(schema='C38-P2-streaming-graded64-v0.1',status='FROZEN_BEFORE_NEW_PHYSICS',source_sha256={f:sha(ROOT/f) for f in source},evidence_sha256={str(f.relative_to(ROOT)):sha(f) for f in [parent,execution,parity]},reference48=reference,cells=[64,32,32],all_from_REST=True,physics=dict(E_Pa=4500,nu=.35,rho=625,gravity=0,pattern='transverse_end',force_N=[0,0,-.12],support='left_face',mapping='clamp_graded',quadrature=3),chunk_size=8,CG_cap=4000,GPU=0,CPU_threads=2,hard_cap_s=2400,GPU_allocated_guard_bytes=8*1024**3,RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,gates=dict(original_free_R_N=1e-9,global_force_N=1e-8,actual_moment_Nm=1e-8,clamp_m=0,minimum_J=.1,positive_whole_element_J=True,conditional_spatial_indicator_m=20e-6),whole_J='Positiveoldnormsufficientbound OR positiveNEWdegree3Bernsteinlowerbound; bothreported. NonpositiveBernstein isINCONCLUSIVE notinversion; no clipping.',indicator='delta48to64/(actualhmax48/actualhmax64-1), assumedp>=1engineeringindicator NOT rigorousbound',limits=['Samephysics/control/fullleftclamp; no otherBC guardbypass','AlloldQ3spatialfailures retained','Q80quadrature/time/contact/inference/dataset qualification separate','All63originalfixedprobes includingedges retained; reportpeak/99th/RMSstrain andmotion; 2to8percentpeakgoalnotrelaxed'])
    if (OP/'protocol.json').exists():raise FileExistsError('Preserve frozen protocol')
    write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
    if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedprotocol/GPU')
    p=json.loads((OP/'protocol.json').read_text())
    for f,h in {**p['source_sha256'],**p['evidence_sha256']}.items():
        if sha(ROOT/f)!=h:raise ValueError('Changed pin '+f)
    r=p['reference48']['capture']
    if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed48 field')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,old_failed_gates_preserved=True,quadrature_or_dynamic_or_inference_qualified=False)
    def save():s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
    def interrupt(*_):raise TimeoutError('Frozen2400s/RSS guard')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+2400,p['RSS_guard_bytes'],OUT/'resource-guard.json'),daemon=True);guard.start()
    try:
        import torch
        from experiments.campaign_p2_geometry import P2Geometry
        from experiments.campaign_p2_streaming import StreamingTwoLevelP2
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(p['GPU_allocated_guard_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        def check():
            if torch.cuda.max_memory_allocated()>p['GPU_allocated_guard_bytes']:raise MemoryError('Frozen8GiBGPU allocated')
            if shutil.disk_usage(ROOT).free<p['minimum_disk_free_bytes']:raise RuntimeError('Frozen30GiBdisk reserve')
        class Budget(StreamingTwoLevelP2):
            def evaluate(self,u):
                a=super().evaluate(u);check();return a
            def tangent(self,c,v):
                a=super().tangent(c,v);check();return a
        check();m=P2Geometry(tuple(p['cells']),mapping='clamp_graded');t=Budget(m,'cuda:0',4000,8);answer=t.solve();q=answer['positions'].cpu().numpy();res=(answer['cache']['gradient']-t.force).cpu().numpy()
        # Preserve converged field before any independent certificate/query processing.
        f=OUT/'tets393216.npz'
        with np.load(ROOT/r['path'],allow_pickle=False) as z:points=z['material_probes_m'].copy();oldQ=z['probe_positions_m'].copy()
        Q=m.query(q,points);np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=Q)
        s.update(capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),history=answer['history'],linear_records=t.linear_records,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());save()
        diag=m.diagnostics(dict(positions=q,residual=res,J=answer['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=answer['iterations']));cert=whole_p2_jacobian_bounds(m,q);norm=diag['global_affine_F_positive_J_sufficient_lower_bound'];positive=bool((norm or 0)>0 or cert['all_elements_positive'])
        physical=bool(diag['original_free_residual_N']<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and positive)
        delta=float(np.linalg.norm(Q-oldQ,axis=-1).max());factor=1/(p['reference48']['hmax_m']/m.maximum_reference_tet_edge_m-1);indicator=delta*factor
        s.update(completed=True,passed=physical and indicator<=20e-6,physics_passed=physical,diagnostics=diag,hmax_m=m.maximum_reference_tet_edge_m,whole_J_Bernstein=dict(minimum_lower_bound=cert['minimum_lower_bound'],minimum_coefficient=float(cert['minimum_coefficient'].min()),all_elements_positive=cert['all_elements_positive'],scope=cert['scope']),old_norm_positive=bool((norm or 0)>0),max63query_delta48to64_m=delta,actual_hmax_indicator_factor=factor,conditional_spatial_indicator_m=indicator,spatial_gate_passed=indicator<=20e-6)
    except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:save();guard.terminate();guard.join(timeout=2)
    return s

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='run':
        r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','conditional_spatial_indicator_m','error']}));raise SystemExit(0 if r['completed'] else 1)
    else:
        if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
        if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changedprotocol')
        r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=2400);write(OP/'execution.json',r)
        with (OP/'worker.log').open('x') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_streaming_graded64','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
            try:code=child.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode;r['timeout']=True
        r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
