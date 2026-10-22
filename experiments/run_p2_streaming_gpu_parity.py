"""One prospective Q3 12k streaming/operator/from-rest parity check on GPU2."""
import argparse,json,multiprocessing,os,shutil,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-streaming-gpu-parity-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-streaming-gpu-parity-v01'

def prepare():
    previous=json.loads((ROOT/'runs/operator-dev/p2-streaming-v01/protocol.json').read_text())
    sources=list(previous['source_sha256'])+['experiments/run_p2_streaming_gpu_parity.py','experiments/campaign_p2_geometry.py','experiments/campaign_block_surface_controls.py','experiments/run_p2_torch_benchmark.py','experiments/run_static_p2_axial_continuation.py']
    result=ROOT/'runs/gaussian-campaign-v01/p2-two-level-static-v01/tets12000.json'
    row=json.loads(result.read_text())
    if not row['completed'] or not row['physics_passed']:raise ValueError('Completed frozen monolithic reference required')
    p=dict(schema='C38-streamingP2-GPU-parity-v0.1',status='FROZEN_BEFORE_GPU_CALLS',source_sha256={s:sha(ROOT/s) for s in sources},reference_result=dict(path=str(result.relative_to(ROOT)),sha256=sha(result)),reference_capture=row['capture'],cells=[20,10,10],physics=dict(E_Pa=4500,nu=.35,rho=625,gravity=0,pattern='transverse_end',support='left_face',mapping='uniform',quadrature=3),quadrature_chunk_size=8,maximum_CG_iterations=4000,GPU=2,CPU_threads=2,hard_cap_s=180,GPU_allocated_guard_bytes=8*1024**3,RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,controls_at_frozen_reference='monolithic BlockJacobiP2 energy/gradient/full HVP/node3x3/full consistent mass vs streamed operator; fixed random direction seed2026102204',operator_relative_tolerance=1e-10,operator_absolute_tolerance=1e-11,from_rest=True,gates=dict(original_R_N=1e-9,global_force_N=1e-8,actual_moment_Nm=1e-8,clamp_m=0,minimum_J=.1,positive_global_J=True,max_allnode_reference_delta_m=1e-8),limits=['One fixed-discretization Q3 implementation parity only','No fine Q80 launch or spatial/time/dataset admission','Original FP64 operator/residual/line-search/J and physical force unchanged'])
    if (OP/'protocol.json').exists():raise FileExistsError('Preserve frozen protocol')
    write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
    if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Protocol/GPU mismatch')
    p=json.loads((OP/'protocol.json').read_text())
    for f,h in p['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
    for r in [p['reference_result'],p['reference_capture']]:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed frozen evidence')
    if shutil.disk_usage(ROOT).free<p['minimum_disk_free_bytes']:raise RuntimeError('Disk reserve')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,controls={},long_Q80_launched=False)
    def save():s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
    def interrupt(sig,frame):raise TimeoutError('Frozen180s/12GiB RSS guard')
    signal.signal(signal.SIGTERM,interrupt)
    guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+180,12*1024**3,OUT/'resource-guard.json'),daemon=True);guard.start()
    try:
        import torch
        from experiments.campaign_p2_geometry import P2Geometry
        from experiments.campaign_p2_streaming import StreamingTwoLevelP2
        from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(p['GPU_allocated_guard_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        def check():
            if torch.cuda.max_memory_allocated()>p['GPU_allocated_guard_bytes']:raise MemoryError('Frozen8GiB GPU allocated guard')
        class Budget(StreamingTwoLevelP2):
            def tangent(self,cache,v):
                result=super().tangent(cache,v);check();return result
        m=P2Geometry(tuple(p['cells']));t=Budget(m,'cuda:0',4000,8);mono=BlockJacobiP2(m,'cuda:0',4000)
        with np.load(ROOT/p['reference_capture']['path'],allow_pickle=False) as a:reference=a['positions_m'].copy();points=a['material_probes_m'].copy()
        u=torch.tensor(reference-m.rest,dtype=torch.float64,device='cuda:0');v=torch.tensor(np.random.default_rng(2026102204).normal(size=m.rest.shape),dtype=torch.float64,device='cuda:0')
        cs=t.evaluate(u);cm=mono.evaluate(u)
        pairs={'energy_J':(cs['energy'],cm['energy']),'gradient_N':(cs['gradient'],cm['gradient']),'full_HVP_N':(t.tangent(cs,v),mono.tangent(cm,v)),'node_blocks_N_m':(t.nodal_blocks(cs),mono.nodal_blocks(cm)),'consistent_mass_kg_m':(t.mass(v),mono.mass(v))}
        for name,(a,b) in pairs.items():
            difference=float(abs(a-b).max());scale=float(abs(b).max());passed=difference<=p['operator_absolute_tolerance']+p['operator_relative_tolerance']*scale
            s['controls'][name]=dict(max_absolute_difference=difference,reference_max_absolute=scale,passed=passed)
        check();save()
        if not all(x['passed'] for x in s['controls'].values()):raise RuntimeError('Original operator parity failed')
        del pairs,cs,cm,mono,u,v;torch.cuda.empty_cache()
        answer=t.solve();q=answer['positions'].cpu().numpy();residual=(answer['cache']['gradient']-t.force).cpu().numpy()
        diag=m.diagnostics(dict(positions=q,residual=residual,J=answer['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=answer['iterations']))
        physical=bool(diag['original_free_residual_N']<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and (diag['global_affine_F_positive_J_sufficient_lower_bound'] or 0)>0)
        delta=float(np.linalg.norm(q-reference,axis=1).max());check();f=OUT/'streaming-Q3-tets12000.npz'
        np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=m.query(q,points))
        s.update(completed=True,passed=physical and delta<=1e-8,physics_passed=physical,allnode_reference_max_delta_m=delta,diagnostics=diag,history=answer['history'],linear_records=t.linear_records,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),GPU_name=torch.cuda.get_device_name(0),physical_GPU=2)
    except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:save();guard.terminate();guard.join(timeout=2)
    return s

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='run':
        s=run(a.sha);print(json.dumps({k:s.get(k) for k in ['completed','passed','elapsed_s','allnode_reference_max_delta_m','error']}));raise SystemExit(0 if s['passed'] else 1)
    else:
        if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
        if sha(OP/'protocol.json')!=a.sha:raise ValueError('Protocol changed')
        r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha);write(OP/'execution.json',r)
        with (OP/'worker.log').open('x') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_streaming_gpu_parity','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
            try:code=child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode;r['timeout']=True
        r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
