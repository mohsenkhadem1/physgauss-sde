"""Two fixed 48-mesh Newton LINEAR corrections; no full nonlinear rerun."""
import argparse,json,multiprocessing,os,shutil,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-graded-coarse-benchmark-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-graded-coarse-benchmark-v01'

def prepare():
    warmproto=ROOT/'runs/operator-dev/p2-warm-exact80-48-v01/protocol.json';wp=json.loads(warmproto.read_text())
    warm=ROOT/'runs/gaussian-campaign-v01/p2-warm-exact80-48-v01/summary.json';w=json.loads(warm.read_text())
    parity=ROOT/'runs/gaussian-campaign-v01/p2-keast-operator-parity-v01/summary.json'
    if not w['passed'] or not json.loads(parity.read_text())['passed']:raise ValueError('Completed immutable48 controls required')
    sources=json.loads((ROOT/'runs/operator-dev/p2-keast-operator-parity-v01/protocol.json').read_text())['source_sha256']
    for f in ['experiments/campaign_p2_graded_coarse.py','tests/test_campaign_p2_graded_coarse.py','experiments/run_p2_graded_coarse_benchmark.py']:sources[f]=sha(ROOT/f)
    p=dict(schema='C38-gradedcoarse48-linear-benchmark-v0.1',status='FROZEN_BEFORE_GPU_CALLS',source_sha256=sources,evidence_sha256={str(f.relative_to(ROOT)):sha(f) for f in [warmproto,warm,parity,ROOT/'runs/operator-dev/p2-graded-coarse-v01/tests.txt']},reference_Q3=wp['reference']['capture'],reference_exact80=w['capture'],cells=[48,24,24],models=['olduniform4x2x2restcorrection','newgraded8x4x4restcorrection'],same_physical_operator='Keast24degree6exactstableNH/allold80+10Jguards/fullconsistentM; unchangedE4500/nu.35/rho625/.12Ntransverse/fullleft/gradedfine',initialization='BothsameimmutableQ3u; nofromRESTequilibriumrerun',linear_system='H_exact24(u_Q3) delta=-(g_exact24(u_Q3)-F)_free',CG_rtol='Originalmin(.01,max(1e-8,.1*normR**.5))',CG_atol_N=1e-12,CG_cap=4000,Newton_solves=0,linear_solves=2,candidate_evaluation='Evaluateu_Q3+deltaoncepercase; not a nonlinearstep/trajectory/admission',GPU=0,CPU_threads=2,hard_cap_s=600,GPU_allocated_guard_bytes=8*1024**3,RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,gates=dict(original_linear_R='FrozenCGtrue-residualacceptance, also<=1e-10N',candidate_original_free_R_N=1e-10,exact_clamp_m=0,minimum_J=.1,positive_Bernstein=True,max_same_system_direction_difference_m=1e-8,max_candidate_vs_frozen_exact80_allnode_m=1e-8),limits=['Onlypreconditionerchanges; physicalfineoperatorunchanged','All64wallcap/oldspatialfailures retained','Benchmarkcost/iterationsnotphysicalaccuracyproof','Noautomaticfull64retry'])
    if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
    write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
    if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedprotocol/GPU')
    p=json.loads((OP/'protocol.json').read_text())
    for f,h in {**p['source_sha256'],**p['evidence_sha256']}.items():
        if sha(ROOT/f)!=h:raise ValueError('Changedpin '+f)
    for r in [p['reference_Q3'],p['reference_exact80']]:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changedimmutablefield')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,cases=[],Newton_solves=0,linear_solves=0,full64_reruns=0)
    def save():s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
    def interrupt(*_):raise TimeoutError('Frozen600s/RSSguard')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+600,p['RSS_guard_bytes'],OUT/'resource-guard.json'),daemon=True);guard.start()
    try:
        import torch
        from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry
        from experiments.campaign_p2_streaming import StreamingTwoLevelP2
        from experiments.campaign_p2_graded_coarse import StreamingGradedCoarseP2
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(p['GPU_allocated_guard_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        def check():
            if torch.cuda.max_memory_allocated()>p['GPU_allocated_guard_bytes']:raise MemoryError('Frozen8GiBGPUallocated')
            if shutil.disk_usage(ROOT).free<p['minimum_disk_free_bytes']:raise RuntimeError('Frozen30GiBdiskreserve')
        with np.load(ROOT/p['reference_Q3']['path'],allow_pickle=False) as z:q3=z['positions_m'].copy();reference_rest=z['rest_m'].copy();reference_force=z['applied_force_N'].copy()
        with np.load(ROOT/p['reference_exact80']['path'],allow_pickle=False) as z:q80=z['positions_m'].copy()
        directions=[];initial_forces=[]
        for name,cls in [('uniform4',StreamingTwoLevelP2),('graded8',StreamingGradedCoarseP2)]:
            begin=time.monotonic();row=dict(name=name,completed=False)
            try:
                m=KeastBulkP2Geometry(tuple(p['cells']),mapping='clamp_graded')
                if not np.array_equal(m.rest,reference_rest) or not np.array_equal(m.load,reference_force):raise ValueError('Samephysicaldiscreteinputs required')
                t=cls(m,'cuda:0',4000,8);check();u=torch.tensor(q3-m.rest,dtype=torch.float64,device=t.device);cache=t.evaluate(u);R=(cache['gradient']-t.force).masked_fill(t.mask,0.);norm=float(torch.linalg.vector_norm(R));rtol=min(.01,max(1e-8,.1*norm**.5));initial_forces.append(R.cpu().numpy());initialization=time.monotonic()-begin
                clock=time.monotonic();d,k=t.cg(cache,-R,0.,rtol=rtol,atol=1e-12);torch.cuda.synchronize();linear_s=time.monotonic()-clock;s['linear_solves']+=1
                true=(-R-t.tangent(cache,d)).masked_fill(t.mask,0.);true_R=float(torch.linalg.vector_norm(true));trial=t.evaluate(u+d);trial_R=float(torch.linalg.vector_norm((trial['gradient']-t.force).masked_fill(t.mask,0.)));q=(t.rest+u+d).cpu().numpy();cert=whole_p2_jacobian_bounds(m,q);delta=float(np.linalg.norm(q-q80,axis=1).max());direction=d.cpu().numpy();directions.append(direction)
                physical=bool(true_R<=1e-10 and trial_R<=1e-10 and not bool((d[t.mask]!=0).any()) and float(trial['minimum_J'])>.1 and cert['all_elements_positive'] and delta<=1e-8);check();f=OUT/f'{name}-correction.npz';np.savez_compressed(f,correction_m=direction)
                row.update(completed=True,passed=physical,initial_free_R_N=norm,requested_CG_rtol=rtol,original_linear_R_N=true_R,candidate_free_R_N=trial_R,candidate_minimum_J=float(trial['minimum_J']),candidate_Bernstein_lower_bound=cert['minimum_lower_bound'],candidate_vs_frozen_exact80_maxnode_m=delta,coarse_free_dofs=int(t.coarse_free.numel()),initialization_s=initialization,linear_s=linear_s,CG_iterations=k,linear_records=t.linear_records,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
                if not physical:raise RuntimeError('Originalsame-systemlinear/candidateparity gates failed')
            except Exception as e:row.update(error=repr(e));raise
            finally:row['elapsed_s']=time.monotonic()-begin;s['cases'].append(row);save()
            del t,m,u,cache,R,d,true,trial;torch.cuda.empty_cache()
        maximum=float(np.linalg.norm(directions[1]-directions[0],axis=1).max());rhsdiff=float(abs(initial_forces[1]-initial_forces[0]).max());s.update(completed=True,passed=maximum<=1e-8 and rhsdiff<=1e-11 and all(r['passed'] for r in s['cases']),max_correction_difference_m=maximum,max_RHS_difference_N=rhsdiff,graded8_to_uniform4_linear_time_ratio=s['cases'][1]['linear_s']/s['cases'][0]['linear_s'],graded8_to_uniform4_iterations_ratio=s['cases'][1]['CG_iterations']/s['cases'][0]['CG_iterations'])
    except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:save();guard.terminate();guard.join(timeout=2)
    return s

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='run':
        r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error','graded8_to_uniform4_linear_time_ratio']}));raise SystemExit(0 if r['completed'] else 1)
    else:
        if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
        if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changedprotocol')
        r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=600);write(OP/'execution.json',r)
        with (OP/'worker.log').open('x') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_graded_coarse_benchmark','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
            try:code=child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode;r['timeout']=True
        r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
