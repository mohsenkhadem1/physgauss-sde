"""No solves: exact24/exact80 operators at the immutable corrected48 field."""
import argparse,json,multiprocessing,os,shutil,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-keast-operator-parity-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-keast-operator-parity-v01'

def prepare():
    oldfile=ROOT/'runs/gaussian-campaign-v01/p2-warm-exact80-48-v01/summary.json';old=json.loads(oldfile.read_text())
    terminal=ROOT/'runs/operator-dev/p2-warm-exact80-48-v01/terminal-receipt.json'
    if not old['completed'] or not old['passed'] or not json.loads(terminal.read_text())['GPU2_released_to_root_gravity_stage']:raise ValueError('Completed exact80 corrected48 field required')
    row=dict(cells=[48,24,24],capture=old['capture'])
    files=list(json.loads((ROOT/'runs/operator-dev/p2-warm-exact80-48-v01/protocol.json').read_text())['source_sha256'])+['experiments/run_p2_keast_operator_parity.py','experiments/campaign_p2_keast_quadrature.py','tests/test_campaign_p2_keast_quadrature.py']
    p=dict(schema='C38-Keast24-exact80-operator-parity-v0.1',status='FROZEN_BEFORE_OPERATOR_CALLS',source_sha256={f:sha(ROOT/f) for f in files},evidence_sha256={str(f.relative_to(ROOT)):sha(f) for f in [oldfile,terminal]},reference=row,physical_changes=0,nonlinear_solves=0,CG_solves=0,operators=['energy','gradient','onefullHVP','node3x3blocks','fullconsistentmass'],HVP_direction='publicrestv=[0,0,.001*(x/.24)^2];exactclampzero; notresponsefitted',quadrature=[80,24],chunk_size=8,GPU=2,CPU_threads=2,hard_cap_s=180,GPU_allocated_guard_bytes=8*1024**3,RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,operator_absolute_tolerance=1e-11,operator_relative_tolerance=1e-10,original_free_R_tolerance_N=1e-10,decision='Require operator parity and both originalfreeR<=1e-10 at sameexact80equilibrium. No newequilibrium or displacement/spatialqualification.',CPU_evidence='Root reported both exhaustive degree6moments and independent80operator controls PASS2.20s before this protocol',limits=['SameconstantmaterialstableNH/fullleft/graded48/.12Ntransverse','Alloldspatialfailures unchanged','Exact24 retains old80 and10corner/edge guardpoints','Published decimalrule is numerically degree6exact at FP64 precision','NoNewton/CG/newtrajectory or inferenceadmission'])
    if (OP/'protocol.json').exists():raise FileExistsError('Preserve protocol')
    write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
    if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Changedprotocol/GPU')
    p=json.loads((OP/'protocol.json').read_text())
    for f,h in {**p['source_sha256'],**p['evidence_sha256']}.items():
        if sha(ROOT/f)!=h:raise ValueError('Changedpin '+f)
    r=p['reference']['capture']
    if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changedimmutableexact80field')
    OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,protocol_sha256=digest,nonlinear_solves=0,CG_solves=0)
    def save():s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
    def interrupt(*_):raise TimeoutError('Frozen180s/RSSguard')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),begin+180,p['RSS_guard_bytes'],OUT/'resource-guard.json'),daemon=True);guard.start()
    try:
        import torch
        from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry
        from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
        from experiments.campaign_p2_streaming import StreamingTwoLevelP2
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(p['GPU_allocated_guard_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        def budget():
            if torch.cuda.max_memory_allocated()>p['GPU_allocated_guard_bytes']:raise MemoryError('Frozen8GiB GPUallocated')
            if shutil.disk_usage(ROOT).free<p['minimum_disk_free_bytes']:raise RuntimeError('Frozen30GiBdiskreserve')
        with np.load(ROOT/r['path'],allow_pickle=False) as z:q=z['positions_m'].copy();reference_rest=z['rest_m'].copy();reference_force=z['applied_force_N'].copy()
        outputs={}
        for key,constructor in [('exact80',ExactBulkP2Geometry),('exact24',KeastBulkP2Geometry)]:
            start=time.monotonic();m=constructor((48,24,24),mapping='clamp_graded')
            if not np.array_equal(m.rest,reference_rest) or not np.array_equal(m.load,reference_force):raise ValueError('Identicalrest/controlrequired')
            t=StreamingTwoLevelP2(m,'cuda:0',4000,8);u=torch.tensor(q-m.rest,dtype=torch.float64,device=t.device);v=torch.zeros_like(t.rest);v[:,2]=.001*(t.rest[:,0]/.24)**2;v.masked_fill_(t.mask,0.);cache=t.evaluate(u);H=t.tangent(cache,v);blocks=t.nodal_blocks(cache);Mv=t.mass(v);R=(cache['gradient']-t.force).masked_fill(t.mask,0.)
            outputs[key]=dict(energy_J=float(cache['energy']),gradient=cache['gradient'].cpu().numpy(),HVP=H.cpu().numpy(),nodeblocks=blocks.cpu().numpy(),fullmass=Mv.cpu().numpy(),minimum_J=float(cache['minimum_J']))
            s[key]=dict(energy_J=float(cache['energy']),original_free_residual_N=float(torch.linalg.vector_norm(R)),minimum_J=float(cache['minimum_J']),elapsed_s=time.monotonic()-start,quadrature_points=len(m.bary),additional_J_guard_points=len(m.check_bary));budget();save();del cache,t,m,u,v,H,blocks,Mv,R;torch.cuda.empty_cache()
        a,b=outputs['exact80'],outputs['exact24'];s['energy_difference_J']=b['energy_J']-a['energy_J'];s['operator_changes']={}
        for name in ['gradient','HVP','nodeblocks','fullmass']:
            delta=b[name]-a[name];norm=float(np.linalg.norm(delta));scale=float(np.linalg.norm(a[name]));s['operator_changes'][name]=dict(max_absolute_difference=float(abs(delta).max()),L2_difference=norm,exact80_L2=scale,relative_L2=norm/scale if scale else None,passed=float(abs(delta).max())<=p['operator_absolute_tolerance']+p['operator_relative_tolerance']*float(abs(a[name]).max()))
        file=OUT/'operator-differences.npz';np.savez_compressed(file,gradient_difference_N=b['gradient']-a['gradient'],HVP_difference_N=b['HVP']-a['HVP']);s['energy_parity_passed']=abs(s['energy_difference_J'])<=p['operator_absolute_tolerance']+p['operator_relative_tolerance']*abs(a['energy_J']);s['original_free_residuals_passed']=all(s[key]['original_free_residual_N']<=p['original_free_R_tolerance_N'] for key in ['exact80','exact24']);s.update(completed=True,passed=s['energy_parity_passed'] and s['original_free_residuals_passed'] and all(r['passed'] for r in s['operator_changes'].values()),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)),inference_or_displacement_accuracy_qualified=False)
    except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:save();guard.terminate();guard.join(timeout=2)
    return s

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='run':
        r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','elapsed_s','error','operator_changes']}));raise SystemExit(0 if r.get('passed') else 1)
    else:
        if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
        if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changedprotocol')
        r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=180);write(OP/'execution.json',r)
        with (OP/'worker.log').open('x') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_keast_operator_parity','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
            try:code=child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode;r['timeout']=True
        r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
