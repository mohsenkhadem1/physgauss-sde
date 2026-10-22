"""Accuracy-qualified optional FP32 corrections; physical arithmetic stays FP64."""
import argparse,json,os,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
OP=ROOT/'runs/operator-dev/p2-mixed-benchmark-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-mixed-benchmark-v01'
CASES=[('axial',5000,'axial_pull','runs/gaussian-campaign-v01/p2-torch-spatial-v01/order3.npz'),('bend',4500,'transverse_end','runs/gaussian-campaign-v01/p2-varied-static-v01/case0-tets12000.npz'),('torsion',6000,'torsion_couple','runs/gaussian-campaign-v01/p2-torsion-blockpcg-v01/torsion-tets12000.npz')]
SOURCES=['experiments/run_p2_mixed_benchmark.py','experiments/campaign_p2_mixed_cg.py','experiments/campaign_p2_block_preconditioner.py','experiments/campaign_p2_torch.py','experiments/campaign_block_surface_controls.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','tests/test_campaign_p2_mixed_cg.py']
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen protocol')
 p=dict(schema='P2-mixed-correction-benchmark-v0.1',status='FROZEN_BEFORE_GPU_CALLS',GPU=0,CPU_threads=2,wall_cap_s=600,cases=[dict(name=a,E=b,force=c,reference=d) for a,b,c,d in CASES],source_sha256={s:sha(ROOT/s) for s in SOURCES},reference_sha256={d:sha(ROOT/d) for _,_,_,d in CASES},gates=dict(original_FP64_force_R_N=1e-10,reference_node_difference_m=1e-8,reference_query_difference_m=1e-8,force_balance_N=1e-8,moment_balance_Nm=1e-8),linear_resources=dict(maximum_total_FP32_iterations=4000,maximum_FP64_fallback_iterations=4000,maximum_refinement_rounds=8),selection='All3 frozen cases; no speed-based case filtering',nonclaims=['Mixedsolverbenchmarknotspatial/dynamic/materialqualification','FP32searchcurvaturesarenotFP64curvaturecertificates','FinalFP64originalresidual/nonlineardescent/Jguardsunchanged','Historicalwalltimesarenotpairedspeedupmeasurements'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in {**p['source_sha256'],**p['reference_sha256']}.items():
  if sha(ROOT/f)!=h:raise ValueError('Changedsource '+f)
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,cases=[],protocol_sha256=digest)
 try:
  import torch
  from experiments.campaign_p2_mixed_cg import MixedCorrectionP2
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_block_surface_controls import surface_control
  torch.set_num_threads(2)
  for name,E,pattern,reference in CASES:
   begin=time.monotonic();row=dict(name=name,completed=False,passed=False)
   try:
    m=StableP2Block((20,10,10),young=E,gravity=0,traction=1,order=3);m.load,control=surface_control(m.rest,m.tets,m.faces,pattern,m.elements);t=MixedCorrectionP2(m,'cuda:0');a=t.solve();q=a['positions'].cpu().numpy();residual=(a['cache']['gradient']-t.force).cpu().numpy();R=float(np.linalg.norm(residual[~m.mask]));reaction=np.where(m.mask,residual,0);balance=float(np.linalg.norm((reaction+m.load).sum(0)));moment=float(np.linalg.norm(np.cross(q,reaction+m.load).sum(0)))
    with np.load(ROOT/reference) as z:
     if not np.array_equal(m.rest,z['rest_m']) or not np.array_equal(m.elements,z['elements']):raise ValueError('Referencegeometrymismatch')
     node=float(np.linalg.norm(q-z['positions_m'],axis=-1).max());query=float(np.linalg.norm(m.query(q,z['material_probes_m'])-z['probe_positions_m'],axis=-1).max())
    row.update(completed=True,passed=R<=1e-10 and node<=1e-8 and query<=1e-8 and balance<=1e-8 and moment<=1e-8,original_FP64_free_R_N=R,maximum_reference_node_error_m=node,maximum_reference_query_error_m=query,force_balance_N=balance,moment_balance_Nm=moment,Newton_history=a['history'],mixed_records=t.mixed_records,known_control=control)
   except Exception as e:row.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc()
   row['elapsed_s']=time.monotonic()-begin;state['cases'].append(row);write(OUT/(name+'.json'),row);write(OUT/'summary.json',state)
  state.update(completed=True,passed=all(x['passed'] for x in state['cases']))
 except Exception as e:state.update(error=str(e),traceback=traceback.format_exc())
 finally:state['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':
  r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
  r=dict(pid=os.getpid(),terminal=False,hard_cap_s=600,protocol_sha256=a.sha);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_mixed_benchmark','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
