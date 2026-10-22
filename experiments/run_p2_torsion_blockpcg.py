"""One prospective numerical-solver repair of failed12000tet torsion case."""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-torsion-blockpcg-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-torsion-blockpcg-v01'
from experiments.run_p2_torch_benchmark import sha,write

def prepare():
 old=ROOT/'runs/gaussian-campaign-v01/p2-varied-static-v01/case2.json';data=json.loads(old.read_text());coarse=data['meshes'][0]
 if coarse['tetrahedra']!=6144 or not coarse['completed'] or data.get('error')!='Original PCG residual exceeds iteration cap':raise ValueError('Expected preserved original fine solver cap failure')
 sources=['experiments/run_p2_torsion_blockpcg.py','experiments/campaign_p2_block_preconditioner.py','experiments/campaign_p2_torch.py','experiments/campaign_block_surface_controls.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/boundary_block.py']
 p=dict(schema='C38-P2-torsion-blockPCG-v0.1',status='FROZEN_BEFORE_RETRY',source_sha256={f:sha(ROOT/f) for f in sources},original_failure=dict(path=str(old.relative_to(ROOT)),sha256=sha(old)),coarse6144=coarse['capture'],cells=[20,10,10],order=3,physics=dict(E_Pa=6000,nu=.35,rho=625,gravity=0,support='left_face',force_pattern='torsion_couple',torque_Nm=.02,initial='REST'),numerical_change='Exact nodal3x3 block-Jacobi only; explicit4000CG cap; original operator/objective/fullH/stationarity/line-search untouched',solver=dict(CG_cap=4000,nonlinear_cap=25,original_solve_tolerance_N=1e-10),gates=dict(original_R_N=1e-9,balance_N=1e-8,actual_moment_balance_Nm=1e-8,clamp_m=0,minimum_J=.1,global_J_certificate_positive=True,conditional4delta_probe_m=20e-6),GPU=2,CPU_threads=2,hard_cap_s=600,cost_forecast_s=[30,180],nonclaims=['Original failed run retained','No changed physical forces/material/BC or numerical-error threshold','4delta conditional p>=1 indicator, not rigorous continuum bound','Quadrature and dynamics remain separately unqualified'])
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen protocol')
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Changed protocol or GPU')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 for k in ['original_failure','coarse6144']:
  if sha(ROOT/p[k]['path'])!=p[k]['sha256']:raise ValueError('Changed original evidence')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();result=dict(completed=False,passed=False,protocol_sha256=digest,original_failure_preserved=True)
 try:
  import torch
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_block_surface_controls import surface_control,support_groups
  from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
  torch.set_num_threads(2);m=StableP2Block(tuple(p['cells']),young=6000,gravity=0,traction=1,order=3);groups,active=support_groups(m.rest,'left_face');m.mask=np.repeat(groups.any(0)[:,None],3,1);m.free=np.flatnonzero(~m.mask.ravel());m.load,control=surface_control(m.rest,m.tets,m.faces,'torsion_couple',m.elements);t=BlockJacobiP2(m,'cuda:0',maximum_cg_iterations=4000);answer=t.solve();q=answer['positions'].cpu().numpy();residual=(answer['cache']['gradient']-t.force).cpu().numpy();diag=m.diagnostics(dict(positions=q,residual=residual,J=answer['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=answer['iterations']))
  with np.load(ROOT/p['coarse6144']['path'],allow_pickle=False) as a:points=a['material_probes_m'].copy();coarse=a['probe_positions_m'].copy()
  queries=m.query(q,points);delta=float(np.linalg.norm(queries-coarse,axis=-1).max());file=OUT/'torsion-tets12000.npz';np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=queries)
  physical=diag['original_free_residual_N']<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and (diag['global_affine_F_positive_J_sufficient_lower_bound'] or 0)>0
  result.update(completed=True,physics_passed=physical,conditional_spatial_indicator_m=4*delta,maximum_fixed_query_delta_m=delta,spatial_indicator_passed=4*delta<=20e-6,passed=physical and 4*delta<=20e-6,diagnostics=diag,known_control=control,history=answer['history'],linear_records=t.linear_records,capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)),quadrature_or_dynamic_or_training_qualified=False)
 except Exception as e:result.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc();result['linear_records']=getattr(locals().get('t'),'linear_records',[])
 finally:result['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',result)
 return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':
  r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error','conditional_spatial_indicator_m']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
  if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changed protocol')
  # Original varied-stage PID proof is required before using its released GPU2.
  prior=json.loads((ROOT/'runs/operator-dev/p2-varied-static-v01/execution.json').read_text())
  if not prior['terminal'] or any(Path('/proc',str(prior[k])).exists() for k in ['pid','worker_pid']):raise ValueError('Original GPU2 stage not drained')
  r=dict(terminal=False,pid=os.getpid(),protocol_sha256=a.sha,hard_cap_s=600);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_torsion_blockpcg','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=600)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
