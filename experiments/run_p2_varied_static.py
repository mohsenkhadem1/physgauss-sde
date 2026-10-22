"""New materials/known loads/hard supports: prospective spatial validation."""
import argparse,json,os,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
OP=ROOT/'runs/operator-dev/p2-varied-static-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-varied-static-v01'
CASES=[(4500,'left_face','transverse_end'),(9000,'left_face','bending_couple'),(6000,'left_face','torsion_couple'),(6000,'both_ends','top_middle_down'),(7500,'separated_bottom_patches','top_middle_down'),(4500,'right_face','axial_pull'),(9000,'bottom_face','axial_pull')]
SOURCES=['experiments/run_p2_varied_static.py','experiments/campaign_block_surface_controls.py','experiments/campaign_p2_torch.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/boundary_block.py']
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen protocol')
 p=dict(schema='P2-varied-static-v0.1',status='FROZEN_BEFORE_SOLVES',source_sha256={f:sha(ROOT/f) for f in SOURCES},cases=[dict(E_Pa=e,support=s,force_pattern=f) for e,s,f in CASES],meshes=[[16,8,8],[20,10,10]],order=3,GPU=2,CPU_threads=2,wall_cap_s=900,physics=dict(law='Newton1.7 stable polynomial NH, kappa=lambda+mu',nu=.35,density=625,gravity=0,forcing='Known absolute force independent of E/support; from frozen compiler'),gates=dict(original_R_N=1e-9,force_balance_N=1e-8,moment_balance_Nm=1e-8,conditional_spatial_indicator_m=20e-6),spatial_indicator='4*max63fixedquerydifference; conditional p>=1, NOT proven continuum bound',nonclaims=['Quadrature convergence for new conditions is a separate gate','No dynamic/contact/unknown-boundary calibration claim','Restrictive zero-motion controls retained, never counted as informative deformation'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Changed protocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,cases=[],protocol_sha256=digest)
 try:
  import torch
  from experiments.campaign_p2_torch import TorchP2
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_block_surface_controls import support_groups,surface_control
  torch.set_num_threads(2);points=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
  for k,(E,bc,pattern) in enumerate(CASES):
   row=dict(id=k,E_Pa=E,support=bc,force_pattern=pattern,meshes=[],completed=False,passed=False);fields=[]
   try:
    for cells in p['meshes']:
     begin=time.monotonic();m=StableP2Block(tuple(cells),young=E,gravity=0,traction=1,order=3);groups,active=support_groups(m.rest,bc);m.mask=np.repeat(groups.any(0)[:,None],3,axis=1);m.free=np.flatnonzero(~m.mask.ravel());m.load,control=surface_control(m.rest,m.tets,m.faces,pattern,m.elements);t=TorchP2(m,'cuda:0');a=t.solve();q=a['positions'].cpu().numpy();u=q-m.rest;Q=m.query(q,points);fields.append(Q)
     # Full original residual includes reactions; core solve returns free-only.
     residual=(a['cache']['gradient']-t.force).cpu().numpy();R=float(np.linalg.norm(residual[~m.mask]));J=a['cache']['minimum_J'].cpu().numpy()[None]
     diag=m.diagnostics(dict(positions=q,residual=residual,J=J,tolerance_N=1e-10,iterations=a['iterations']))
     passed=R<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['global_affine_F_positive_J_sufficient_lower_bound'] is not None
     file=OUT/f'case{k}-tets{len(m.tets)}.npz';np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=Q)
     r=dict(tetrahedra=len(m.tets),completed=True,passed=passed,elapsed_s=time.monotonic()-begin,known_control=control,diagnostics=diag,history=a['history'],capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)));row['meshes'].append(r);write(OUT/f'case{k}.json',row)
    delta=float(np.linalg.norm(fields[1]-fields[0],axis=-1).max());row.update(completed=True,maximum_fixedquery_delta_m=delta,conditional_spatial_indicator_m=4*delta,passed=all(x['passed'] for x in row['meshes']) and 4*delta<=20e-6)
   except Exception as e:row.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc()
   state['cases'].append(row);write(OUT/f'case{k}.json',row);write(OUT/'summary.json',state)
  state.update(completed=True,passed=all(x['passed'] for x in state['cases']))
 except Exception as e:state.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:state['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':
  r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
  r=dict(pid=os.getpid(),terminal=False,hard_cap_s=900,protocol_sha256=a.sha);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_varied_static','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
