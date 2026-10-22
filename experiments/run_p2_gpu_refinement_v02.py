"""Prospective fine-mesh extension; previous coarse fields retained unchanged."""
import argparse,json,os,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
OP=ROOT/'runs/operator-dev/p2-gpu-refinement-v02'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-gpu-refinement-v02'
SOURCES=['experiments/run_p2_gpu_refinement_v02.py','experiments/campaign_p2_torch.py','experiments/campaign_p2_transient.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/boundary_block.py']
JOBS=[dict(cells=[16,8,8],hz=3840),dict(cells=[20,10,10],hz=3840),dict(cells=[20,10,10],hz=7680)]

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen protocol')
 p=dict(schema='P2-BDF2-refinement-v0.2',status='FROZEN_BEFORE_RUN',source_sha256={f:sha(ROOT/f) for f in SOURCES},jobs=JOBS,wall_cap_s=900,GPU=0,CPU_threads=2,physics=dict(E_Pa=5000,nu=.35,rho=625,traction_N=1,gravity=0,BC='full_left_face',duration_s=.1,force='sin^2(pi/2*min(t/.05,1)), evaluated at step end'),common_times_s=np.linspace(0,.1,25).tolist(),gates=dict(original_physical_R_N=1e-9,field_refinement_m=20e-6,minimum_J=.1),selection='Three predeclared fine cases; earlier coarse cases remain unchanged; no inference/data admission from residual alone',nonclaims=['Two-mesh difference is not a rigorous continuum error bound','No contact, material generalization or predictive inference claim'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changed protocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,cases=[],comparisons=[],protocol_sha256=digest)
 try:
  import torch
  from experiments.campaign_p2_torch import TorchP2
  from experiments.campaign_p2_transient import P2State,step_p2
  from experiments.campaign_newton_stable_material import StableP2Block
  torch.set_num_threads(2);fields={};X=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3);common=np.linspace(0,.1,25)
  for job in JOBS:
   begin=time.monotonic();m=StableP2Block(tuple(job['cells']),young=5000,gravity=0,traction=1,order=3);model=TorchP2(m,'cuda:0');force=model.force;current=P2State(torch.zeros_like(model.rest),torch.zeros_like(model.rest));previous=None;h=1/job['hz'];steps=round(.1/h);indices=np.rint(common/h).astype(int);U=[current.u.cpu().numpy()];V=[current.v.cpu().numpy()];ledger=[]
   for k in range(steps):
    clock=(k+1)*h;scale=np.sin(np.pi/2*min(clock/.05,1))**2
    new,row=step_p2(model,current,previous,h,force*scale);previous,current=current,new;ledger.append(row)
    if k+1 in indices:U.append(current.u.cpu().numpy());V.append(current.v.cpu().numpy())
   U=np.asarray(U);V=np.asarray(V);Q=np.asarray([m.query(m.rest+u,X) for u in U]);name=f'tets{len(m.tets)}-hz{job["hz"]}';fields[name]=Q
   file=OUT/(name+'.npz');np.savez_compressed(file,rest_m=m.rest,elements=m.elements,tets=m.tets,faces=m.faces,constraint_mask=m.mask,times_s=common,positions_m=m.rest+U,velocities_m_s=V,material_probes_m=X,probe_positions_m=Q)
   row=dict(id=name,completed=True,cells=job['cells'],hz=job['hz'],elapsed_s=time.monotonic()-begin,maximum_physical_R_N=max(x['original_physical_free_R_N'] for x in ledger),minimum_J=min(x['minimum_J'] for x in ledger),ledger=ledger,capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)));state['cases'].append(row);write(OUT/(name+'.json'),row);write(OUT/'summary.json',state)
  for a,b,kind in [('tets6144-hz3840','tets12000-hz3840','space'),('tets12000-hz3840','tets12000-hz7680','time')]:
   error=float(np.linalg.norm(fields[a]-fields[b],axis=-1).max());state['comparisons'].append(dict(a=a,b=b,kind=kind,maximum_field_difference_m=error,passed=error<=20e-6))
  state.update(completed=True,passed=all(x['passed'] for x in state['comparisons']))
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
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_gpu_refinement_v02','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
