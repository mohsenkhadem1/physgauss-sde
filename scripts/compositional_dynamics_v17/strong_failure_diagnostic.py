"""Read-only localized replay from last saved frame to reproduce declared guard."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
import torch
from scripts.phystwin.tensor_fixture import basis
from scripts.compositional_dynamics_v17.stronger_development import ROOT,REPORT,RAW,PLAN,build,sha,write,verify

def main():
 torch.set_num_threads(1);p=json.loads(PLAN.read_text());verify(p);rows=[];inputs={}
 for c in p['cases']:
  f=RAW/c['name']/'complete.json'
  if not f.exists():continue
  r=json.loads(f.read_text())
  if r['completed']or 'Jacobian guard'not in r.get('error',''):continue
  path=f.parent/'trajectory.npz';inputs[str(f.relative_to(ROOT))]=sha(f);inputs[str(path.relative_to(ROOT))]=sha(path)
  with np.load(path)as d:u=d['u'][-1].copy();v=d['velocities'][-1].copy();last=float(d['times_s'][-1])
  m,e,_=build(p,c);s=e.initial_state(u=u,v=v);s.time=last;attempted=None
  for j in range(42):
   trial=s.u+p['dt_s']*(s.v+.5*p['dt_s']*e.acceleration(s.force))
   try:s=e.step(s,p['dt_s'])
   except ValueError as ex:
    if 'Jacobian guard'not in str(ex):raise
    attempted=trial;break
  if attempted is None:raise ValueError('Original guard not reproduced within one sample stride')
  sets=[]
  for label,g,quad in [('check',m.engine.check_grad,m.check),('quadrature',m.engine.grad,m.quad)]:
   ue=attempted[m.elements];grad=g.detach().cpu().numpy();F=np.eye(3)+np.einsum('eai,qeaj->qeij',ue,grad);J=np.linalg.det(F);N,_=basis(quad);rest=np.einsum('qa,eai->qei',N,m.rest[m.elements]);cur=np.einsum('qa,eai->qei',N,(m.rest+attempted)[m.elements]);order=np.argsort(J.ravel())[:8]
   sets.append(dict(kind=label,minimum_J=float(J.min()),below_guard_count=int((J<=.1).sum()),total_samples=J.size,worst=[dict(J=float(J.ravel()[i]),rest_m=rest.reshape(-1,3)[i].tolist(),current_m=cur.reshape(-1,3)[i].tolist())for i in order]))
  rows.append(dict(case=c['name'],last_saved_time_s=last,replayed_steps_before_failed_attempt=j,failed_attempt_time_s=s.time+p['dt_s'],original_steps_completed=r['steps_completed'],sets=sets,original_energy_drift=r['maximum_relative_energy_drift'],original_dense_depth_m=r['maximum_dense_penetration_m']))
 write(REPORT/'strong-failure-location-diagnostic.json',dict(source_sha256=sha(__file__),strong_plan_sha256=sha(PLAN),input_sha256=inputs,rows=rows,scope='Exact same-state short replay only for failed completed jobs; no source/law/parameter/guard changes. Location evidence does not establish constitutive-versus-discretization cause.'))
if __name__=='__main__':main()
