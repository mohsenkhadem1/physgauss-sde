"""One separate CPU-only Q16 dimension/scaling control; no trajectory inputs."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_rhs import WarpP2Force
from experiments.campaign_p2_torch_contact import PlaneContactP2
from experiments.run_p2_contact_validation import ROOT,build


def test_q16_cpu_force_energy_J_at_rest_and_manufactured_partial_contact():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert len(os.sched_getaffinity(0))==2
 torch.set_num_threads(2)
 assert wp.__version__=='1.18.0'
 op=ROOT/'runs/operator-dev/p2-warp-surface-q16-v01';wp.config.kernel_cache_dir=str(op/'prelaunch-cpu-warp-cache')
 m=build([4,2,2],300000.,9.81);ref=PlaneContactP2(m,'cpu',kappa_A=1e9,surface_order=16);a={k:getattr(ref,k).numpy().copy() for k in WarpP2Force.REQUIRED};obj=WarpP2Force(a,young=m.young,poisson=m.poisson,kappa_A=1e9,device='cpu')
 rest=np.zeros_like(m.rest);partial=np.zeros_like(rest);partial[:,2]=-.051+.002*(m.rest[:,0]/.24)**2
 rows=[]
 for name,u in [('rest',rest),('manufactured_quadratic_partial_contact',partial)]:
  expected=ref.evaluate(torch.tensor(u,dtype=torch.float64));actual=obj.evaluate(u);active=int(expected['contact_active'].sum());total=expected['contact_active'].numel()
  row=dict(state=name,force_L2_difference_N=float(np.linalg.norm(actual['gradient']-expected['gradient'].numpy())),energy_difference_J=abs(actual['energy']-float(expected['energy'])),minimum_J_difference=abs(actual['minimum_J']-float(expected['minimum_J'])),active_quadrature_points=active,total_surface_quadrature_points=total)
  if name=='rest':assert active==0
  else:assert 0<active<total
  assert a['surface_shapes'].shape==(256,6)
  assert row['force_L2_difference_N']<=1e-10
  assert row['energy_difference_J']<=1e-12
  assert row['minimum_J_difference']<=2e-14
  rows.append(row)
 source=Path(__file__).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=dict(completed=True,passed=True,role='SeparateprelaunchCPU-onlymanufacturedQ16control; notnewtrajectory/data',source=dict(path=str(source.relative_to(ROOT)),sha256=sha(source)),reviewed_Q16_protocol_sha256=sha(op/'protocol.json'),versions=dict(torch=torch.__version__,warp=wp.__version__,numpy=np.__version__),CPU_affinity_cores=sorted(os.sched_getaffinity(0)),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],states=rows,physical_inputs=dict(E_Pa=300000.,nu=.35,rho=625.,kappa_N_m3=1e9,surface_Q=16,cells=[4,2,2]),synthetic_field='u_z=-.051+.002*(rest_x/.24)^2;u_x=u_y=0 (P2compatible, finitepartialcontact)',no_actual_trajectory_values_read=True)
 (op/'prelaunch-Q16-CPU-control.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(rows))
