"""NEW softer-floor1e6 CPU controls; oldfloor law/results stay unchanged."""
import argparse,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from experiments.run_p2_corrected_contact_order128 import ROOT,sha,write,ENV_SITE
OP=ROOT/'runs/operator-dev/p2-soft-floor-controls-v01'
SOURCES=['experiments/diagnose_soft_floor_controls.py','experiments/campaign_soft_floor_controls.py','tests/test_campaign_soft_floor_controls.py']

def prepare():
 prior=json.loads((ROOT/'runs/operator-dev/p2-contact-root128-time-control-v01/protocol.json').read_text());pins={**prior['source_sha256'],**{x:sha(ROOT/x) for x in SOURCES}}
 p=dict(schema='CPU-NEWsoftfloor1e6-threegeometry-Eg-controls-v01',status='FROZEN_BEFORE_CPU_DIAGNOSTIC_NO_GPU',source_sha256=pins,installed_warp_source_sha256=prior['installed_warp_source_sha256'],scipy_source_sha256=prior['scipy_source_sha256'],CPU_affinity_cores=[0,1],hard_wall_cap_s=180,meshes=[[4,2,2],[6,3,3],[8,4,4]],states=['analytic_uniform_freefall_T0','smooth_small_noncontact','uniform_slight_penetration','smooth_small_partial_contact'],gates=dict(E_J=1e-12,g_L2_N=1e-10,J_difference=2e-14,mass_R_N=1e-8,adaptive_E_atol_J=1e-12,adaptive_g_atol_N=1e-10,adaptive_rtol=1e-10),physics={**prior['physics'],'kappa_N_m3':1e6,'boundary_change':'NEWfloor1000xsofter;NOToldfailurefix'},preserved_original_full_H_failure=dict(path='runs/operator-dev/p2-contact-geometry-controls-v01/preserved-full-H-failure.json',sha256=sha(ROOT/'runs/operator-dev/p2-contact-geometry-controls-v01/preserved-full-H-failure.json')),scope='NEW independent NumPy bulk + strict adaptive E/g-only plane vsCPU128; CPU64/128 fixed-state check, fullM gravity/solve. NO force/Hessian/tolerance changes to old production; no trajectory/surface-order qualification.',no_GPU_or_saved_response_inputs=True)
 if (OP/'CPU-protocol.json').exists():raise FileExistsError('Preserve protocol')
 write(OP/'CPU-protocol.json',p);return sha(OP/'CPU-protocol.json')

def run(digest):
 path=OP/'CPU-protocol.json'
 if sha(path)!=digest:raise ValueError('Changedprotocol')
 p=json.loads(path.read_text())
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('CPU2 ONLY')
 if any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedsource')
 sys.path.append(str(ENV_SITE));import warp as wp
 from experiments.campaign_p2_warp_reference_order128 import WarpReferenceForce128
 from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
 from experiments.campaign_soft_floor_controls import assess_geometry_controls
 from experiments.run_p2_contact_validation import build
 def cap(*a):raise TimeoutError('Frozen180s CPUcap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(180);wp.config.kernel_cache_dir=str(OP/'warp-cache-CPU');start=time.monotonic();s=dict(completed=False,passed=False,meshes=[],protocol_sha256=digest)
 try:
  for cells in p['meshes']:
   m=build(cells,300000.,9.81);a=WarpReferenceForce128(m,kappa_A=1e6,device='cpu');b=WarpReferenceForce(m,kappa_A=1e6,device='cpu');row=dict(cells=cells,completed=False);s['meshes'].append(row)
   def save(rows):row['controls']=rows;write(OP/'CPU-result.json',s)
   row.update(assess_geometry_controls(m,a,b,kappa_A=1e6,after_state=save));write(OP/'CPU-result.json',s)
  s.update(completed=True,passed=all(r['passed'] for r in s['meshes']))
 except BaseException as error:s['error']=repr(error)
 finally:signal.alarm(0);s['seconds']=time.monotonic()-start;write(OP/'CPU-result.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r));raise SystemExit(0 if a.action=='prepare' or r['completed'] else 1)
