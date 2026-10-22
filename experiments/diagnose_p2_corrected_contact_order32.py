"""CPU-only prospective26state root32/64 E/g control before any new rollout."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_reference_order32 import WarpReferenceForce32
from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
from experiments.run_p2_contact_validation import build
from experiments.run_p2_contact_refinement import T0,END
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-corrected-contact-order32-v01';PRIOR=ROOT/'runs/operator-dev/p2-corrected-contact-reference-v01';CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-reference-v01/accuracy_1e10/trajectory.npz';EXPECTED='633ba220edf8f4d5cfb87adb7e6e7db54a74a6bd321270b4e2e454801c314d1c'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def prepare():
 if (OP/'CPU-protocol.json').exists():raise FileExistsError('Preserveprotocol')
 prior=json.loads((PRIOR/'execution.json').read_text())
 if not prior.get('terminal') or not prior.get('completed') or not prior.get('time_reference_qualified') or sha(CAPTURE)!=EXPECTED:raise ValueError('Completedtimequalified64captureonly')
 sources=['experiments/campaign_p2_warp_reference_order32.py','tests/test_campaign_p2_warp_reference_order32.py','experiments/diagnose_p2_corrected_contact_order32.py','experiments/campaign_p2_warp_reference_force.py','experiments/campaign_p2_warp_root_contact.py','experiments/campaign_p2_warp_rhs.py','experiments/run_p2_contact_validation.py']
 p=dict(schema='CPU-root32-vs-timequalified64-fixed26state-Eg-v1',status='FROZEN_BEFORE_TIGHT_CAPTURE_VALUES',source_sha256={f:sha(ROOT/f) for f in sources},prior_evidence_sha256={str((PRIOR/f).relative_to(ROOT)):sha(PRIOR/f) for f in ['protocol.json','execution.json','terminal-verification.json']},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),selection=dict(all_common_output_indices=list(range(9)),accepted_nearest_uniform_requested_times_s=np.linspace(T0,END,17).tolist(),total_rows=26,retain_duplicate_states_with_different_labels=True,no_response_dependent_selection=True),physics=dict(cells=[4,2,2],E_Pa=300000.,nu=.35,rho=625.,kappa_N_m3=1e9,bulk='SAMEKeast24',surface='ONLYrootouter64to32;samefrozenroot_surfacekernel/innerGauss3/topology/sin²Jac',guards=141,fullM_force_geometry_unchanged=True),gates=dict(energy_J=1e-12,contact_energy_J=1e-12,gradient_L2_N=1e-10),CPU_affinity_cores=[30,31],hard_cap_s=120,limits=['NoGPUornewtrajectory','Force/EONLY; old32to64HFAILretained,noHclaim','New32constructor explicit; frozen64sourceuntouched','CPUfixedstateparitynottrajectorysurfaceorderstability or spatialadmission'])
 write(OP/'CPU-protocol.json',p);return sha(OP/'CPU-protocol.json')

def run(digest):
 p=json.loads((OP/'CPU-protocol.json').read_text())
 if sha(OP/'CPU-protocol.json')!=digest or sha(CAPTURE)!=EXPECTED or any(sha(ROOT/f)!=h for f,h in {**p['source_sha256'],**p['prior_evidence_sha256']}.items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31}:raise ValueError('CPU2only')
 def cap(*a):raise TimeoutError('CPU120scap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(120);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,states=[])
 try:
  wp.config.kernel_cache_dir=str(OP/'warp-cpu-cache');m=build([4,2,2],300000.,9.81);a=WarpReferenceForce32(m,device='cpu');b=WarpReferenceForce(m,device='cpu')
  for k in ['rest','elements','grad','check_grad','weights','guard_bary']:np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
  with np.load(CAPTURE,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['rest_m'],m.rest);np.testing.assert_array_equal(z['elements'],m.elements);q=z['positions_m'];times=z['times_s'];states=[('common_'+str(i),float(times[i]),np.ascontiguousarray(q[i]-m.rest),None) for i in p['selection']['all_common_output_indices']];t=z['accepted_times_s'];Y=z['accepted_states']
   for j,target in enumerate(p['selection']['accepted_nearest_uniform_requested_times_s']):
    i=int(np.argmin(abs(t-target)));states.append(('accepted_uniform_'+str(j),float(t[i]),np.ascontiguousarray(Y[:m.rest.size,i].reshape(m.rest.shape)),dict(requested_time_s=target,accepted_index=i)))
  for name,t,u,selection in states:
   aa=a.evaluate(u);bb=b.evaluate(u);row=dict(name=name,time_s=t,selection=selection,total_energy_difference_J=abs(aa['energy']-bb['energy']),contact_energy_difference_J=abs(aa['contact_energy']-bb['contact_energy']),gradient_L2_difference_N=float(np.linalg.norm(aa['gradient']-bb['gradient'])),minimum_J_difference=abs(aa['minimum_J']-bb['minimum_J']));g=p['gates'];row['passed']=row['total_energy_difference_J']<=g['energy_J'] and row['contact_energy_difference_J']<=g['contact_energy_J'] and row['gradient_L2_difference_N']<=g['gradient_L2_N'];s['states'].append(row);write(OP/'CPU-result.json',s)
  assert len(s['states'])==26;s.update(completed=True,passed=all(r['passed'] for r in s['states']),maximum_energy_difference_J=max(r['total_energy_difference_J'] for r in s['states']),maximum_contact_energy_difference_J=max(r['contact_energy_difference_J'] for r in s['states']),maximum_gradient_L2_difference_N=max(r['gradient_L2_difference_N'] for r in s['states']),unchanged_bulk_and_all141_J_arrays=True,limitations=p['limits'])
 except Exception as e:s['error']=repr(e);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-start;write(OP/'CPU-result.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r[k] for k in ['completed','passed','elapsed_s']}))
