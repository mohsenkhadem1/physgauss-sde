"""Prospective bounded WarpCPU fixed-state E/g parity; no GPU/dynamics."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_root_contact import WarpRootContact
from experiments.campaign_p2_contact_root_integration import integrate_face
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-warp-root-contact-v01';CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz';EXPECTED='ac850acac4f8906c0b046b813952d07e198736569418f6edc04823b8cbe7f408'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def prepare():
 if (OP/'CPU-protocol.json').exists():raise FileExistsError('Preservefrozenprotocol')
 if sha(CAPTURE)!=EXPECTED:raise ValueError('OnlyexactcompletedQ16capture')
 sources=['experiments/campaign_p2_warp_root_contact.py','tests/test_campaign_p2_warp_root_contact.py','experiments/diagnose_p2_warp_root_contact_cpu.py','experiments/campaign_p2_contact_root_integration.py','experiments/campaign_p2_contact_root_fixed.py']
 p=dict(schema='WarpCPU-rootcontact64-forceenergy-controls-v1',status='FROZEN_BEFORE_SAVED_FIELD_VALUES',source_sha256={f:sha(ROOT/f) for f in sources},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),evidence_sha256={f:sha(ROOT/f) for f in ['runs/operator-dev/p2-contact-root-fixed-v02/protocol.json','runs/operator-dev/p2-contact-root-fixed-v02/result-summary.json']},selection=dict(common_output_indices=[2,3,6],accepted_nearest_time_s=.102),physics=dict(kappa_N_m3=1e9,floor_z_m=0.,reference_metric='2*physicalresttrianglearea',outer_order=64),gates=dict(energy_J=1e-12,gradient_L2_N=1e-10),CPU_threads=2,hard_cap_s=120,interface='E/gONLY:noHessianclaim;inputu=q-rest, compareatACTUALbinary64rest+u forbothimplementations',limits=['NoGPUornewtrajectory','StandaloneNEWsurfaceoperator; oldDuffy/bulk/fullM/integratorunchanged','Force/Eimplementationparityonly; no temporal/spatial/datasetadmission','Fixed32to64HFAILpreserved; no productionHessianclaim'])
 write(OP/'CPU-protocol.json',p);return sha(OP/'CPU-protocol.json')

def run(digest):
 p=json.loads((OP/'CPU-protocol.json').read_text())
 if sha(OP/'CPU-protocol.json')!=digest or sha(CAPTURE)!=EXPECTED or any(sha(ROOT/f)!=h for f,h in {**p['source_sha256'],**p['evidence_sha256']}.items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(os.sched_getaffinity(0))!=2:raise ValueError('CPU2only')
 def cap(*a):raise TimeoutError('FrozenCPU120scap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(120);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,states=[],production_Hessian=False)
 try:
  wp.config.kernel_cache_dir=str(OP/'warp-cpu-cache')
  with np.load(CAPTURE,allow_pickle=False) as z:
   rest=z['rest_m'];faces=z['surface_elements'];times=z['times_s'];q=z['positions_m'];states=[('common_'+str(j),float(times[j]),q[j]) for j in p['selection']['common_output_indices']];t=z['accepted_times_s'];i=int(np.argmin(abs(t-.102)));states.append(('accepted_nearest_.102',float(t[i]),rest+z['accepted_states'][:rest.size,i].reshape(rest.shape)))
  areas=.5*np.linalg.norm(np.cross(rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]]),axis=1);obj=WarpRootContact(rest=rest,surface_elements=faces,reference_areas=areas,kappa_A=1e9,device='cpu')
  for name,t,q in states:
   u=np.ascontiguousarray(q-rest);actualq=rest+u;clock=time.monotonic();actual=obj.evaluate(u);elapsed=time.monotonic()-clock;comparisons=[]
   for kind in ['fixed64','adaptive_original']:
    E=0.;g=np.zeros_like(rest)
    for face,A in zip(faces,areas):
     heights=np.ascontiguousarray(actualq[face,2]);r=integrate_face_fixed(heights,float(A),kappa_A=1e9,outer_order=64) if kind=='fixed64' else integrate_face(heights,float(A),kappa_A=1e9);E+=r['energy_J'];np.add.at(g[:,2],face,r['gradient_N'])
    row=dict(reference=kind,energy_difference_J=abs(actual['contact_energy']-E),gradient_L2_difference_N=float(np.linalg.norm(actual['gradient']-g)),resultant_difference_N=float(np.linalg.norm((actual['gradient']-g).sum(0))),torque_difference_Nm=float(np.linalg.norm(np.cross(actualq,actual['gradient']-g).sum(0))));row['passed']=row['energy_difference_J']<=p['gates']['energy_J'] and row['gradient_L2_difference_N']<=p['gates']['gradient_L2_N'];comparisons.append(row)
   s['states'].append(dict(name=name,time_s=t,WarpCPU_evaluate_s=elapsed,energy_J=actual['contact_energy'],gradient_L2_N=float(np.linalg.norm(actual['gradient'])),binary64_world_roundtrip_max_m=float(abs(actualq-q).max()),comparisons=comparisons,passed=all(r['passed'] for r in comparisons)));write(OP/'CPU-fixedstate-result.json',s)
  s.update(completed=True,passed=all(r['passed'] for r in s['states']),limitations=p['limits'])
 except Exception as e:s['error']=repr(e);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-start;write(OP/'CPU-fixedstate-result.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r[k] for k in ['completed','passed','elapsed_s']}))
