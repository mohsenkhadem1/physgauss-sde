"""CPU fixedouter32/64 against ORIGINAL successful adaptive reference; no status2 waiver."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.campaign_p2_contact_root_integration import integrate_face
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-contact-root-fixed-v02';OLD=ROOT/'runs/operator-dev/p2-contact-root-integration-v01'
CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz';EXPECTED='ac850acac4f8906c0b046b813952d07e198736569418f6edc04823b8cbe7f408'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 if sha(CAPTURE)!=EXPECTED:raise ValueError('Exactcaptureonly')
 source=['experiments/campaign_p2_contact_root_integration.py','experiments/campaign_p2_contact_root_fixed.py','tests/test_campaign_p2_contact_root_fixed_v02.py','experiments/diagnose_p2_contact_root_fixed_v02.py']
 p=dict(schema='CPU-rootcontact-fixedouter32to64-originalreference-controls-v2',status='FROZEN_BEFORE_SELECTED_SAVED_STATE_VALUES',source_sha256={f:sha(ROOT/f) for f in source},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),previous_reference_sha256={str((OLD/f).relative_to(ROOT)):sha(OLD/f) for f in ['diagnostic-protocol.json','fixed-state-result.json']},preserved_negative_sha256={str((ROOT/'runs/operator-dev/p2-contact-root-fixed-v01'/f).relative_to(ROOT)):sha(ROOT/'runs/operator-dev/p2-contact-root-fixed-v01'/f) for f in ['protocol.json','result-summary.json','terminal-receipt.json']},selection=dict(common_output_indices=[2,3,6],accepted_nearest_time_s=.102),physics=dict(kappa_N_m3=1e9,floor_z_m=0.,reference_metric='2*restarea'),fixed_outer_orders=[32,64],mapping='lo+(hi-lo)*sin²(pi*t/2);Jac=(hi-lo)*pi/2*sin(pi*t)',adaptive_base=dict(energy_atol_J=1e-12,gradient_atol_N=1e-10,hessian_atol_N_m=1e-7,area_atol_m2=1e-13,rtol=1e-10),adaptive_tight=dict(energy_atol_J=1e-13,gradient_atol_N=1e-11,hessian_atol_N_m=1e-8,area_atol_m2=1e-14,rtol=1e-11),comparison_gates=dict(absolute_energy_J=1e-12,global_gradient_L2_N=1e-10,maximum_global_H_N_m=1e-7),CPU_threads=2,hard_cap_s=120,limits=['NoGPU/physicalcalls/dynamics or modifiedplane law','Compareactualrefinementchanges notonlyheuristicestimates','Noautorelax/autohigherorderif32/64fail; tighteradaptivefailurepreserved;noquad_vecstatus2waiver','Fixedouterandadaptiveouterindependentnumericalrules; sharedexactinnerpolynomial/rootalgebra disclosed'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest or sha(CAPTURE)!=EXPECTED or any(sha(ROOT/f)!=h for f,h in {**p['source_sha256'],**p['previous_reference_sha256'],**p['preserved_negative_sha256']}.items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(os.sched_getaffinity(0))!=2:raise ValueError('CPU2only')
 def stop(*a):raise TimeoutError('FrozenCPU120scap')
 signal.signal(signal.SIGALRM,stop);signal.alarm(120);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,states=[])
 try:
  with np.load(CAPTURE,allow_pickle=False) as z:
   rest=z['rest_m'];faces=z['surface_elements'];times=z['times_s'];q=z['positions_m'];states=[('common_'+str(j),float(times[j]),q[j]) for j in p['selection']['common_output_indices']];t=z['accepted_times_s'];i=int(np.argmin(abs(t-.102)));states.append(('accepted_nearest_.102',float(t[i]),rest+z['accepted_states'][:rest.size,i].reshape(rest.shape)))
  area=.5*np.linalg.norm(np.cross(rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]]),axis=1)
  def assembled(q,kind):
   energy=0.;g=np.zeros(len(rest));H=np.zeros((len(rest),len(rest)));error=np.zeros(3);clock=time.monotonic();evaluations=0
   for face,A in zip(faces,area):
    if isinstance(kind,int):a=integrate_face_fixed(np.ascontiguousarray(q[face,2]),float(A),kappa_A=1e9,outer_order=kind)
    else:a=integrate_face(np.ascontiguousarray(q[face,2]),float(A),kappa_A=1e9,**p[kind])
    energy+=a['energy_J'];np.add.at(g,face,a['gradient_N']);H[np.ix_(face,face)]+=a['hessian_N_m'];evaluations+=a['outer_evaluations']
    if isinstance(kind,str):error+=[a['estimated_energy_error_J'],np.linalg.norm(a['estimated_gradient_errors_N']),np.linalg.norm(a['estimated_hessian_errors_N_m'])]
   return dict(energy=energy,g=g,H=H,estimated_errors=error.tolist() if isinstance(kind,str) else None,elapsed_s=time.monotonic()-clock,outer_evaluations=evaluations,H_min_eigenvalue=float(np.linalg.eigvalsh(H).min()))
  def compare(a,b):
   row=dict(energy_difference_J=abs(a['energy']-b['energy']),global_gradient_L2_difference_N=float(np.linalg.norm(a['g']-b['g'])),maximum_global_H_difference_N_m=float(abs(a['H']-b['H']).max()));g=p['comparison_gates'];row['passed']=row['energy_difference_J']<=g['absolute_energy_J'] and row['global_gradient_L2_difference_N']<=g['global_gradient_L2_N'] and row['maximum_global_H_difference_N_m']<=g['maximum_global_H_N_m'];return row
  for name,t,q in states:
   a=assembled(q,'adaptive_base');f32=assembled(q,32);f64=assembled(q,64)
   row=dict(name=name,time_s=t,fixed32_to_fixed64=compare(f32,f64),fixed64_to_original_adaptive=compare(f64,a),fixed32_to_original_adaptive=compare(f32,a),methods={k:{kk:vv for kk,vv in v.items() if kk not in ['g','H']} for k,v in [('adaptive_base',a),('fixed32',f32),('fixed64',f64)]});row['passed']=all(row[k]['passed'] for k in ['fixed32_to_fixed64','fixed64_to_original_adaptive','fixed32_to_original_adaptive']);s['states'].append(row);write(OP/'result-summary.json',s)
  s.update(completed=True,passed=all(r['passed'] for r in s['states']),limitations=p['limits'])
 except Exception as e:s['error']=repr(e);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-start;write(OP/'result-summary.json',s)
 return s
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--sha');a=a.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else dict(completed=r['completed'],passed=r['passed'],elapsed_s=r['elapsed_s'])))
