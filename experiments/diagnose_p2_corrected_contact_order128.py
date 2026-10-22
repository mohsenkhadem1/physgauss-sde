"""CPU-only root64 vs fixed128 vs successful adaptive plane, SAME26 states."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
from experiments.campaign_p2_contact_root_integration import integrate_face
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.diagnose_p2_contact_bulk_quadrature import response
from experiments.run_p2_contact_validation import build
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-corrected-contact-order128-v01';PREVIOUS=ROOT/'runs/operator-dev/p2-corrected-contact-order32-v01';CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-reference-v01/accuracy_1e10/trajectory.npz';EXPECTED='633ba220edf8f4d5cfb87adb7e6e7db54a74a6bd321270b4e2e454801c314d1c'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def compare(a,b):
 r=dict(energy_difference_J=abs(a['energy']-b['energy']),contact_energy_difference_J=abs(a['contact_energy']-b['contact_energy']),gradient_L2_difference_N=float(np.linalg.norm(a['gradient']-b['gradient'])));r['passed']=r['energy_difference_J']<=1e-12 and r['contact_energy_difference_J']<=1e-12 and r['gradient_L2_difference_N']<=1e-10;return r

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 if sha(CAPTURE)!=EXPECTED:raise ValueError('Exactqualifiedtight64captureonly')
 old=json.loads((PREVIOUS/'CPU-protocol.json').read_text());sources=['experiments/diagnose_p2_corrected_contact_order128.py','tests/test_diagnose_p2_corrected_contact_order128.py','experiments/campaign_p2_warp_reference_force.py','experiments/campaign_p2_contact_root_integration.py','experiments/campaign_p2_contact_root_fixed.py','experiments/diagnose_p2_contact_bulk_quadrature.py','experiments/campaign_p2_warp_root_contact.py','experiments/campaign_p2_warp_rhs.py']
 p=dict(schema='CPU-root64-128-adaptive-same26states-v1',status='FROZEN_BEFORE_NEW_REFERENCE_VALUES',source_sha256={f:sha(ROOT/f) for f in sources},preserved_negative_sha256={str((PREVIOUS/f).relative_to(ROOT)):sha(PREVIOUS/f) for f in ['CPU-protocol.json','CPU-result.json']},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),selection=old['selection'],physics=old['physics'],gates=old['gates'],reference='Originalsuccessfuladaptivecomponenttolerances unchanged; no status2 waiver; fixed128endpointmappedouter withsameexactinner, notDuffy128',CPU_affinity_cores=[30,31],hard_cap_s=180,limits=['ALL26states retained evenifindividualadaptivefails; failureexplicitandaggregatefalse','NoGPU/newtrajectory/Hessianproductionclaim','64/128same-fieldorder/crossrulecheckONLY, nottrajectorysurface/spatialadmission','Old32strictforceFAILandoldHFAILpreserved'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest or sha(CAPTURE)!=EXPECTED or any(sha(ROOT/f)!=h for f,h in {**p['source_sha256'],**p['preserved_negative_sha256']}.items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31}:raise ValueError('CPU2only')
 def cap(*a):raise TimeoutError('CPU180scap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(180);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,states=[])
 try:
  wp.config.kernel_cache_dir=str(PREVIOUS/'warp-cpu-cache');m=build([4,2,2],300000.,9.81);obj=WarpReferenceForce(m,device='cpu');a=obj.arrays;faces=obj.surface.faces.numpy();areas=obj.surface.areas.numpy()
  with np.load(CAPTURE,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['rest_m'],m.rest);q=z['positions_m'];times=z['times_s'];states=[('common_'+str(i),float(times[i]),np.ascontiguousarray(q[i]-m.rest),None) for i in p['selection']['all_common_output_indices']];t=z['accepted_times_s'];Y=z['accepted_states']
   for j,target in enumerate(p['selection']['accepted_nearest_uniform_requested_times_s']):
    i=int(np.argmin(abs(t-target)));states.append(('accepted_uniform_'+str(j),float(t[i]),np.ascontiguousarray(Y[:m.rest.size,i].reshape(m.rest.shape)),dict(requested_time_s=target,accepted_index=i)))
  for name,t,u,selection in states:
   begin=time.monotonic();actual=obj.evaluate(u);bulk=response(m,u,a['bary'],a['weights'][:,0]/m.volumes[0],a['guard_bary']);baseE=bulk['energy_J'];baseg=bulk['gradient_N'];positions=m.rest+u;fixed=dict(energy=baseE,contact_energy=0.,gradient=baseg.copy());adaptive=dict(energy=baseE,contact_energy=0.,gradient=baseg.copy());errors=[]
   for faceid,(face,A) in enumerate(zip(faces,areas)):
    z=np.ascontiguousarray(positions[face,2]);f=integrate_face_fixed(z,float(A),kappa_A=1e9,outer_order=128);fixed['energy']+=f['energy_J'];fixed['contact_energy']+=f['energy_J'];np.add.at(fixed['gradient'][:,2],face,f['gradient_N'])
    try:r=integrate_face(z,float(A),kappa_A=1e9)
    except RuntimeError as error:errors.append(dict(face_id=faceid,error=str(error)));continue
    adaptive['energy']+=r['energy_J'];adaptive['contact_energy']+=r['energy_J'];np.add.at(adaptive['gradient'][:,2],face,r['gradient_N'])
   row=dict(name=name,time_s=t,selection=selection,fixed64_to128=compare(actual,fixed),adaptive_reference_completed=not errors,adaptive_errors=errors,elapsed_s=time.monotonic()-begin)
   if not errors:row.update(fixed64_to_adaptive=compare(actual,adaptive),fixed128_to_adaptive=compare(fixed,adaptive));row['passed']=all(row[k]['passed'] for k in ['fixed64_to128','fixed64_to_adaptive','fixed128_to_adaptive'])
   else:row['passed']=False;row['adaptive_partial_sums_NOT_REFERENCE']=True
   s['states'].append(row);write(OP/'result-summary.json',s)
  assert len(s['states'])==26;s.update(completed=True,passed=all(r['passed'] for r in s['states']),all26_retained=True,limitations=p['limits'])
 except Exception as error:s['error']=repr(error);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-start;write(OP/'result-summary.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r[k] for k in ['completed','passed','elapsed_s']}))
