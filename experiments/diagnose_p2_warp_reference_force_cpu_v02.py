"""Corrected combined reference force: hash-bound CPU E/g/J/fullRHS checks."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
from experiments.run_p2_contact_validation import build
from experiments.diagnose_p2_contact_bulk_quadrature import response
from experiments.campaign_p2_contact_root_integration import integrate_face
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_static_p2_block import shape_gradients
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-warp-reference-force-v01';CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz';EXPECTED='ac850acac4f8906c0b046b813952d07e198736569418f6edc04823b8cbe7f408'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def actual_reference_J(m,u,checks):
 minimum=float("inf")
 for b in checks:
  g=shape_gradients(b[None],m.linear_gradients);F=np.eye(3)+np.einsum("eai,eaj->eij",u[m.elements],g);minimum=min(minimum,float(np.linalg.det(F).min()))
 return minimum

def prepare():
 if (OP/'CPU-protocol-v02.json').exists():raise FileExistsError('Preserveprotocol')
 if sha(CAPTURE)!=EXPECTED:raise ValueError('Exactcaptureonly')
 sources=['experiments/campaign_p2_warp_reference_force.py','tests/test_campaign_p2_warp_reference_force_v02.py','experiments/diagnose_p2_warp_reference_force_cpu_v02.py','experiments/campaign_p2_warp_rhs.py','experiments/campaign_p2_warp_root_contact.py','experiments/campaign_p2_torch_contact.py','experiments/campaign_p2_keast_quadrature.py','experiments/campaign_p2_exact_quadrature.py','experiments/diagnose_p2_contact_bulk_quadrature.py','experiments/campaign_p2_contact_root_integration.py','experiments/campaign_p2_consistent_transient.py','experiments/run_p2_contact_validation.py']
 p=dict(schema='correctedbulk24-root64-combined-CPU4state-v2',status='FROZEN_BEFORE_SAVED_FIELDS',J_reference='ActualNumPyminimuminitializedinf,notcappedat1; originalJ>.1guardunchanged',superseded_unexecuted_protocol=dict(path='runs/operator-dev/p2-warp-reference-force-v01/CPU-protocol.json',sha256=sha(OP/'CPU-protocol.json')),source_sha256={f:sha(ROOT/f) for f in sources},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),selection=dict(common_output_indices=[2,3,6],accepted_nearest_time_s=.102),physics=dict(cells=[4,2,2],E_Pa=300000.,nu=.35,rho=625.,kappa_N_m3=1e9,gravity_m_s2=[0.,0.,-9.81],bulk_rule='Keast24',surface_rule='root64',guards=141,fullM_unchanged=True,force_unchanged=True),independent_reference='NumPystableW/PbulkKeast24+adaptiveCPUplane;TorchCPUfullMCholeskyRHSvsSciPyfullMCholeskyreference',gates=dict(energy_J=1e-12,gradient_L2_N=1e-10,minimum_J_difference=2e-14,consistent_mass_R_N=1e-8),CPU_affinity_cores=[30,31],hard_cap_s=120,limits=['ExplicitTWOquadraturecorrectionsvsoldQ3/Duffy; nooldsourceedit','NoGPU/dynamiccalls; no temporal/spatial/surfaceadmission','OldQ3/Duffycaptures/tolerancechecksretainoriginalscope','NoHessianproductionclaim; unchangedconstantmaterial/law/kappa/fullM/bodyforce/initializer'])
 write(OP/'CPU-protocol-v02.json',p);return sha(OP/'CPU-protocol-v02.json')

def run(digest):
 p=json.loads((OP/'CPU-protocol-v02.json').read_text())
 if sha(OP/'CPU-protocol-v02.json')!=digest or sha(CAPTURE)!=EXPECTED or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31}:raise ValueError('CPU2only')
 def cap(*a):raise TimeoutError('CPU120scap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(120);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,states=[])
 try:
  import torch
  torch.set_num_threads(2);wp.config.kernel_cache_dir=str(OP/'warp-cpu-cache');m=build([4,2,2],300000.,9.81);rest=m.rest.copy();elems=m.elements.copy();force=m.load.copy();M=scalar_mass(m).toarray();Mt=torch.tensor(M,dtype=torch.float64);Lt=torch.linalg.cholesky(Mt);L=cho_factor(M);obj=WarpReferenceForce(m,device='cpu');arr=obj.arrays;faces=obj.surface.faces.numpy();areas=obj.surface.areas.numpy();np.testing.assert_allclose(M@np.ones_like(rest)*np.array([0.,0.,-9.81]),force,atol=1e-13,rtol=1e-13)
  with np.load(CAPTURE,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['rest_m'],rest);np.testing.assert_array_equal(z['elements'],elems);q=z['positions_m'];times=z['times_s'];states=[('common_'+str(i),float(times[i]),np.ascontiguousarray(q[i]-rest)) for i in p['selection']['common_output_indices']];t=z['accepted_times_s'];i=int(np.argmin(abs(t-.102)));states.append(('accepted_nearest_.102',float(t[i]),np.ascontiguousarray(z['accepted_states'][:rest.size,i].reshape(rest.shape))))
  for name,t,u in states:
   actual=obj.evaluate(u);ref=response(m,u,arr['bary'],arr['weights'][:,0]/m.volumes[0],arr['guard_bary']);E=ref['energy_J'];g=ref['gradient_N'].copy();q=rest+u
   for face,A in zip(faces,areas):r=integrate_face(np.ascontiguousarray(q[face,2]),float(A),kappa_A=1e9);E+=r['energy_J'];np.add.at(g[:,2],face,r['gradient_N'])
   net=force-actual['gradient'];acc=torch.cholesky_solve(torch.tensor(net,dtype=torch.float64),Lt).numpy();R=float(np.linalg.norm(M@acc-net));refacc=cho_solve(L,force-g);refR=float(np.linalg.norm(M@refacc-(force-g)));row=dict(name=name,time_s=t,absolute_energy_difference_J=abs(actual['energy']-E),gradient_L2_difference_N=float(np.linalg.norm(actual['gradient']-g)),minimum_J_difference=abs(actual['minimum_J']-actual_reference_J(m,u,arr['guard_bary'])),minimum_J=actual['minimum_J'],Torch_consistent_mass_R_N=R,SciPy_reference_consistent_mass_R_N=refR,fullRHS_acceleration_L2_difference_m_s2=float(np.linalg.norm(acc-refacc)),reference_energy_J=E,CPU_bulk_Warp_energy_J=actual['elastic_energy'],CPU_surface_Warp_energy_J=actual['contact_energy']);gate=p['gates'];row['passed']=row['absolute_energy_difference_J']<=gate['energy_J'] and row['gradient_L2_difference_N']<=gate['gradient_L2_N'] and row['minimum_J_difference']<=gate['minimum_J_difference'] and max(R,refR)<=gate['consistent_mass_R_N'];s['states'].append(row);write(OP/'CPU-fixedstate-v02-result.json',s)
  np.testing.assert_array_equal(m.rest,rest);np.testing.assert_array_equal(m.elements,elems);np.testing.assert_array_equal(m.load,force);np.testing.assert_array_equal(scalar_mass(m).toarray(),M);s.update(completed=True,passed=all(r['passed'] for r in s['states']),unchanged_rest_elements_forces_fullM=True,guard_points=141,limitations=p['limits'])
 except Exception as e:s['error']=repr(e);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-start;write(OP/'CPU-fixedstate-v02-result.json',s)
 return s
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--sha');a=a.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r[k] for k in ['completed','passed','elapsed_s']}))
