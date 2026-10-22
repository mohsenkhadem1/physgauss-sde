"""CPU-only fixed-field Q3/Keast24/exact80 bulk diagnostic, no dynamics."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from experiments.run_p2_contact_validation import build
from experiments.campaign_static_p2_block import shape_gradients,duffy
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule
from experiments.campaign_newton_stable_material import stable_material
from experiments.campaign_p2_consistent_transient import scalar_mass
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-contact-bulk-quadrature-v01';CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz';EXPECTED='ac850acac4f8906c0b046b813952d07e198736569418f6edc04823b8cbe7f408'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def response(m,u,bary,weights,checks):
 u=np.asarray(u)
 if u.dtype!=np.float64 or u.shape!=m.rest.shape or not np.isfinite(u).all():raise ValueError('FinitecompatibleFP64displacementrequired')
 energy=0.;g=np.zeros_like(u);minimum=1.;checkminimum=1.
 for b,w in zip(bary,weights):
  grad=shape_gradients(b[None],m.linear_gradients);F=np.eye(3)+np.einsum('eai,eaj->eij',u[m.elements],grad);J=np.linalg.det(F)
  if not np.isfinite(J).all() or np.any(J<=m.minimum_J):raise ValueError('OriginalpositiveJguardfailed')
  W,P,_=stable_material(F,m.young,m.poisson,False);energy+=float(np.dot(m.volumes*w,W));local=np.einsum('e,eij,eaj->eai',m.volumes*w,P,grad);np.add.at(g,m.elements,local);minimum=min(minimum,float(J.min()))
 for b in checks:
  grad=shape_gradients(b[None],m.linear_gradients);J=np.linalg.det(np.eye(3)+np.einsum('eai,eaj->eij',u[m.elements],grad))
  if not np.isfinite(J).all() or np.any(J<=m.minimum_J):raise ValueError('CommonretainedpositiveJguardfailed')
  checkminimum=min(checkminimum,float(J.min()))
 return dict(energy_J=energy,gradient_N=g,integration_minimum_J=minimum,common_guard_minimum_J=checkminimum,internal_resultant_N=np.sum(g,axis=0))

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 if sha(CAPTURE)!=EXPECTED:raise ValueError('ExactcompletedQ16captureonly')
 sources=['experiments/diagnose_p2_contact_bulk_quadrature.py','tests/test_diagnose_p2_contact_bulk_quadrature.py','experiments/run_p2_contact_validation.py','experiments/campaign_static_p2_block.py','experiments/campaign_newton_stable_material.py','experiments/campaign_p2_keast_quadrature.py','experiments/campaign_p2_exact_quadrature.py','experiments/campaign_p2_consistent_transient.py']
 p=dict(schema='CPU-fixed4state-bulkQ3-24-80-diagnostic-v1',status='FROZEN_BEFORE_SAVED_FIELDS',source_sha256={f:sha(ROOT/f) for f in sources},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),selection=dict(common_output_indices=[2,3,6],accepted_nearest_time_s=.102),physics=dict(cells=[4,2,2],young_Pa=300000.,nu=.35,rho=625.,gravity_m_s2=[0.,0.,-9.81],all_free=True,stable_NH_constant_material=True,reference_geometry_forces_fullM_unchanged=True),rules=['originalDuffy3x3x3_27','positiveKeast24_degree6','positiveDuffy5x4x4_80_degree6'],checks='UNIONoriginalQ3+old80+original10nodes+new24; eachruleintegrationJalsoreported',degree6_comparison_gates=dict(energy_J=1e-12,global_gradient_L2_N=1e-10),CPU_threads=2,CPU_affinity_cores=[30,31],hard_cap_s=120,limits=['NoGPUornewtrajectory','Q3→24differencequantifiednotretroactivetimefailure:oldQ3timechecksvalidONLYitsdiscreteoperator','Nophysicalmaterial/force/fullMchange; quadraturebiasseparatelylabelled','Exact24/80claimonlyconstantpolynomialstableNHcompatibleP2bulkinreal-arithmetic','NoGPUoperatoradoption/surface/spatial/datasetadmission'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest or sha(CAPTURE)!=EXPECTED or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('CPU2onlyfixedcores')
 def cap(*a):raise TimeoutError('Frozen120sCPUcap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(120);start=time.monotonic();s=dict(completed=False,degree6_rules_match=False,protocol_sha256=digest,states=[])
 try:
  m=build([4,2,2],300000.,9.81);M=scalar_mass(m);mass_hash=hashlib.sha256(M.data.tobytes()+M.indices.tobytes()+M.indptr.tobytes()).hexdigest();rest=m.rest.copy();elements=m.elements.copy();force=m.load.copy();rules={'Q3':(m.bary.copy(),m.weights.copy()),'Keast24':positive_keast24(),'Exact80':exact_degree6_rule()};checks=np.concatenate([m.check_bary]+[rules[k][0] for k in rules]);np.testing.assert_allclose(M@np.ones_like(m.rest)*np.array([0.,0.,-9.81]),m.load,atol=1e-13,rtol=1e-13)
  with np.load(CAPTURE,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['rest_m'],m.rest);np.testing.assert_array_equal(z['elements'],m.elements);times=z['times_s'];q=z['positions_m'];states=[('common_'+str(i),float(times[i]),np.ascontiguousarray(q[i]-rest)) for i in p['selection']['common_output_indices']];t=z['accepted_times_s'];i=int(np.argmin(abs(t-.102)));states.append(('accepted_nearest_.102',float(t[i]),np.ascontiguousarray(z['accepted_states'][:rest.size,i].reshape(rest.shape))))
  for name,t,u in states:
   outputs={label:response(m,u,b,w,checks) for label,(b,w) in rules.items()};comparisons=[]
   for a,b in [('Q3','Exact80'),('Keast24','Exact80'),('Q3','Keast24')]:
    aa=outputs[a];bb=outputs[b];row=dict(a=a,b=b,absolute_energy_difference_J=abs(aa['energy_J']-bb['energy_J']),global_gradient_L2_difference_N=float(np.linalg.norm(aa['gradient_N']-bb['gradient_N'])),maximum_nodal_gradient_difference_N=float(np.linalg.norm(aa['gradient_N']-bb['gradient_N'],axis=1).max()),relative_energy_difference=abs(aa['energy_J']-bb['energy_J'])/max(abs(bb['energy_J']),1e-30),relative_gradient_L2_difference=float(np.linalg.norm(aa['gradient_N']-bb['gradient_N']))/max(float(np.linalg.norm(bb['gradient_N'])),1e-30));row['within_degree6_parity_gates']=row['absolute_energy_difference_J']<=1e-12 and row['global_gradient_L2_difference_N']<=1e-10;comparisons.append(row)
   s['states'].append(dict(name=name,time_s=t,methods={label:dict(energy_J=r['energy_J'],gradient_L2_N=float(np.linalg.norm(r['gradient_N'])),integration_minimum_J=r['integration_minimum_J'],common_guard_minimum_J=r['common_guard_minimum_J'],internal_resultant_N=r['internal_resultant_N'].tolist()) for label,r in outputs.items()},comparisons=comparisons));write(OP/'result-summary.json',s)
  np.testing.assert_array_equal(m.rest,rest);np.testing.assert_array_equal(m.elements,elements);np.testing.assert_array_equal(m.load,force);Mafter=scalar_mass(m);assert hashlib.sha256(Mafter.data.tobytes()+Mafter.indices.tobytes()+Mafter.indptr.tobytes()).hexdigest()==mass_hash;s.update(completed=True,degree6_rules_match=all(r['comparisons'][1]['within_degree6_parity_gates'] for r in s['states']),unchanged_rest_elements_forces_fullM=True,fullM_sha256=mass_hash,common_guard_points=len(checks),limitations=p['limits'])
 except Exception as e:s['error']=repr(e);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-start;write(OP/'result-summary.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r[k] for k in ['completed','degree6_rules_match','elapsed_s']}))
