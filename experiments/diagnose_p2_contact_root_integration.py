"""Bounded CPU-only fixed-state contact integration diagnostic; no dynamics."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_contact_root_integration import integrate_face,triangle_shapes
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-contact-root-integration-v01'
CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz'
EXPECTED='ac850acac4f8906c0b046b813952d07e198736569418f6edc04823b8cbe7f408'
SOURCES=['experiments/campaign_p2_contact_root_integration.py','tests/test_campaign_p2_contact_root_integration.py','experiments/diagnose_p2_contact_root_integration.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def prepare():
 if (OP/'diagnostic-protocol.json').exists():raise FileExistsError('Preservefrozenprotocol')
 if sha(CAPTURE)!=EXPECTED:raise ValueError('ExactcompletedQ16capturerequired')
 p=dict(schema='CPU-fixedfield-rootcontact-diagnostic-v1',status='FROZEN_BEFORE_SAVED_FIELD_VALUES',source_sha256={f:sha(ROOT/f) for f in SOURCES},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=EXPECTED),selection=dict(common_output_indices=[2,3,6],accepted_nearest_time_s=.102,no_response_dependent_selection=True),physics=dict(kappa_N_m3=1e9,floor_z_m=0.,reference_metric='2*physicalresttrianglearea',bulk_mass_state_unchanged=True),integration=dict(inner='quadraticroots splitnegatives intervals;3pointGauss exactly integratesquartic on eachactiveinterval',outer='adaptiveGaussKronrodquad_vec, edge/discriminanttopologybreakpoints',energy_atol_J=1e-12,gradient_atol_N=1e-10,hessian_atol_N_m=1e-7,area_atol_m2=1e-13,rtol=1e-10),comparison_Q=[8,16],CPU_threads=2,hard_cap_s=120,limits=['SAMEsavedfields only; no dynamic/sensor/likelihood/GPU queries','Adaptiveerror estimates are not rigorousbounds','NoQ32/newcontacttrajectory or admission','SavedQ16 itselffailedsurfaceconvergence, nottruth'])
 write(OP/'diagnostic-protocol.json',p);return sha(OP/'diagnostic-protocol.json')

def duffy(q):
 x,w=leggauss(q);x=(x+1)/2;w=w/2;r,s=np.meshgrid(x,x,indexing='ij');wa,wb=np.meshgrid(w,w,indexing='ij');return triangle_shapes(r.ravel(),((1-r)*s).ravel()),(wa*wb*(1-r)).ravel()

def run(digest):
 p=json.loads((OP/'diagnostic-protocol.json').read_text())
 if sha(OP/'diagnostic-protocol.json')!=digest or sha(CAPTURE)!=p['capture']['sha256'] or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedfrozeninputs')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(os.sched_getaffinity(0))!=2:raise ValueError('CPU-only2coresrequired')
 def cap(*a):raise TimeoutError('Frozen120sCPUdiagnosticcap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(120);start=time.monotonic();report=dict(completed=False,passed=False,protocol_sha256=digest,states=[])
 try:
  with np.load(CAPTURE,allow_pickle=False) as z:
   rest=z['rest_m'];faces=z['surface_elements'];q=z['positions_m'];times=z['times_s'];states=[(f'common_{i}',float(times[i]),q[i].copy()) for i in p['selection']['common_output_indices']];t=z['accepted_times_s'];index=int(np.argmin(abs(t-.102)));positions=rest+z['accepted_states'][:rest.size,index].reshape(rest.shape);states.append(('accepted_nearest_.102',float(t[index]),positions))
  area=.5*np.linalg.norm(np.cross(rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]]),axis=1);settings=p['integration'];settings={k:settings[k] for k in ['energy_atol_J','gradient_atol_N','hessian_atol_N_m','area_atol_m2','rtol']};rules={Q:duffy(Q) for Q in p['comparison_Q']}
  for name,t,q in states:
   E=0.;g=np.zeros_like(rest);errors=np.zeros(2);calls=0;maximumintervals=0;activearea=0.;clock=time.monotonic()
   for face,A in zip(faces,area):
    a=integrate_face(np.ascontiguousarray(q[face,2]),float(A),kappa_A=1e9,**settings);E+=a['energy_J'];np.add.at(g[:,2],face,a['gradient_N']);errors[0]+=a['estimated_energy_error_J'];errors[1]+=float(np.linalg.norm(a['estimated_gradient_errors_N']));calls+=a['outer_evaluations'];maximumintervals=max(maximumintervals,a['outer_intervals']);activearea+=a['active_reference_area_m2']
   comparisons=[]
   for Q,(N,w) in rules.items():
    z=N@q[faces,2].T;negative=np.minimum(z,0.);weights=2*area[None]*w[:,None];oldE=float(.5e9*np.sum(weights*negative**2));local=1e9*np.einsum('qf,qf,qa->fa',weights,negative,N);oldg=np.zeros_like(rest);np.add.at(oldg[:,2],faces.ravel(),local.ravel());comparisons.append(dict(Q=Q,energy_J=oldE,absolute_energy_difference_J=abs(oldE-E),gradient_L2_difference_N=float(np.linalg.norm(oldg-g)),maximum_nodal_gradient_difference_N=float(abs(oldg-g).max()),upward_contact_resultant_N=float(-oldg.sum(0)[2])))
   report['states'].append(dict(name=name,time_s=t,reference_energy_J=E,reference_upward_contact_resultant_N=float(-g.sum(0)[2]),reference_gradient_L2_N=float(np.linalg.norm(g)),reference_gradient_sum_N=float(g.sum()),reference_active_area_m2=activearea,estimated_total_energy_error_J=float(errors[0]),estimated_global_gradient_L2_error_bound_estimate_N=float(errors[1]),outer_integrand_calls=calls,maximum_adaptive_intervals_perface=maximumintervals,CPU_elapsed_s=time.monotonic()-clock,comparisons=comparisons));write(OP/'fixed-state-result.json',report)
  report.update(completed=True,passed=True,no_physical_runs=True,limitations=p['limits'])
 except Exception as e:report['error']=repr(e);raise
 finally:signal.alarm(0);report['elapsed_s']=time.monotonic()-start;write(OP/'fixed-state-result.json',report)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else dict(completed=r['completed'],passed=r['passed'],elapsed_s=r['elapsed_s'])))
