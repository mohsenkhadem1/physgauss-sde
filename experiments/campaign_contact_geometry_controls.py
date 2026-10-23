"""New-geometry manufactured controls; no saved response/state inputs."""
import numpy as np
from experiments.campaign_p2_contact_adaptive_energy_gradient import integrate_face_energy_gradient as integrate_face
from experiments.diagnose_p2_contact_bulk_quadrature import response
from experiments.campaign_p2_torch_contact import boundary_trace

CONTROL_NAMES=('analytic_uniform_freefall_T0','smooth_small_noncontact','uniform_slight_penetration','smooth_small_partial_contact')

def manufactured_states(rest):
 X=np.asarray(rest)
 if X.ndim!=2 or X.shape[1]!=3 or X.dtype!=np.float64 or not np.isfinite(X).all():raise ValueError('Finite FP64 public rest required')
 x=(X[:,0]-X[:,0].min())/np.ptp(X[:,0]);y=(X[:,1]-X[:,1].min())/np.ptp(X[:,1]);z0=float(X[:,2].min());states=[]
 u=np.zeros_like(X);u[:,2]=-.5*9.81*(1/12)**2;states.append(u)
 u=np.column_stack([1e-5*np.sin(np.pi*x),8e-6*np.sin(2*np.pi*y),-5e-6*np.cos(np.pi*x)*np.cos(np.pi*y)]);states.append(u)
 u=np.zeros_like(X);u[:,2]=-z0-1e-5;states.append(u)
 u=np.column_stack([2e-5*x*(1-x),-1e-5*(y-.5),-z0+2e-5*(2*x-1)]);states.append(u)
 return list(zip(CONTROL_NAMES,[np.ascontiguousarray(u) for u in states]))

def independent_adaptive_response(model,u,arrays):
 """Independent NumPy bulk + adaptive root-split surface at original tolerances.

 Bulk uses the same declared Keast24/141 rules but independent array arithmetic.
 Surface uses NEW strict E/g-only adaptive outer integration, no H claim.
 """
 a=arrays;bulk=response(model,u,a['bary'],a['weights'][:,0]/model.volumes[0],a['guard_bary']);faces,areas=boundary_trace(model);g=bulk['gradient_N'].copy();contact=0.;errors=[]
 for i,(face,A) in enumerate(zip(faces,areas)):
  try:r=integrate_face(np.ascontiguousarray((model.rest+u)[face,2]),float(A),kappa_A=1e9)
  except RuntimeError as error:errors.append(dict(face_id=i,error=str(error)));continue
  contact+=r['energy_J'];np.add.at(g[:,2],face,r['gradient_N'])
 if errors:raise RuntimeError('Independent adaptive plane failed; partial sums NOT reference: '+str(errors))
 return dict(energy=bulk['energy_J']+contact,contact_energy=contact,gradient=g,minimum_J=min(bulk['integration_minimum_J'],bulk['common_guard_minimum_J']),all_adaptive_faces_completed=True)

def differences(a,b):
 d=dict(E_J=abs(a['energy']-b['energy']),g_L2_N=float(np.linalg.norm(a['gradient']-b['gradient'])),J_difference=abs(a['minimum_J']-b['minimum_J']))
 d['passed']=d['E_J']<=1e-12 and d['g_L2_N']<=1e-10 and d['J_difference']<=2e-14
 return d

def assess_geometry_controls(model,cpu128,cpu64,gpu128=None,after_state=None):
 from scipy.linalg import cho_factor,cho_solve
 from experiments.campaign_p2_consistent_transient import scalar_mass
 M=scalar_mass(model);L=cho_factor(M.toarray());rows=[]
 gravity_error=float(np.linalg.norm(M@np.ones_like(model.rest)*np.array([0.,0.,-9.81])-model.load))
 for name,u in manufactured_states(model.rest):
  row=dict(name=name,passed=False)
  try:
   a=cpu128.evaluate(u);b=cpu64.evaluate(u);r=independent_adaptive_response(model,u,cpu128.arrays);net=model.load-a['gradient'];acc=cho_solve(L,net);mass_R=float(np.linalg.norm(M@acc-net));row.update(CPU128_vs_independent_adaptive_Eg=differences(a,r),CPU64_vs_CPU128=differences(a,b),original_full_mass_R_N=mass_R)
   if gpu128 is not None:row['GPU128_vs_CPU128']=differences(gpu128.evaluate(u),a)
   row['passed']=all(v['passed'] for v in row.values() if isinstance(v,dict) and 'passed' in v) and mass_R<=1e-8 and gravity_error<=1e-12
  except (ValueError,RuntimeError) as error:row['error']=repr(error)
  rows.append(row)
  if after_state:after_state(rows)
 return dict(completed=True,passed=all(x['passed'] for x in rows),controls=rows,gravity_full_mass_identity_R_N=gravity_error,manufactured_only_no_response_arrays=True,no_Hessian_qualification=True,CPU64_128_surface_control_fixed_states_only=True)
