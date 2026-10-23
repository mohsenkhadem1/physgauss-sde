"""NEW explicitly parameterized E/g controls for a different compliant floor.

The frozen old1e9 boundary helpers are unchanged. Known rest-control programs,
NumPy bulk and strict E/g adaptive integration are reused without monkeypatches.
"""
import numpy as np
from experiments.campaign_contact_geometry_controls import manufactured_states,differences
from experiments.campaign_p2_contact_adaptive_energy_gradient import integrate_face_energy_gradient
from experiments.diagnose_p2_contact_bulk_quadrature import response
from experiments.campaign_p2_torch_contact import boundary_trace

def independent_adaptive_response(model,u,arrays,*,kappa_A):
 if not np.isfinite(kappa_A) or kappa_A<=0:raise ValueError('Explicitpositiveknownfloorcompliance required')
 a=arrays;bulk=response(model,u,a['bary'],a['weights'][:,0]/model.volumes[0],a['guard_bary']);faces,areas=boundary_trace(model);g=bulk['gradient_N'].copy();contact=0.;errors=[]
 for i,(face,A) in enumerate(zip(faces,areas)):
  try:r=integrate_face_energy_gradient(np.ascontiguousarray((model.rest+u)[face,2]),float(A),kappa_A=kappa_A)
  except RuntimeError as error:errors.append(dict(face_id=i,error=str(error)));continue
  contact+=r['energy_J'];np.add.at(g[:,2],face,r['gradient_N'])
 if errors:raise RuntimeError('StrictnewflooradaptiveEgfailed; partialsNOTreference:'+str(errors))
 return dict(energy=bulk['energy_J']+contact,contact_energy=contact,gradient=g,minimum_J=min(bulk['integration_minimum_J'],bulk['common_guard_minimum_J']),all_adaptive_faces_completed=True,kappa_A=kappa_A)

def assess_geometry_controls(model,cpu128,cpu64,gpu128=None,*,kappa_A,after_state=None):
 from scipy.linalg import cho_factor,cho_solve
 from experiments.campaign_p2_consistent_transient import scalar_mass
 M=scalar_mass(model);L=cho_factor(M.toarray());rows=[];gravity_error=float(np.linalg.norm(M@np.ones_like(model.rest)*np.array([0.,0.,-9.81])-model.load))
 for name,u in manufactured_states(model.rest):
  row=dict(name=name,passed=False,kappa_A=kappa_A)
  try:
   a=cpu128.evaluate(u);b=cpu64.evaluate(u);r=independent_adaptive_response(model,u,cpu128.arrays,kappa_A=kappa_A);net=model.load-a['gradient'];acc=cho_solve(L,net);mass_R=float(np.linalg.norm(M@acc-net));row.update(CPU128_vs_independent_adaptive_Eg=differences(a,r),CPU64_vs_CPU128=differences(a,b),original_full_mass_R_N=mass_R)
   if gpu128 is not None:row['GPU128_vs_CPU128']=differences(gpu128.evaluate(u),a)
   row['passed']=all(v['passed'] for v in row.values() if isinstance(v,dict) and 'passed' in v) and mass_R<=1e-8 and gravity_error<=1e-12
  except (ValueError,RuntimeError) as error:row['error']=repr(error)
  rows.append(row)
  if after_state:after_state(rows)
 return dict(completed=True,passed=all(x['passed'] for x in rows),controls=rows,gravity_full_mass_identity_R_N=gravity_error,kappa_A=kappa_A,manufactured_only_no_response_arrays=True,no_Hessian_qualification=True)
