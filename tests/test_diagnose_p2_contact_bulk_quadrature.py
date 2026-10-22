import numpy as np
from experiments.diagnose_p2_contact_bulk_quadrature import build,response
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule

def test_degree6_monomial_and_manufactured_P2_bulk_24vs80_and_affine_patch():
 m=build([1,1,1],300000.,9.81);rules=[(m.bary,m.weights),positive_keast24(),exact_degree6_rule()];exact=1/84
 assert abs(np.dot(rules[1][1],rules[1][0][:,1]**6)-exact)<1e-15
 assert abs(np.dot(rules[2][1],rules[2][0][:,1]**6)-exact)<1e-15
 assert abs(np.dot(rules[0][1],rules[0][0][:,1]**6)-exact)>1e-6
 X=m.rest;u=np.column_stack([.4*X[:,0]**2,.5*X[:,1]**2,.6*(X[:,2]-.09)**2]);checks=np.concatenate([m.check_bary]+[b for b,w in rules]);r=[response(m,u,b,w,checks) for b,w in rules]
 assert abs(r[1]['energy_J']-r[2]['energy_J'])<1e-12
 assert np.linalg.norm(r[1]['gradient_N']-r[2]['gradient_N'])<1e-10
 assert np.linalg.norm(r[0]['gradient_N']-r[2]['gradient_N'])>1e-5
 affine=np.column_stack([.04*X[:,0],-.02*X[:,1],.03*X[:,2]]);a=[response(m,affine,b,w,checks) for b,w in rules]
 for x in a[1:]:
  assert abs(x['energy_J']-a[0]['energy_J'])<1e-12
  assert np.linalg.norm(x['gradient_N']-a[0]['gradient_N'])<1e-10
 assert all(np.linalg.norm(x['internal_resultant_N'])<1e-11 for x in r)
