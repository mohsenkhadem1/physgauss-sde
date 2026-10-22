"""CPU-only manufactured contact-law integration; no campaign fields."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_contact_root_integration import integrate_face,triangle_shapes,real_roots,height_coefficients,topology_events
NODES=np.array([[0.,0.],[1,0],[0,1],[.5,0],[0,.5],[.5,.5]])

def nodal(fun):return np.asarray(fun(NODES[:,0],NODES[:,1]),dtype=np.float64)

def independent_affine_rule(a,area,kappa,order=14):
 x,w=leggauss(order);x=(x+1)/2;w=w/2;rs=[];ws=[]
 for i,r0 in enumerate(x):
  r=a*r0
  for j,t in enumerate(x):rs.append([r,(1-r)*t]);ws.append(2*area*a*(1-r)*w[i]*w[j])
 rs=np.array(rs);ws=np.array(ws);N=triangle_shapes(rs[:,0],rs[:,1]);z=rs[:,0]-a
 return .5*kappa*np.dot(ws,z*z),kappa*N.T@(ws*z),kappa*np.einsum('q,qa,qb->ab',ws,N,N),rs,ws,z

def test_inactive_fullactive_constant_nearlinear_and_zero_nonsmoothH():
 area=.03;k=1500.
 for height in [0.,.01,-.001]:
  r=integrate_face(np.full(6,height),area,kappa_A=k)
  assert abs(r['energy_J']-.5*k*area*min(height,0)**2)<1e-15
  assert abs(r['gradient_N'].sum()-k*area*min(height,0))<1e-12
  assert np.linalg.eigvalsh(r['hessian_N_m']).min()>-1e-11
  if height>=0:np.testing.assert_array_equal(r['gradient_N'],0.)
  if height==0:assert r['whole_face_zero_H_selection'];np.testing.assert_array_equal(r['hessian_N_m'],0.)
 base=integrate_face(nodal(lambda r,s:r-.31),area,kappa_A=k)
 near=integrate_face(nodal(lambda r,s:r-.31+1e-15*s*s),area,kappa_A=k)
 np.testing.assert_allclose(near['gradient_N'],base['gradient_N'],atol=1e-11,rtol=1e-10)
 np.testing.assert_allclose(real_roots([1.,-2.,1.]),[1.]);np.testing.assert_allclose(real_roots([-.3,1.,0.]),[.3])
 assert real_roots([1.,0.,1.])==[] and real_roots([0.,0.,0.])==[]


def test_affine_clipping_normalization_resultant_torque_and_reference_metric():
 a=.37;area=3.;k=1500.;r=integrate_face(nodal(lambda x,y:x-a),area,kappa_A=k)
 E,g,H,rs,ws,z=independent_affine_rule(a,area,k)
 assert abs(r['energy_J']-E)<1e-10;np.testing.assert_allclose(r['gradient_N'],g,atol=1e-9,rtol=1e-11);np.testing.assert_allclose(r['hessian_N_m'],H,atol=1e-8,rtol=1e-11)
 assert abs(r['active_reference_area_m2']-area*(2*a-a*a))<1e-12
 assert abs(g.sum()-2*area*k*(-a*a/2+a**3/6))<1e-10
 X=np.c_[2*NODES[:,0],3*NODES[:,1],np.zeros(6)];F=np.zeros((6,3));F[:,2]=-r['gradient_N'];torque=np.cross(X,F).sum(0);points=np.c_[2*rs[:,0],3*rs[:,1],np.zeros(len(rs))];forceq=np.c_[np.zeros((len(rs),2)),-k*ws*z];np.testing.assert_allclose(torque,np.cross(points,forceq).sum(0),atol=1e-10,rtol=1e-12)
 np.testing.assert_allclose(F.sum(0),forceq.sum(0),atol=1e-10,rtol=1e-12)


def test_quadratic_circle_tangency_event_splits_FD_gradient_fullH_symmetry_PSD():
 h=nodal(lambda r,s:(r-.3)**2+(s-.3)**2-.04);area=.03;k=1500.;answer=integrate_face(h,area,kappa_A=k)
 events=answer['topology_r_events'];assert min(abs(np.array(events)-.1))<1e-13 and min(abs(np.array(events)-.5))<1e-13
 np.testing.assert_allclose(answer['hessian_N_m'],answer['hessian_N_m'].T,atol=1e-12,rtol=0);assert np.linalg.eigvalsh(answer['hessian_N_m']).min()>-1e-10
 d=np.random.default_rng(52).normal(size=6);d/=np.linalg.norm(d);eps=2e-6
 plus=integrate_face(h+eps*d,area,kappa_A=k);minus=integrate_face(h-eps*d,area,kappa_A=k)
 assert abs((plus['energy_J']-minus['energy_J'])/(2*eps)-np.dot(answer['gradient_N'],d))<2e-9
 np.testing.assert_allclose((plus['gradient_N']-minus['gradient_N'])/(2*eps),answer['hessian_N_m']@d,atol=2e-7,rtol=2e-6)
 tangent=integrate_face(nodal(lambda r,s:(r-.3)**2),area,kappa_A=k);assert tangent['energy_J']<1e-25 and np.linalg.norm(tangent['gradient_N'])<1e-18
 # Passive height-unit rescaling with corresponding kappa preserves gradient scale.
 scaled=integrate_face(h*100,area,kappa_A=k/100)
 np.testing.assert_allclose(scaled['gradient_N'],answer['gradient_N'],atol=1e-9,rtol=2e-10)
 assert answer['estimated_energy_error_J']<=1e-12+1e-10*abs(answer['energy_J'])
 assert np.all(answer['estimated_gradient_errors_N']<=1e-10+1e-10*abs(answer['gradient_N']))
