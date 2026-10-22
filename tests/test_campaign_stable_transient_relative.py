import numpy as np
from experiments.campaign_stable_transient import StableBlockTransient
from experiments.campaign_stable_transient_relative import RelativeStableTransient

def test_exact_same_elastic_energy_and_local_objective_gradient():
 m=RelativeStableTransient();rng=np.random.default_rng(3);u=rng.normal(size=m.model.rest.shape)*1e-5;u[m.mask]=0;z=rng.normal(size=u.shape)*1e-7;z[m.mask]=0;pred=rng.normal(size=u.shape)*1e-7;pred[m.mask]=0;h=1/7680;v=rng.normal(size=u.shape);v[m.mask]=0;v/=np.linalg.norm(v);eps=1e-9
 def energy(z):return m.model.elastic(m.model.rest+u+z)[0]+.5*np.sum(m.mass[:,None]/h**2*(z-pred)**2)-np.sum(m.force*z)
 g=m.model.elastic(m.model.rest+u+z)[1]+m.mass[:,None]/h**2*(z-pred)-m.force
 finite=(energy(z+eps*v)-energy(z-eps*v))/(2*eps)
 np.testing.assert_allclose(finite,np.sum(g*v),atol=2e-6,rtol=2e-6)

def test_same_method_no_force_or_time_changes_and_tight_early_parity():
 a=StableBlockTransient();b=RelativeStableTransient();sa=a.initial();sb=b.initial();pa=pb=None;h=1/3840
 np.testing.assert_array_equal(a.force,b.force);np.testing.assert_array_equal(a.mass,b.mass)
 for i in range(3):
  na,da=a.step(sa,h,'BDF2',pa);nb,db=b.step(sb,h,'BDF2',pb)
  np.testing.assert_allclose(na.q,b.model.rest+nb.u,atol=1e-13,rtol=0)
  assert db['objective_free_R_N']<=1e-10 and db['physical_original_free_R_N']<=1e-9
  assert db['maximum_world_displacement_roundtrip_m']<1e-16 and db['constraint_defect_m']==0
  assert db['actual_method']==da['actual_method'] and db['h_eff_s']==da['h_eff_s']
  pa,sa=sa,na;pb,sb=sb,nb
