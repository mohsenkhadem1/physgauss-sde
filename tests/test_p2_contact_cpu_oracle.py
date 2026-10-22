import numpy as np
from experiments.campaign_p2_contact_cpu_oracle import CPUContactOracle,surface_response


def test_independent_surface_area_partition_allactive_resultant_and_FD():
 c=CPUContactOracle();m=c.model;faces,A,N,w=c.surface
 assert abs(A.sum()-(2*(.24*.08+.24*.08+.08*.08)))<1e-15
 np.testing.assert_allclose(N.sum(1),1,atol=5e-16)
 q=m.rest.copy();q[:,2]-=1.;E,g,H,p=surface_response(q,*c.surface,kappa=100.,tangent=True)
 # Every surface point is below floor; reference surface centroid z=.09.
 np.testing.assert_allclose(g.sum(0),[0,0,100*A.sum()*(.09-1)],atol=1e-13)
 d=np.random.default_rng(4).normal(size=q.shape);eps=1e-6
 ep,gp,_,_=surface_response(q+eps*d,*c.surface,kappa=100.);em,gm,_,_=surface_response(q-eps*d,*c.surface,kappa=100.)
 np.testing.assert_allclose((ep-em)/(2*eps),np.sum(g*d),atol=2e-9)
 np.testing.assert_allclose((gp-gm)/(2*eps),(H@d.ravel()).reshape(q.shape),atol=2e-8)
 assert np.linalg.eigvalsh(H.toarray()).min()>-1e-12


def test_bulk_mass_and_gravity_rigid_translation_consistency():
 c=CPUContactOracle();m=c.model
 np.testing.assert_allclose(c.mass@np.broadcast_to([0,0,-9.81],m.rest.shape),m.load,atol=2e-15)
 np.testing.assert_allclose(m.load.sum(0),[0,0,-.96*9.81],atol=1e-14)
 E,g,H,J=c.elastic_u(np.zeros_like(m.rest),True)
 assert abs(E)<1e-25 and np.max(abs(g))<1e-13 and J.min()>1-1e-14
