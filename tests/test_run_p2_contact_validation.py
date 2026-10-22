import numpy as np
from experiments.run_p2_contact_validation import analytic_triangle_mass,exact_allactive_surface,build,JOBS
from experiments.campaign_p2_torch_contact import triangle_rule,boundary_trace

def test_independent_exact_triangle_polynomial_mass():
 c,G=analytic_triangle_mass();b,N,w=triangle_rule(5)
 np.testing.assert_allclose(c,w@N,atol=2e-16)
 np.testing.assert_allclose(G,np.einsum('q,qa,qb->ab',w,N,N),atol=2e-16)
 assert np.linalg.eigvalsh(G).min()>0
 np.testing.assert_allclose(G.sum(),1.)

def test_closed_surface_exact_reference_and_fixedphysics():
 m=build([1,1,1],300000.,9.81);U,g=exact_allactive_surface(m,m.rest,100.,1.)
 np.testing.assert_allclose(g.sum(0),[0,0,100*.0896*(.09-1)],atol=1e-13)
 assert U>0 and not m.mask.any()
 assert len(JOBS)==7 and all(j['kappa'] in [1e9,1e10] for j in JOBS)
