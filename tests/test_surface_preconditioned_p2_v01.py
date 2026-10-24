import numpy as np
from experiments import campaign_surface_p2_static_v02 as p2
from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference


def test_same_potential_and_equilibrium_as_independent_direct_NH_target():
    f=p2.SparseStaticForward(2);E=4000.;K=12.;x=f.X[:,0];known=np.zeros_like(f.X)
    known[:,2]=2e-4*np.sin(np.pi*x/.36)**2;known[f.fixed]=0
    direct=p2.P2NonlinearReference.from_forward(f,E,K,quadrature_order=80)
    force=direct.evaluate(known,force=np.zeros_like(known))['residual']
    expected,a=direct.solve(force)
    model=PreconditionedP2Reference.from_forward(f,E,K,quadrature_order=80)
    result,b=model.solve(force)
    np.testing.assert_allclose(result,expected,rtol=1e-8,atol=2e-12)
    np.testing.assert_allclose(model.evaluate(result,force=force)['potential_J'],direct.evaluate(result,force=force)['potential_J'],rtol=0,atol=0)
    assert b['maximum_free_residual_N']<=1e-10 and b['unmodified_target_potential']
    assert b['rest_factorizations']==1 and len(b['linear_directions'])==b['iterations']


def test_shared_rest_factor_serves_linear_and_separate_quadrature_targets():
    f=p2.SparseStaticForward(1);E=4000.;K=12.;rest=ExactRestPCG(E*f.Kunit_free+K*f.W_free)
    models=[PreconditionedP2Reference.from_forward(f,E,K,quadrature_order=q,direction_solver=rest)for q in (24,80)]
    force=np.zeros_like(f.X);force.ravel()[f.free]=np.linspace(-1e-5,1e-5,len(f.free))
    expected=f.staticforward(np.array([[E,K]]),force)[0,0].ravel()[f.free]
    np.testing.assert_allclose(rest.rest_solve(force.ravel()[f.free]),expected,rtol=1e-11,atol=1e-16)
    states=[m.solve(force)[0]for m in models]
    assert models[0].direction_solver is models[1].direction_solver
    assert rest.rest_factorizations==1
    np.testing.assert_allclose(states[0],states[1],rtol=1e-8,atol=2e-12)
