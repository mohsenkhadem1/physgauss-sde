import numpy as np
import pytest
from experiments.run_p2_supported_gravity_parity import NumpyBulkP2Force,models,pair
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_supported_gravity_dop import SupportedBulkDynamics


def test_vectorized_independent_force_matches_loop_and_directional_energy_gradient():
 m,_=models();backend=NumpyBulkP2Force(m);X=m.rest
 u=np.column_stack([.01*X[:,0]+.02*X[:,1]*X[:,2],-.015*X[:,1],.01*X[:,2]])
 control=StableP2Block((4,2,2),young=100000.,poisson=.35,density=625.,gravity=0.,traction=0.,order=3)
 for key in ['bary','weights','N','check_bary']:setattr(control,key,getattr(m,key).copy())
 W,g,_,J=control.elastic(X+u);actual=backend.evaluate(u)
 assert abs(W-actual['elastic_energy_J'])<1e-12
 np.testing.assert_allclose(actual['gradient_N'],g,atol=2e-11,rtol=2e-12)
 direction=np.random.default_rng(227).normal(size=u.shape);direction/=np.linalg.norm(direction);eps=1e-7
 FD=(backend.evaluate(u+eps*direction)['elastic_energy_J']-backend.evaluate(u-eps*direction)['elastic_energy_J'])/(2*eps)
 assert abs(FD-np.sum(actual['gradient_N']*direction))<2e-7
 with pytest.raises(ValueError,match='J guard'):backend.evaluate(-2*X)


def test_pure_body_law_exact_mass_and_fixed_reaction_not_hidden_traction():
 cpu,gpu=models();np.testing.assert_allclose(cpu.load,gpu.load,atol=1e-12,rtol=0)
 assert np.max(abs(cpu.load[:,:2]))==0 and not np.any(gpu.public_force_components['surface_N'])
 np.testing.assert_allclose(cpu.load.sum(0),[0,0,-625*.24*.08*.08*9.81],atol=3e-13,rtol=0)
 np.testing.assert_allclose(np.cross(cpu.rest-np.array([.12,0,0]),cpu.load).sum(0),0,atol=2e-14,rtol=0)
 d=SupportedBulkDynamics(cpu,NumpyBulkP2Force(cpu));zero=np.zeros_like(cpu.rest);state=d.mechanics(d.pack(zero,zero))
 assert state['mass_residual_N']<1e-12 and np.linalg.norm(d.mass[d.fixed_nodes][:,d.free_nodes]@state['acceleration_m_s2'][d.free_nodes])>.001
 assert not state['acceleration_m_s2'][cpu.mask].any()
 np.testing.assert_allclose((d.mass@state['acceleration_m_s2']).sum(0),cpu.load.sum(0)+state['reaction_N'][d.fixed_nodes].sum(0),atol=2e-12,rtol=0)


def test_error_metrics_use_all_nodes_and_both_physical_components():
 q=np.zeros((25,3,3));v=q.copy();a=q.copy();a[-1,-1,2]=2e-8
 assert not pair(q,v,a,v,qgate=1e-8,vgate=1e-5)['passed']
 b=v.copy();b[10,1,0]=2e-5
 assert not pair(q,v,q,b,qgate=1e-8,vgate=1e-5)['passed']
 assert pair(q,v,q,v,qgate=1e-8,vgate=1e-5)['passed']
