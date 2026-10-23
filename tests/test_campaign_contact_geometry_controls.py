import numpy as np
import pytest
from experiments.campaign_contact_geometry_controls import manufactured_states,independent_adaptive_response,CONTROL_NAMES
from experiments.run_p2_contact_validation import build
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule
from experiments.run_p2_contact_validation import exact_allactive_surface

def test_manufactured_states_use_new_geometry_without_response_arrays():
 for cells in [(1,1,1),(2,1,1)]:
  m=build(cells,300000.,9.81);states=manufactured_states(m.rest);assert tuple(x[0] for x in states)==CONTROL_NAMES
  bottom=np.isclose(m.rest[:,2],m.rest[:,2].min());np.testing.assert_allclose((m.rest+states[2][1])[bottom,2],-1e-5,atol=1e-17)
  heights=(m.rest+states[3][1])[bottom,2];assert heights.min()<0<heights.max()
  np.testing.assert_array_equal(states[0][1][:,2],np.full(len(m.rest),-.5*9.81*(1/12)**2))

def test_deep_penetration_strict_adaptive_failure_is_explicit_not_partial_sum():
 m=build((1,1,1),300000.,9.81);b,w=positive_keast24();old80,_=exact_degree6_rule();a=dict(bary=b,weights=w[:,None]*m.volumes[None],guard_bary=np.concatenate([m.check_bary,m.bary,old80,b]))
 # Uniform translation makes bulk exactly stress-free; ALL faces fully active.
 u=np.zeros_like(m.rest);u[:,2]=-float(m.rest[:,2].max())-.001
 with pytest.raises(RuntimeError,match='partial sums NOT reference'):independent_adaptive_response(m,u,a)
