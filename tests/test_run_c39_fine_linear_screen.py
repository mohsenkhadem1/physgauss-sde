import numpy as np
from experiments.run_c39_fine_linear_screen import canonical,JOBS,ACTIONS,MESHES

def test_canonical_uses_total_gravity_loaded_down_response():
 g=np.arange(12).reshape(4,3)*.003;unit=np.ones((4,3))*.002
 np.testing.assert_allclose(canonical(g,g-2*unit),g+unit,rtol=1e-15,atol=1e-17)
 assert ACTIONS==('gravity','middle_down_2N','middle_side_2N')
 assert len({j['GPU_uuid'] for j in JOBS})==2
 assert MESHES==((32,16,16),(40,20,20))
