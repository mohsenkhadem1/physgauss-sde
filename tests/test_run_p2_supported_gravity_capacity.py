import numpy as np
import pytest
from experiments.campaign_c38_static_family import build_geometry
from experiments.run_p2_supported_gravity_capacity import structured_p2_counts,prescribed_static_state,sha,ROOT


def test_structured_storage_counts():
 for c in [(4,2,2),(8,4,4)]:
  m=build_geometry(c,100000.,'both_ends',Fz=0.,rule='keast24');a=structured_p2_counts(c)
  assert a['nodes']==len(m.rest) and a['tetrahedra']==len(m.elements) and a['fixed_nodes']==m.mask[:,0].sum()
 assert structured_p2_counts((20,10,10))['nodes']==18081
 assert structured_p2_counts((24,12,12))['nodes']==30625


def test_historical_state_guard_allows_only_known_fixed_load_difference(tmp_path):
 m=build_geometry((4,2,2),100000.,'both_ends',Fz=0.,rule='keast24');load=m.load.copy();fixed=np.flatnonzero(m.mask[:,0]);load[fixed[0],2]-=.12
 p=tmp_path/'fake.npz'
 def save(load):np.savez(p,rest_m=m.rest,elements=m.elements,tets=m.tets,constraint_mask=m.mask,positions_m=m.rest,known_load_N=load)
 save(load);r=dict(path=str(p),sha256=sha(p));u,metadata=prescribed_static_state(m,r);assert not u.any() and metadata['historical_right_traction_only_at_fixed_nodes']
 changed=load.copy();changed[np.flatnonzero(~m.mask[:,0])[0],2]-=.01;save(changed);r['sha256']=sha(p)
 with pytest.raises(ValueError,match='Only known fixed-node'):prescribed_static_state(m,r)
 save(load);r['sha256']='0'*64
 with pytest.raises(ValueError,match='Changed prescribed'):prescribed_static_state(m,r)
