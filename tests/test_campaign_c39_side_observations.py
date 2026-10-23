import copy
import numpy as np
import pytest
from experiments import campaign_c39_side_observations as m
from test_campaign_c38_public_prefix_inference_v02 import design,payload


def test_typed_side_payload_exactcov_noFz_or_future_and_prefix_unchanged():
 p=payload();original=copy.deepcopy(p);y=p['reference_points_m']+.001;q=m.public_probe(p,m.SIDE,y,design=design());r=m.decode(p,m.encode(p,q,design=design()),design=design())
 assert r['action_id']==m.SIDE and np.array_equal(r['force_vector_N'],[0.,2.,0.])and set(r)==m.PUBLIC_KEYS and 'force_Fz_N'not in r
 for k in p:
  if isinstance(p[k],np.ndarray):np.testing.assert_array_equal(p[k],original[k])
  else:assert p[k]==original[k]
 assert not{'young_Pa','support','private_configuration_index','repeat_index','seed','source_records','target_positions_m'}&set(r)
 for k,v in [('force_Fz_N',0.),('counterfactual_DOWN_y',np.zeros((9,3))),('young_Pa',90000.)]:
  with pytest.raises(ValueError,match='allowlist'):m.validate_public_probe(p,dict(q,**{k:v}),design=design())
 bad=copy.deepcopy(q);bad['force_vector_N']=np.array([0.,0.,-2.])
 with pytest.raises(ValueError,match='worldvector'):m.validate_public_probe(p,bad,design=design())


def test_only_newSIDE_seedstream_fakeRNG_no_identity_or_oldDOWN_draw():
 entropies=[]
 class RNG:
  def standard_normal(self,shape):assert shape==(9,3);return np.ones(shape)
 def factory(seed):entropies.append(seed.entropy);return RNG()
 e=m.side_measurement_error(2,7,rng_factory=factory);np.testing.assert_array_equal(e,np.full((9,3),.0002));assert entropies==[[2026102305,2,7,0]]
 for c,r in [(4,0),(0,16),(True,0)]:
  with pytest.raises(ValueError):m.side_measurement_error(c,r,rng_factory=factory)
 assert len(entropies)==1
