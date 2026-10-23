"""Manufactured endpoint payloads; no actual campaign archive/noise/truth reads."""
import copy
import numpy as np
import pytest
from experiments import campaign_c38_public_prefix_inference_v02 as oldprefix
from experiments import campaign_c39_postprobe_inference as oldside
from experiments import campaign_resolved_prefix_inference as prefix
from experiments import campaign_resolved_SIDE_inference as side
from test_campaign_c38_public_prefix_inference_v02 import design,payload
from test_campaign_c39_postprobe_inference import probe


class Forward:
 categories=prefix.CATEGORIES;young_nodes=np.array([75000.,np.sqrt(75000.*150000.),150000.]);forces_N=(-2.,0.,1.,2.)
 def __init__(self,amplitude):self.amplitude=amplitude;self.initial_candidate=self;self.calls=[]
 def observation_coefficients(self,c,F,ids):
  self.calls.append((c,F,tuple(ids)));a=np.zeros((3,len(ids),3))
  if F in (1.,'canonical_up_1N'):
   for j,p in enumerate(ids):
    if p==prefix.POINT_IDS[0]:a[1,j,2]=self.amplitude
  return a.reshape(3,-1)


QUANTILE_KEYS={'canonical_quantile_details','canonical_displacement_equal_tail_m','canonical_position_equal_tail_m','canonical_displacement_quantile_lower_m','canonical_displacement_quantile_upper_m','canonical_position_quantile_lower_m','canonical_position_quantile_upper_m','canonical_position_interval95_numerical_enclosure_m'}


def nonquantile(result):
 r=copy.deepcopy(result)
 for key in ('quantile_resolution','cache_stats','GL64_endpoint_CDF_checks'):r.pop(key,None)
 for method in r['methods'].values():
  for key in QUANTILE_KEYS:method.pop(key,None)
 for comparison in r.get('comparisons',{}).values():comparison.pop('maximum_canonical_interval_endpoint_difference_m',None)
 return r


def bit_equal(a,b):
 assert type(a)is type(b),(type(a),type(b))
 if isinstance(a,np.ndarray):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:bit_equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):bit_equal(x,y)
 elif isinstance(a,(float,np.floating)):assert np.float64(a).tobytes()==np.float64(b).tobytes()
 else:assert a==b


@pytest.mark.parametrize('compiler',['prefix','SIDE'])
@pytest.mark.parametrize('amplitude',[1e-12,1e-3])
def test_full_loops_nonquantile_bitidentity_and_resolved_loguniform_targets(monkeypatch,compiler,amplitude):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('actual file read')))
 monkeypatch.setattr(np.random,'default_rng',lambda *a,**k:(_ for _ in ()).throw(AssertionError('noise draw')))
 p=payload();f=Forward(amplitude)
 if compiler=='prefix':old=oldprefix.infer_public_prefix(f,p,design=design());new=prefix.infer_public_prefix(Forward(amplitude),p,design=design())
 else:
  acquired=probe(p);old=oldside.infer_public_probe(f,p,acquired,design=design());new=side.infer_public_probe(Forward(amplitude),p,acquired,design=design())
 assert old['completed']and old['numerical_controls_passed'],old['failures']
 assert new['completed']and new['numerical_controls_passed'],new['failures']
 bit_equal(nonquantile(old),nonquantile(new))
 exact=amplitude*2**(np.array(prefix.PROBABILITIES)-1)
 for name in ('adaptive','GL128'):
  row=new['methods'][name];q=row['canonical_displacement_equal_tail_m'][0,2]
  np.testing.assert_allclose(q,exact,atol=amplitude*6e-9,rtol=0)
  assert q[0]<q[1]<q[2]
  if amplitude==1e-12:
   oldq=old['methods'][name]['canonical_displacement_equal_tail_m'][0,2];assert oldq[0]==oldq[1]==oldq[2]
  details=row['canonical_quantile_details'][2]['quantiles']
  for probability,expected,detail in zip(prefix.PROBABILITIES,exact,details):
   assert detail['lower_m']<=expected<=detail['upper_m']
   bracket=detail['CDF_bracket'];assert bracket['maximum_CDF_error_bound']<=1e-6
   assert bracket['upper']-bracket['lower']<=min(1e-8,1e-8*bracket['initial_bracket_span'])
 assert len(new['GL64_endpoint_CDF_checks'])==162
 assert all(r['passed']for r in new['GL64_endpoint_CDF_checks'])
 assert new['quantile_resolution']['target_returned_CDF_probability_tolerance']==1e-6


def test_new_compilers_keep_public_role_and_typed_SIDE_barriers():
 f=Forward(.001);p=payload();bad=dict(p,young_Pa=90000.)
 assert not prefix.infer_public_prefix(f,bad,design=design())['completed']and not f.calls
 q=probe(p);q['action_id']='middle_down_2N'
 assert not side.infer_public_probe(f,p,q,design=design())['completed']and not f.calls


def test_only_coordinate_resolver_and_metadata_change_in_cloned_method_loops():
 import ast,inspect
 for old,new,name in [(oldprefix,prefix,'infer_public_prefix'),(oldside,side,'infer_public_probe')]:
  a=ast.parse(inspect.getsource(getattr(old,name)));b=ast.parse(inspect.getsource(getattr(new,name)))
  class StripMetadata(ast.NodeTransformer):
   def visit_keyword(self,node):
    if node.arg=='quantile_resolution':node.value=ast.Constant(value='QUANTILE_METADATA')
    return self.generic_visit(node)
  assert ast.dump(StripMetadata().visit(a))==ast.dump(StripMetadata().visit(b))
