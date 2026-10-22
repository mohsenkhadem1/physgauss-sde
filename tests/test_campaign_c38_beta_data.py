import numpy as np
from experiments.campaign_c38_beta_data import contexts_from_latents,simulate_objects


def test_context_counterfactual_noise_and_prefix_invariance():
    unit=np.array([[.3,.4],[.8,.2]]);pn=np.array([[.0001,-.0002],[.0002,.0001]]);an=np.array([[.0002,.0003,.0004,.0005],[-.0001,.0002,-.0003,.0004]])
    a,private=contexts_from_latents(unit,pn,an);b,_=contexts_from_latents(unit,pn,an+1.)
    np.testing.assert_array_equal(a['features'][:,:2],b['features'][:,:2])
    # Same acquired left/right frame is reused across counterfactual prefixes.
    np.testing.assert_array_equal(a['observed_m'][:,2,1],a['observed_m'][:,5,1])
    np.testing.assert_array_equal(a['observed_m'][:,3,1],a['observed_m'][:,6,1])
    assert a['features'].shape==(2,8,10) and np.isfinite(a['features']).all()
    assert not {'unit','theta','canonical_positions_m'}&set(a)
    assert private['canonical_positions_m'].shape==(2,8,3)


def test_whole_object_contexts_reproducible_streams():
    a,p,x=simulate_objects(4,7);b,q,y=simulate_objects(4,7)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(p['unit'],q['unit']);np.testing.assert_array_equal(x['features'],a['features'][np.arange(4),x['camera']])
    assert np.unique(p['unit'],axis=0).shape[0]==4
