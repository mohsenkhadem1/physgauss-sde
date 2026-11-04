import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.bank_retrieval import retrieve


def fixture():
    data={'train_end':3,'observed_initial':np.zeros((2,3)),
          'observed':np.arange(18,dtype=float).reshape(3,2,3),
          'observed_mask':np.ones((3,2),bool)}
    plan={'prefix_frames':[0,2],'point_ids':[1],'coordinate_mask_count':6}
    query=(data['observed'][[0,2]][:,[1]]-data['observed'][0:1,[1]]).reshape(-1)
    features=np.stack([query+2,query,query,query+1]);latent=np.arange(8).reshape(4,2)
    return data,plan,features,latent


def test_nearest_uses_exact_mask_and_stable_tie_break():
    data,plan,features,z=fixture()
    ids,error,q=retrieve(data,plan,features,z,3)
    assert ids.tolist()==[1,2,3]
    assert error.tolist()==[4,0,0,1]
    # Unselected observed point cannot influence retrieval.
    data['observed'][:,0]=np.nan
    assert retrieve(data,plan,features,z,3)[0].tolist()==[1,2,3]


def test_masked_coordinates_do_not_influence_retrieval():
    data,plan,features,z=fixture();data['observed_mask'][2,1]=False
    plan['coordinate_mask_count']=3
    ids,error,q=retrieve(data,plan,features[:,:3],z,1)
    assert ids.tolist()==[1] and q.shape==(3,)


@pytest.mark.parametrize('kind',['future','missing_zero','mask','nan','short','point'])
def test_invalid_retrieval_contracts_fail(kind):
    data,plan,features,z=fixture()
    if kind=='future':plan['prefix_frames']=[0,3]
    elif kind=='missing_zero':plan['prefix_frames']=[1,2]
    elif kind=='mask':plan['coordinate_mask_count']=1
    elif kind=='nan':features[0,0]=np.nan
    elif kind=='short':features=features[:,:2]
    else:plan['point_ids']=[-1]
    with pytest.raises(ValueError):retrieve(data,plan,features,z)
