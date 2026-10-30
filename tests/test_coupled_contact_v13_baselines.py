import numpy as np
import pytest
from scripts.coupled_contact_v13.baselines import fit_baselines


def fixture():
    initial=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],float)
    times=np.linspace(0,1,21);tools=np.zeros((21,2,3));tools[:,0,0]=times*.01
    observed=initial[None]+np.stack([times*.02,times*0,times*0],axis=1)[:,None]
    return initial,times,tools,observed


def test_new_interaction_initial_and_rigid_distances():
    initial,times,tools,observed=fixture();model=fit_baselines(initial,observed,tools,times)
    new=initial+np.array([.2,.3,.4]);forecast=model.predict(new,tools*.5+np.array([.2,.3,.4]),times)
    for values in forecast.values():np.testing.assert_allclose(values[0],new,atol=1e-14)
    distances=np.linalg.norm(forecast['position_rigid'][:,:,None]-forecast['position_rigid'][:,None,:],axis=-1)
    np.testing.assert_allclose(distances,np.broadcast_to(distances[0],distances.shape),atol=1e-14)
    assert np.isfinite(forecast['position_ridge']).all()


def test_translation_generalization_from_position_only():
    initial,times,tools,observed=fixture();model=fit_baselines(initial,observed,tools,times,ridge=1e-6)
    newtools=tools*.8
    pred=model.predict(initial,newtools,times)
    expected=initial[None]+(observed-initial[None])*.8
    np.testing.assert_allclose(pred['position_ridge'],expected,atol=1e-7)
    np.testing.assert_allclose(pred['position_rigid'],expected,atol=1e-7)


def test_hidden_observations_cannot_affect_fit():
    initial,times,tools,observed=fixture();mask=np.ones(observed.shape[:2],bool);mask[3:12,0]=False
    corrupt=observed.copy();corrupt[~mask]=np.nan
    a=fit_baselines(initial,observed,tools,times,mask).predict(initial,tools,times)
    b=fit_baselines(initial,corrupt,tools,times,mask).predict(initial,tools,times)
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_invalid_public_inputs_rejected():
    initial,times,tools,observed=fixture();model=fit_baselines(initial,observed,tools,times)
    with pytest.raises(ValueError):model.predict(initial,tools[:,:1],times)
    with pytest.raises(ValueError):model.predict(initial,tools,times[::-1])


def test_deformation_ridge_and_rigid_are_distinct():
    initial,times,tools,observed=fixture();observed[:,1,0]+=times*.03
    model=fit_baselines(initial,observed,tools,times,ridge=1e-6);pred=model.predict(initial,tools,times)
    assert np.linalg.norm(pred['position_ridge'][-1]-pred['position_rigid'][-1])>.01
