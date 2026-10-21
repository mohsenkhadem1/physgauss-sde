import copy
import numpy as np
import pytest
from experiments.campaign_linear_state_baseline import fit_models,rollout,hardware_fingerprint


def toy():
    rng=np.random.default_rng(4);R=8;N=4;rest=np.tile(np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]]),(R,1,1))
    rest[4:,:,0]+=.1;controls=rng.normal(size=(R,61,12))*.01;controls[:,0]=0;q=np.repeat(rest[:,None],61,axis=1)
    for i in range(R):
        displacement=np.cumsum(controls[i,:,0])*.01;q[i,:,:,0]+=displacement[:,None]
    return dict(rest=rest,q=q,controls=controls,support_offsets=np.zeros((R,61,3)),fixture_mask=np.zeros((R,N),bool),
        handle_patches=np.tile([[0],[1]],(R,1,1)),handle_centers=rest[:,:2].copy(),handle_masses=np.ones((R,2)),
        handle_inertias=np.tile(np.eye(3),(R,2,1,1)),handle_com_offsets=np.zeros((R,2,3)),handle_initial_rotations=np.tile(np.eye(3),(R,2,1,1)),
        tets=np.array([[0,1,2,3]]),faces=np.array([[0,1,2]]),train_mask=np.array([1,1,1,0,1,1,1,0],bool))


def test_fit_and_rollout_never_read_held_targets():
    data=toy();models=fit_models(data);prediction=rollout(data,3,models)
    poisoned=copy.deepcopy(data);poisoned['q'][~data['train_mask']]=np.nan
    refit=fit_models(poisoned)
    for key in models:
        np.testing.assert_array_equal(models[key]['basis'],refit[key]['basis']);np.testing.assert_array_equal(models[key]['W'],refit[key]['W'])
    np.testing.assert_array_equal(prediction,rollout(poisoned,3,refit))


def test_known_program_causality_and_unknown_hardware_rejection():
    data=toy();models=fit_models(data);before=rollout(data,3,models)
    future=copy.deepcopy(data);future['controls'][3,31:]*=100
    np.testing.assert_array_equal(before[:31],rollout(future,3,models)[:31])
    future['rest'][3,0,0]+=.01
    with pytest.raises(ValueError,match='Unknown'):rollout(future,3,models)
