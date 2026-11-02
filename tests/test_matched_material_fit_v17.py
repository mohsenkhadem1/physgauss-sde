import numpy as np
import pytest
from scripts.compositional_dynamics_v17.matched_material_fit import observed_loss


def test_hidden_positions_and_prescribed_initial_frame_cannot_affect_fit():
    pred=np.zeros((3,2,3));obs=np.zeros_like(pred);mask=np.array([[1,1],[1,0],[1,0]],bool)
    obs[0]=100;obs[1:,1]=np.nan
    assert observed_loss(pred,obs,mask)==(0.,0.,2)
    obs[2,0,0]=.003
    loss,rms,count=observed_loss(pred,obs,mask)
    assert loss==pytest.approx(.003**2/6) and rms==pytest.approx(.003/np.sqrt(2)) and count==2


def test_no_valid_observations_and_bad_masks_fail():
    x=np.zeros((3,2,3))
    with pytest.raises(ValueError):observed_loss(x,x,np.zeros((3,2),bool))
    with pytest.raises(ValueError):observed_loss(x,x,np.ones((3,2)))
