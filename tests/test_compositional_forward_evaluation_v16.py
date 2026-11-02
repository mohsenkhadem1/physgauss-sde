import numpy as np
import pytest
import torch
from scripts.compositional_neural_mechanics.forward_evaluation import SmallStrainMatchedNeoHookean,score_prediction
from scripts.compositional_neural_mechanics.materials import teacher_energy


def test_misspecified_baseline_has_matched_reference_tangent():
    law=SmallStrainMatchedNeoHookean();z=torch.tensor([.2,.7],dtype=torch.float64)
    identity=torch.eye(3,dtype=z.dtype)
    baseline=torch.autograd.functional.hessian(lambda F:law.energy_density(F,z),identity)
    for family in ('mooney_rivlin','yeoh'):
        tangent=torch.autograd.functional.hessian(lambda F:teacher_energy(F,z,family),identity)
        torch.testing.assert_close(baseline,tangent,rtol=1e-12,atol=1e-10)


def test_world_error_keeps_rigid_error_separate_from_deformation():
    rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    reference=dict(displacement=np.zeros_like(rest),wrenches=np.zeros((2,6)))
    prediction=dict(displacement=np.broadcast_to([.03,0,0],rest.shape).copy(),wrenches=np.ones((2,6)))
    scores=score_prediction(prediction,reference,rest)
    assert np.isclose(scores['displacement_coordinate_rmse_m'],.03/np.sqrt(3))
    assert scores['canonical_nonrigid_coordinate_rmse_m']<1e-14
    assert scores['force_coordinate_rmse_N']==1 and scores['torque_coordinate_rmse_Nm']==1
    prediction['displacement'][0,0]=np.nan
    with pytest.raises(ValueError,match='failures'):score_prediction(prediction,reference,rest)
