import numpy as np
import torch
from scripts.compositional_neural_mechanics.contact_transfer_model import make_contact_model


def test_shared_law_contact_composition_variational_operator():
    for family in ('mr_saturated','yeoh_polynomial'):
        m=make_contact_model(family,'shared_laws',1,16)
        u=torch.zeros_like(m.engine.rest,requires_grad=True);r=m.torch_residual(u);H=m.evaluate(u.detach().numpy(),True)['tangent']
        direction=torch.sin(torch.arange(u.numel(),dtype=u.dtype)+1).reshape_as(u)
        derivative=torch.autograd.grad((r*direction).sum(),u)[0].detach().numpy().ravel()
        np.testing.assert_allclose(derivative,H.T@direction.numpy().ravel(),atol=1e-8,rtol=1e-9)
        assert np.isfinite(r.detach().numpy()).all()
