import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import sys
from pathlib import Path
import numpy as np
import torch
import hashlib
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_forward_models import ZeroReferencedTrajectoryOperator, load_campaign_operator
from gaussian_operator import NeuralTrajectoryOperator


def setup():
    torch.manual_seed(2102)
    rest = np.array([[-.12, -.06, .21], [.12, .06, .29]], dtype=np.float32)
    model = ZeroReferencedTrajectoryOperator(hidden=16, rank=4, depth=1, reference_points=rest)
    p = torch.zeros((1, 25))
    p[:, :4] = torch.tensor([np.log(5000), np.log(600), .35, np.log(.5)])
    p[:, 4:16] = torch.tensor([-.12, 0, .25, -.12, 0, .25, .12, 0, .25, .1, 0, .25])
    return model, torch.tensor(rest), p


def test_zero_force_and_motion_return_rest_for_all_times():
    model, rest, p = setup()
    # Deliberate network bias must cancel; rest cannot depend on initialization.
    with torch.no_grad():
        model.spatial[-1].bias.fill_(1)
        model.temporal[-1].bias.fill_(1)
    times = torch.tensor([0., .5, 1., 2., 3.])
    q = model(rest, times, p, torch.tensor([0]), torch.tensor([0]))
    torch.testing.assert_close(q, rest[None, None].expand_as(q), atol=0, rtol=0)


def test_initial_state_exact_and_force_gradients_retained():
    model, rest, p = setup()
    p[:, 16] = .02
    p.requires_grad_()
    q = model(rest, torch.tensor([0., 1., 2.]), p, torch.tensor([1]), torch.tensor([0]))
    torch.testing.assert_close(q[:, 0], rest[None], atol=0, rtol=0)
    gradient = torch.autograd.grad(q[:, 1:].sum(), p)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, 16:19].abs().sum() > 0


def test_prescribed_support_motion_remains_an_active_input():
    model, rest, p = setup()
    p[:, 22] = .008
    q = model(rest, torch.tensor([1., 2., 3.]), p, torch.tensor([3]), torch.tensor([0]))
    assert (q-rest[None, None]).abs().max() > 0


def test_reference_subtraction_adds_no_parameters_and_preserves_seed_initialization():
    model, rest, _ = setup()
    torch.manual_seed(2102)
    plain = NeuralTrajectoryOperator(hidden=16, rank=4, depth=1, reference_points=rest.numpy())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in plain.parameters())
    for a, b in zip(model.parameters(), plain.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_campaign_checkpoint_roundtrip_and_source_guard(tmp_path):
    model, rest, p = setup()
    p[:, 16] = .02
    directory = Path(__file__).resolve().parents[1]/'experiments'
    source = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
              for name in ['campaign_forward_models.py', 'gaussian_operator.py']}
    checkpoint = {'model_class': 'ZeroReferencedTrajectoryOperator', 'model_kwargs': {'hidden': 16, 'rank': 4, 'depth': 1},
                  'model_state': model.state_dict(), 'source_sha256': source}
    path = tmp_path/'model.pt'
    torch.save(checkpoint, path)
    restored = load_campaign_operator(path)
    times, bc, family = torch.tensor([0., 1., 3.]), torch.tensor([1]), torch.tensor([0])
    torch.testing.assert_close(model(rest, times, p, bc, family), restored(rest, times, p, bc, family), atol=0, rtol=0)
    checkpoint['source_sha256']['campaign_forward_models.py'] = 'wrong'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='source mismatch'):
        load_campaign_operator(path)
