"""Campaign-only nonlinear candidate with an explicit zero-amplitude reference.

Old model definitions/loaders remain unchanged. The new model keeps the same
coordinate MLP and parameter count as NeuralTrajectoryOperator, but subtracts
its prediction at zero force AND zero prescribed support-motion amplitude.
It retains nonlinear force interactions without inventing motion at zero input.
This is not a linear Green map and cannot use analytic Gaussian load filtering.
"""
import hashlib
from pathlib import Path
import torch
from gaussian_operator import NeuralTrajectoryOperator, NeuralGreenOperator
from train_gaussian_inverse import load_frozen_operator


class ZeroReferencedTrajectoryOperator(NeuralTrajectoryOperator):
    def forward(self, points, times, physical, boundary_type, force_type, geometry_reference=None):
        zero = physical.clone()
        zero[:, 16:25] = 0
        actual = super().forward(points, times, physical, boundary_type, force_type, geometry_reference)
        baseline = super().forward(points, times, zero, boundary_type, force_type, geometry_reference)
        reference = points[None, None] if points.ndim == 2 else points[:, None]
        return reference+(actual-baseline)


CLASSES = {'green': NeuralGreenOperator, 'nonlinear': NeuralTrajectoryOperator,
           'zero_referenced': ZeroReferencedTrajectoryOperator}


def load_campaign_operator(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('model_class') != 'ZeroReferencedTrajectoryOperator':
        return load_frozen_operator(path, device)
    expected = checkpoint['source_sha256']
    for name in ['campaign_forward_models.py', 'gaussian_operator.py']:
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError('Campaign operator source mismatch')
    kwargs = dict(checkpoint['model_kwargs'])
    kwargs['reference_points'] = checkpoint['model_state']['geometry_reference'].cpu().numpy()
    model = ZeroReferencedTrajectoryOperator(**kwargs).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
