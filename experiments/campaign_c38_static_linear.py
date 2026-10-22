"""Rest-tangent P2 static baseline for the declared two-support block family.

Two responses at E0 give u(E,F)=(E0/E)*(u_gravity+F*u_unit_surface).
This is a linear baseline, not the nonlinear reference or image assimilation.
Material/support are supplied oracle parameters or candidate hypotheses. Never
silently populate them from private truth in the public-observation track.
"""
from dataclasses import dataclass
import numpy as np
import torch
from experiments.campaign_c38_static_family import (
    compile_known_forces, mask_trace_certificate, SUPPORTS, POISSON, DENSITY)
from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
from experiments.campaign_p2_torch import LinearFailure


@dataclass(frozen=True)
class StaticLinearBasis:
    reference_young_Pa: float
    support_kind: str
    rest_m: np.ndarray
    constraint_mask: np.ndarray
    gravity_response_m: np.ndarray
    unit_surface_response_m_per_N: np.ndarray
    independent_checks: tuple

    def displacement(self, young_Pa, Fz_N):
        if not np.isfinite([young_Pa, Fz_N]).all() or young_Pa <= 0 or not -2 <= Fz_N <= 2:
            raise ValueError('Positive candidate modulus and declared public force required')
        return (self.reference_young_Pa / float(young_Pa)) * (
            self.gravity_response_m + float(Fz_N) * self.unit_surface_response_m_per_N)

    def positions(self, young_Pa, Fz_N):
        return self.rest_m + self.displacement(young_Pa, Fz_N)


def fit_reference_basis(model, support_kind, *, operator=None, device='cpu',
                        residual_tolerance_N=1e-10):
    """Exactly two rest-tangent solves; no Newton, observations or dataset draw.

    An explicit compatible operator may supply a reviewed HVP implementation.
    Every result is checked with the original Torch elastic tangent. No physical
    mass is changed or used to disguise static equilibrium as a tiny time step.
    """
    if support_kind not in SUPPORTS or model.poisson != POISSON or model.density != DENSITY:
        raise ValueError('Declared support, Poisson ratio and known density required')
    if not np.isfinite([model.young, residual_tolerance_N]).all() or model.young <= 0 or residual_tolerance_N <= 0:
        raise ValueError('Finite positive reference modulus and residual tolerance required')
    mask_trace_certificate(model.rest, model.elements, model.faces, model.mask, support_kind)
    force = compile_known_forces(model.rest, model.tets, model.faces, model.elements, 1.)
    op = operator if operator is not None else StreamingSupportedP2(
        model, device, 4000, 8, support_kind=support_kind)
    if not np.array_equal(op.rest.detach().cpu().numpy(), model.rest) or not np.array_equal(
            op.mask.detach().cpu().numpy(), model.mask):
        raise ValueError('Operator must use the identical reference geometry and supports')
    if op.young != model.young or op.poisson != model.poisson:
        raise ValueError('Operator must use identical reference constitutive parameters')
    for field, tensor in [('elements', op.elements), ('linear_gradients', op.linear_gradients),
                          ('volumes', op.volumes), ('bary', op.bary),
                          ('weights', op.quadrature_weights)]:
        if not np.array_equal(tensor.detach().cpu().numpy(), getattr(model, field)):
            raise ValueError('Operator must use identical geometry and quadrature: ' + field)
    if op.declared_support_kind != support_kind:
        raise ValueError('Identical declared support required')
    cache = op.evaluate(torch.zeros_like(op.rest))
    responses, checks = [], []
    for name, load in [('gravity', force['body_N']), ('unit_surface', force['surface_N'])]:
        rhs = torch.as_tensor(load, dtype=torch.float64, device=op.rest.device)
        direction, iterations = op.cg(cache, rhs, 0., rtol=0., atol=residual_tolerance_N / 4)
        # Direct class call bypasses any prospective backend's tangent override.
        applied = StreamingSupportedP2.tangent(op, cache, direction)
        residual = (applied - rhs).masked_fill(op.mask, 0.)
        norm = float(torch.linalg.vector_norm(residual))
        displacement = direction.detach().cpu().numpy().copy()
        if not np.isfinite(displacement).all() or norm > residual_tolerance_N or np.any(displacement[model.mask] != 0):
            raise LinearFailure('Original rest-tangent residual or exact support check failed')
        displacement.setflags(write=False)
        responses.append(displacement)
        checks.append(dict(basis=name, original_Torch_tangent_residual_N=norm,
                           tolerance_N=residual_tolerance_N, CG_iterations=iterations,
                           supplied_or_candidate_physics=True, passed=True))
    rest = model.rest.copy()
    mask = model.mask.copy()
    rest.setflags(write=False)
    mask.setflags(write=False)
    return StaticLinearBasis(float(model.young), support_kind, rest, mask,
                             responses[0], responses[1], tuple(checks))
