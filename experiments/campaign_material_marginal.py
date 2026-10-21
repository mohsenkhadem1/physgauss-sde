"""Normalized material marginal for an oracle with known supports and hardware.

This is the four-dimensional marginal of ConditionalPhysicalPrior, not a new
log-uniform density/drag prior. It does not infer unknown support geometry or
boundary categories. Use periodic_indices=() with the generic tempered SMC.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import numpy as np

try:
    from campaign_physical_prior import ConditionalPhysicalPrior, PriorState
    from campaign_hypothesis_inputs import make_prior, realize_hypothesis, known_hardware_from_public, validate_realized_hypothesis
except ImportError:
    from experiments.campaign_physical_prior import ConditionalPhysicalPrior, PriorState
    from experiments.campaign_hypothesis_inputs import make_prior, realize_hypothesis, known_hardware_from_public, validate_realized_hypothesis


@dataclass(frozen=True)
class MaterialState:
    coordinates: np.ndarray  # [N,4]: log E, log rho, nu, log body drag

    def __len__(self):
        return len(self.coordinates)

    def take(self, indices):
        return MaterialState(np.asarray(self.coordinates)[np.atleast_1d(indices)].copy())

    def with_coordinates(self, coordinates):
        return replace(self,coordinates=np.asarray(coordinates))


class MaterialMarginalPrior:
    """E log-uniform; rho, nu and drag uniform in their physical units.

    Intrinsic density includes the rho and drag log-coordinate Jacobians.
    The original prior factors these material variables from all supports,
    categories and passive-force nuisances, so fixing those leaves this law.
    """
    dimension = 4
    periodic_indices = ()

    def __init__(self, conditional_prior):
        if type(conditional_prior) is not ConditionalPhysicalPrior:
            raise TypeError('The normalized hardware-conditional prior is required')
        self.lower = conditional_prior.lower[:4].copy()
        self.upper = conditional_prior.upper[:4].copy()
        self.physical_lower = self.lower.copy()
        self.physical_upper = self.upper.copy()
        self.physical_lower[[0,1,3]] = np.exp(self.lower[[0,1,3]])
        self.physical_upper[[0,1,3]] = np.exp(self.upper[[0,1,3]])
        self.parent_prior_sha256 = conditional_prior.metadata()['prior_sha256']

    def sample(self, n, rng):
        if type(n) is not int or n<1:
            raise ValueError('A positive integer sample count is required')
        x = rng.uniform(self.lower,self.upper,size=(n,4))
        x[:,1] = np.log(rng.uniform(self.physical_lower[1],self.physical_upper[1],size=n))
        x[:,3] = np.log(rng.uniform(self.physical_lower[3],self.physical_upper[3],size=n))
        return MaterialState(x)

    def log_prob(self, state):
        if type(state) is not MaterialState:
            raise TypeError('Explicit four-dimensional MaterialState required')
        x = np.asarray(state.coordinates,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=4:
            raise ValueError('Expected intrinsic coordinates [N,4]')
        valid = np.isfinite(x).all(1)&(x>=self.lower).all(1)&(x<=self.upper).all(1)
        constant = (-math.log(self.upper[0]-self.lower[0])
                    -math.log(self.upper[2]-self.lower[2])
                    -math.log(self.physical_upper[1]-self.physical_lower[1])
                    -math.log(self.physical_upper[3]-self.physical_lower[3]))
        return np.where(valid,constant+x[:,1]+x[:,3],-np.inf)

    def physical_coordinates(self,state):
        if not np.isfinite(self.log_prob(state)).all():
            raise ValueError('Material state lies outside the declared prior')
        result = np.asarray(state.coordinates,dtype=np.float64).copy()
        result[:,[0,1,3]] = np.exp(result[:,[0,1,3]])
        return result

    def metadata(self):
        return dict(parent_prior_sha256=self.parent_prior_sha256,
            intrinsic_order=['log_E_Pa','log_rho_kg_m3','nu','log_body_drag_s_inverse'],
            lower=self.lower.tolist(),upper=self.upper.tolist(),
            physical_order=['E_Pa','rho_kg_m3','nu','body_drag_s_inverse'],
            density='Normalized intrinsic density including +log_rho +log_drag Jacobians',
            scope='Known-support/hardware oracle only; not an unknown-boundary posterior',
            periodic_indices=[])


def realize_known_support_material(public, template, state):
    """Change material only, preserving every fixed support/hardware nuisance.

    Rebuild the full retained intrinsic state rather than overwriting only its
    displayed material labels, which would invalidate the physical decoder.
    """
    parent = make_prior(public)
    validate_realized_hypothesis(known_hardware_from_public(public),template)
    marginal = MaterialMarginalPrior(parent)
    marginal.physical_coordinates(state)
    raw = deepcopy(template['retained_prior_state'])
    base = PriorState(**{name:np.asarray(raw[name]) for name in PriorState.__dataclass_fields__})
    result = []
    for row in np.asarray(state.coordinates):
        x = base.coordinates.copy()
        x[0,:4] = row
        result.append(realize_hypothesis(public,base.with_coordinates(x),parent))
    return result
