"""Small-strain physical baseline for explicit new hypotheses and known clocks.

Rest tangent stiffness, body mass/drag, and full rigid-hardware auxiliary mass
are taken from the existing physical assembly. This module never calls the
nonlinear step or fits trajectory labels. Infinitesimal handle rotations are
linear coordinates: predicted patches are NOT finite SO(3) rigid motions.
Large rotation/strain, contact, and continuum accuracy are not claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class LinearState:
    time_s: float
    coordinates: torch.Tensor
    generalized_velocity: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    auxiliary_positions: torch.Tensor
    auxiliary_velocities: torch.Tensor
    handle_centers: torch.Tensor
    handle_rotation_coordinates: torch.Tensor


class LinearHypothesisDynamics:
    """Exact reduced implicit Euler for the linearized physical system.

    x=X+Aq+b(t), Y=Y0+JA0*q, Krr=A.T*K*A,
    Mrr=A.T*Mb*A+JA0.T*Ma*JA0, Crr=drag*A.T*Mb*A.
    A Cholesky of Mrr/h²+Crr/h+Krr is cached per h. Krr itself may have
    free/pin zero modes. No stiffness repair or diagonal regularizer is added.
    """
    def __init__(self,physical_model):
        if physical_model.points.device.type!='cpu' or physical_model.points.dtype!=torch.float64:
            raise ValueError('CPU float64 physical assembly required')
        self.model=physical_model
        state=physical_model.initial_state()
        self.A,self.JA=tuple(value.detach().clone() for value in physical_model._jacobians(state))
        self.points=physical_model.points.detach().clone()
        self.auxiliary_reference=state.auxiliary_positions.detach().clone()
        self.body_mass=physical_model.body_mass.repeat_interleave(3).detach().clone()
        self.auxiliary_mass=physical_model.auxiliary_mass.repeat_interleave(3).detach().clone()
        self.K=physical_model.K.detach().clone();self.drag=physical_model.drag
        self.size=self.A.shape[1];self.handle_dofs=physical_model.handle_dofs
        self.Mbody=self.A.T@(self.body_mass[:,None]*self.A)
        self.Maux=self.JA.T@(self.auxiliary_mass[:,None]*self.JA)
        self.M=self.Mbody+self.Maux
        self.C=self.drag*self.Mbody
        self.Kr=self.A.T@self.K@self.A
        self._factors={};self.factorizations=0
        if not torch.allclose(self._lift(0.),torch.zeros_like(self.points),atol=1e-12,rtol=0):
            raise ValueError('Baseline is initialized from the common undeformed rest hypothesis')

    @classmethod
    def from_hypothesis(cls,hardware,realized_hypothesis):
        """Full known hardware plus supplied candidate/diagnostic physical truth."""
        try:
            from campaign_hypothesis_dynamics import build_model
        except ImportError:
            from experiments.campaign_hypothesis_dynamics import build_model
        physical,_=build_model(hardware,realized_hypothesis)
        return cls(physical)

    def _lift(self,time_s):
        value=torch.zeros_like(self.points)
        nodes=self.model.fixture_nodes
        value[nodes]=self.model.targets(time_s)-self.points[nodes]
        return value

    def _array(self,value,shape,name):
        # Public command arrays deliberately have immutable byte backing.
        result=value.to(dtype=torch.float64,device='cpu') if isinstance(value,torch.Tensor) else torch.tensor(value,dtype=torch.float64,device='cpu')
        if tuple(result.shape)!=tuple(shape) or not bool(torch.isfinite(result).all()):
            raise ValueError('Finite '+name+' with declared shape required')
        return result

    def _state(self,time,q,v,body_velocity,aux_velocity):
        displacement=(self.A@q).reshape_as(self.points)+self._lift(time)
        center=torch.stack([self.points[origin]+q[dofs[:3]] for origin,dofs in zip(self.model.origins,self.handle_dofs)]) if self.handle_dofs else self.points.new_zeros((0,3))
        rotations=torch.stack([q[dofs[3:]] for dofs in self.handle_dofs]) if self.handle_dofs else self.points.new_zeros((0,3))
        return LinearState(time,q,v,self.points+displacement,body_velocity,
            self.auxiliary_reference+(self.JA@q).reshape_as(self.auxiliary_reference),aux_velocity,center,rotations)

    def initial_state(self,*,coordinates=None,generalized_velocity=None):
        q=torch.zeros(self.size,dtype=torch.float64) if coordinates is None else self._array(coordinates,(self.size,),'initial coordinates').clone()
        v=torch.zeros_like(q) if generalized_velocity is None else self._array(generalized_velocity,(self.size,),'initial generalized velocity').clone()
        body=(self.A@v).reshape_as(self.points)
        if self.model.fixture_velocity is not None:
            body[self.model.fixture_nodes]=self._array(self.model.fixture_velocity(0.),(len(self.model.fixture_nodes),3),'initial prescribed velocity')
        return self._state(0.,q,v,body,(self.JA@v).reshape_as(self.auxiliary_reference))

    def factor(self,h):
        h=float(h)
        if not math.isfinite(h) or h<=0:raise ValueError('Finite positive timestep required')
        if h not in self._factors:
            effective=self.M/(h*h)+self.C/h+self.Kr
            effective=(effective+effective.T)/2  # only roundoff symmetry, no eigenvalue repair
            factor,info=torch.linalg.cholesky_ex(effective)
            if int(info)!=0 or not bool(torch.isfinite(factor).all()):
                raise ValueError('Physical effective mass/stiffness is not SPD; no regularizer')
            self._factors[h]=factor;self.factorizations+=1
        return self._factors[h]

    def step(self,state,h,*,handle_forces=None,nodal_forces=None):
        """One linear solve with exact moving Dirichlet lift and chord velocity."""
        h=float(h);factor=self.factor(h)
        F=self._array(np.zeros((len(self.handle_dofs),3)) if handle_forces is None else handle_forces,
            (len(self.handle_dofs),3),'world handle forces')
        f=self._array(np.zeros(tuple(self.points.shape)) if nodal_forces is None else nodal_forces,
            self.points.shape,'world nodal forces')
        old_q=self._array(state.coordinates,(self.size,),'old reduced coordinates')
        old_body_v=self._array(state.velocities,self.points.shape,'old body chord velocity').flatten()
        old_aux_v=self._array(state.auxiliary_velocities,self.auxiliary_reference.shape,'old auxiliary chord velocity').flatten()
        next_time=state.time_s+h
        old_b=self._lift(state.time_s).flatten();new_b=self._lift(next_time).flatten();db=new_b-old_b
        applied=self.A.T@f.flatten()
        for force,dofs in zip(F,self.handle_dofs):applied[dofs[:3]]+=force
        rhs=applied+self.M@old_q/(h*h)+self.C@old_q/h
        rhs+=self.A.T@(self.body_mass*old_body_v)/h+self.JA.T@(self.auxiliary_mass*old_aux_v)/h
        rhs-=self.A.T@(self.body_mass*db)/(h*h)+self.drag*self.A.T@(self.body_mass*db)/h
        rhs-=self.A.T@(self.K@new_b)
        q=torch.cholesky_solve(rhs[:,None],factor)[:,0]
        if not bool(torch.isfinite(q).all()):raise FloatingPointError('Nonfinite linear solution')
        v=(q-old_q)/h
        body_v=((self.A@(q-old_q)+db)/h).reshape_as(self.points)
        aux_v=(self.JA@v).reshape_as(self.auxiliary_reference)
        return self._state(next_time,q,v,body_v,aux_v)

    def predict(self,clock,*,integration_hz=120):
        """Run an explicit known clock; returns memory arrays, no file writes.

        This is a supplied-physics small-strain baseline, not a new nonlinear
        reference capture. Public command bytes stay independent of timestep.
        """
        try:
            from campaign_hypothesis_dynamics import validate_clock
        except ImportError:
            from experiments.campaign_hypothesis_dynamics import validate_clock
        refined=validate_clock(clock,integration_hz,len(self.points),len(self.handle_dofs))
        state=self.initial_state();states=[state]
        outputs=set(refined.output_step_indices.tolist())
        for i in range(len(refined.parent_command_indices)):
            nodal=refined.nodal_forces_world_N[i]
            state=self.step(state,1/integration_hz,handle_forces=refined.handle_forces_world_N[i],
                nodal_forces=None if len(nodal)==0 else nodal)
            if i+1 in outputs:states.append(state)
        return {'positions_m':torch.stack([s.positions for s in states]),
            'velocities_point_chord_m_s':torch.stack([s.velocities for s in states]),
            'reduced_coordinates':torch.stack([s.coordinates for s in states]),
            'generalized_velocities':torch.stack([s.generalized_velocity for s in states]),
            'handle_centers_m':torch.stack([s.handle_centers for s in states]),
            'handle_rotation_coordinates_rad':torch.stack([s.handle_rotation_coordinates for s in states]),
            'actual_state_times_s':torch.tensor([s.time_s for s in states],dtype=torch.float64),
            'requested_output_times_s':clock.output_times_s.copy(),
            'public_command_manifest':clock.public_manifest(),
            'role':'linearized_small_strain_supplied_physics_baseline_not_finite_SO3_or_posterior'}
