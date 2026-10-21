"""Independent float64 finite-strain dynamics, not native Isaac simulation.

Constrained point-mass variational implicit Euler: exact rigid patch/auxiliary
positions, positive-J elasticity, body-only viscous damping. Step velocities
are CHORDS. They are deliberately not projected onto rigid tangent velocities;
finite-step rigid energy/angular momentum conservation is not claimed. The
predeclared four-BC gate must pass before any reference/data qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numpy as np
import torch

try:
    from campaign_linear_fem import assemble_tetrahedra, tensor, skew
    from campaign_nonlinear_fem import (FiniteRigidPatchMap, InvalidConfiguration,
        neo_hookean_energy, rotation_matrix, equilibrium_metrics)
    from campaign_rigid_inertia import auxiliary_mass_points
except ImportError:
    from experiments.campaign_linear_fem import assemble_tetrahedra, tensor, skew
    from experiments.campaign_nonlinear_fem import (FiniteRigidPatchMap, InvalidConfiguration,
        neo_hookean_energy, rotation_matrix, equilibrium_metrics)
    from experiments.campaign_rigid_inertia import auxiliary_mass_points


@dataclass
class DynamicState:
    time_s: float
    positions: torch.Tensor
    auxiliary_positions: torch.Tensor
    velocities: torch.Tensor
    auxiliary_velocities: torch.Tensor
    handle_centers: torch.Tensor
    handle_rotations: torch.Tensor
    velocity_semantics: str = 'pointwise chord over preceding step; initial state may supply physical tangent velocities'
    reduced_position_coordinates: torch.Tensor | None = None


def finite_array(value, shape, name):
    value = tensor(value if isinstance(value,torch.Tensor) else np.asarray(value))
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name}: finite array of shape {shape} required')
    return value


class NonlinearDynamics:
    """All four BC families share exact prescribed-node elimination.

    ``fixture_target(t)`` returns absolute fixture positions, with a common
    translation of their rest patch; ``fixture_velocity(t)`` is used ONLY for
    the initial state. Thereafter all stored point velocities are chords.
    Two fixed patches are supplied as their union. The hardware COM offsets
    are relative to each reference handle origin; inertia is about that COM.
    No contacts, native damping equivalence, or continuum accuracy is implied.
    """
    def __init__(self, points, tets, young, poisson, density, *, fixture_nodes=(),
                 load_patches=(), load_nodes=(), handle_masses=(), handle_inertias=(),
                 handle_com_offsets=None, drag=0., boundary_kind='free',
                 fixture_target=None, fixture_velocity=None):
        torch.set_num_threads(2)
        self.points = tensor(points).clone()
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not bool(torch.isfinite(self.points).all()):
            raise ValueError('Finite node/xyz rest points required')
        ids = np.asarray(tets)
        if ids.ndim != 2 or ids.shape[1] != 4 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Integer tetrahedron array required')
        if not len(ids) or np.any(ids < 0) or np.any(ids >= len(self.points)):
            raise ValueError('Nonempty in-range tetrahedra required')
        self.tets = ids.copy()
        self.young, self.poisson, self.density = float(young), float(poisson), float(density)
        if not all(math.isfinite(v) for v in [self.young,self.poisson,self.density]):
            raise ValueError('Finite material parameters required')
        self.drag = float(drag)
        if not np.isfinite(self.drag) or self.drag < 0:
            raise ValueError('Nonnegative finite body-only damping required')
        self.K, mass, self.gradients, self.volumes = assemble_tetrahedra(
            self.points, ids, self.young, self.poisson, self.density)
        self.body_mass = mass.reshape(-1,3)[:,0].clone()
        if not bool((self.body_mass > 0).all()):
            raise ValueError('Every body node needs strictly positive declared lumped mass')
        fixture_nodes = list(map(int, fixture_nodes))
        if len(set(fixture_nodes)) != len(fixture_nodes):
            raise ValueError('Duplicate fixture nodes')
        if boundary_kind not in ['free','point_pin','fixed_patch','two_patches','moving_patch']:
            raise ValueError('Declared boundary family required')
        if (boundary_kind == 'free') != (len(fixture_nodes) == 0):
            raise ValueError('Free family requires no fixture; other families require fixtures')
        if boundary_kind == 'point_pin' and len(fixture_nodes) != 1:
            raise ValueError('Point pin requires exactly one node')
        if boundary_kind == 'moving_patch' and fixture_target is None:
            raise ValueError('Moving patch requires an explicit position law')
        if fixture_target is not None and boundary_kind != 'moving_patch':
            raise ValueError('A position law is only declared for the moving-patch family')
        self.boundary_kind = boundary_kind
        self.mapping = FiniteRigidPatchMap(self.points, fixture_nodes, load_patches, load_nodes)
        self.fixture_nodes = torch.as_tensor(fixture_nodes, dtype=torch.long)
        self.origins = list(map(int, load_nodes))
        self.handle_dofs = self.mapping.handle_dofs
        self.patches = self.mapping.patches
        self.size = self.mapping.size
        self.handle_count = len(self.origins)
        if len(handle_masses) != self.handle_count or len(handle_inertias) != self.handle_count:
            raise ValueError('Mass and full COM inertia required for every rigid handle')
        self.com_offsets = finite_array(np.zeros((self.handle_count,3)) if handle_com_offsets is None
            else handle_com_offsets, (self.handle_count,3), 'handle COM offsets')
        offsets, masses = [], []
        for j,(m,I) in enumerate(zip(handle_masses, handle_inertias)):
            r,mi = auxiliary_mass_points(m,I)
            offsets.append(tensor(r)+self.com_offsets[j]); masses.append(tensor(mi))
        self.auxiliary_offsets = torch.stack(offsets) if offsets else torch.zeros((0,6,3),dtype=torch.float64)
        self.auxiliary_mass = torch.cat(masses) if masses else torch.zeros(0,dtype=torch.float64)
        self.total_mass = float(self.body_mass.sum()+self.auxiliary_mass.sum())
        fixed = set(fixture_nodes)|set(sum([list(map(int,p)) for p in load_patches],[]))
        self.free_nodes = torch.as_tensor([i for i in range(len(self.points)) if i not in fixed],dtype=torch.long)
        self.fixture_target = fixture_target
        self.fixture_velocity = fixture_velocity
        self.length_scale = float((self.points.max(0).values-self.points.min(0).values).max())
        if self.length_scale <= 0:
            raise ValueError('Positive physical length scale required')
        self._factor_cache = {}

    def targets(self, time):
        value = self.points[self.fixture_nodes] if self.fixture_target is None else tensor(self.fixture_target(float(time)))
        value = finite_array(value,(len(self.fixture_nodes),3),'fixture target')
        if len(value):
            shift = value-self.points[self.fixture_nodes]
            if float((shift-shift[:1]).abs().max()) > 1e-12:
                raise ValueError('Only a common prescribed translation of fixture nodes is declared')
        return value

    def configuration(self, q, base_rotations, targets):
        """Rotation coordinates are local LEFT increments, reset every step."""
        x = self.points+(self.mapping.A@q).reshape(-1,3)
        centers, rotations, auxiliary = [], [], []
        for j,(patch,origin,dofs) in enumerate(zip(self.patches,self.origins,self.handle_dofs)):
            R = rotation_matrix(q[dofs[3:]])@base_rotations[j]
            c = self.points[origin]+q[dofs[:3]]
            x = x.index_copy(0,patch,c+(self.points[patch]-self.points[origin])@R.T)
            centers.append(c); rotations.append(R)
            auxiliary.append(c+self.auxiliary_offsets[j]@R.T)
        x = x.index_copy(0,self.fixture_nodes,targets)
        return (x, torch.cat(auxiliary) if auxiliary else torch.zeros((0,3),dtype=torch.float64),
                torch.stack(centers) if centers else torch.zeros((0,3),dtype=torch.float64),
                torch.stack(rotations) if rotations else torch.zeros((0,3,3),dtype=torch.float64))

    def coordinates(self, state):
        if state.reduced_position_coordinates is not None:
            return finite_array(state.reduced_position_coordinates,(self.size,),'state reduced position coordinates').clone()
        q = torch.zeros(self.size,dtype=torch.float64)
        q[:3*len(self.free_nodes)] = (state.positions[self.free_nodes]-self.points[self.free_nodes]).flatten()
        for j,dofs in enumerate(self.handle_dofs):
            q[dofs[:3]] = state.handle_centers[j]-self.points[self.origins[j]]
        return q

    def increments(self, q, old_q, state, targets):
        """Same finite map, subtracting reduced positions before world offsets.

        Avoid M/h² amplification of cancellation in absolute world positions.
        Stored absolute positions may differ from old+increment by rounding;
        that representational discrepancy is explicitly reported, not hidden.
        """
        dq = q-old_q
        dx = (self.mapping.A@dq).reshape(-1,3)
        auxiliaries,centers = [],[]
        identity = torch.eye(3,dtype=torch.float64)
        for j,(patch,origin,dofs) in enumerate(zip(self.patches,self.origins,self.handle_dofs)):
            delta_rotation = rotation_matrix(q[dofs[3:]])-identity
            dc = dq[dofs[:3]]
            local = (self.points[patch]-self.points[origin])@state.handle_rotations[j].T
            dx = dx.index_copy(0,patch,dc+local@delta_rotation.T)
            r = self.auxiliary_offsets[j]@state.handle_rotations[j].T
            auxiliaries.append(dc+r@delta_rotation.T);centers.append(dc)
        dx = dx.index_copy(0,self.fixture_nodes,targets-state.positions[self.fixture_nodes])
        return (dx,torch.cat(auxiliaries) if auxiliaries else torch.zeros((0,3),dtype=torch.float64),
                torch.stack(centers) if centers else torch.zeros((0,3),dtype=torch.float64))

    def elastic(self, x):
        return neo_hookean_energy(x,self.points,self.tets,self.gradients,self.volumes,self.young,self.poisson)

    def initial_state(self, *, positions=None, velocities=None, handle_centers=None,
                      handle_rotations=None, handle_linear_velocities=None,
                      handle_angular_velocities=None, time_s=0.):
        if not math.isfinite(float(time_s)):
            raise ValueError('Finite initial physical time required')
        H = self.handle_count
        centers = self.points[self.origins].clone() if handle_centers is None else finite_array(handle_centers,(H,3),'handle centers')
        rotations = torch.eye(3,dtype=torch.float64).repeat(H,1,1) if handle_rotations is None else finite_array(handle_rotations,(H,3,3),'handle rotations')
        for R in rotations:
            if not torch.allclose(R.T@R,torch.eye(3,dtype=torch.float64),atol=1e-12,rtol=0) or abs(float(torch.linalg.det(R))-1)>1e-12:
                raise ValueError('Proper finite SO(3) handle rotation required')
        x = self.points.clone() if positions is None else finite_array(positions,self.points.shape,'initial positions').clone()
        v = torch.zeros_like(x) if velocities is None else finite_array(velocities,x.shape,'initial velocities').clone()
        lin = finite_array(np.zeros((H,3)) if handle_linear_velocities is None else handle_linear_velocities,(H,3),'initial handle linear velocities')
        ang = finite_array(np.zeros((H,3)) if handle_angular_velocities is None else handle_angular_velocities,(H,3),'initial handle angular velocities')
        auxiliary, aux_v = [], []
        for j,(patch,origin) in enumerate(zip(self.patches,self.origins)):
            local = (self.points[patch]-self.points[origin])@rotations[j].T
            expected = centers[j]+local
            tangent = lin[j]+torch.linalg.cross(ang[j].expand_as(local),local)
            if positions is not None and not torch.allclose(x[patch],expected,atol=1e-12,rtol=0):
                raise ValueError('Initial nodes are incompatible with the exact rigid patch')
            if velocities is not None and not torch.allclose(v[patch],tangent,atol=1e-12,rtol=0):
                raise ValueError('Initial velocities are incompatible with supplied rigid tangent velocities')
            x[patch] = expected; v[patch] = tangent
            r = self.auxiliary_offsets[j]@rotations[j].T
            auxiliary.append(centers[j]+r)
            aux_v.append(lin[j]+torch.linalg.cross(ang[j].expand_as(r),r))
        targets = self.targets(time_s)
        if positions is not None and not torch.allclose(x[self.fixture_nodes],targets,atol=1e-12,rtol=0):
            raise ValueError('Initial fixture positions do not match the prescribed law')
        x[self.fixture_nodes] = targets
        if self.fixture_velocity is not None:
            fv = finite_array(self.fixture_velocity(float(time_s)),targets.shape,'initial fixture velocity')
            if velocities is not None and not torch.allclose(v[self.fixture_nodes],fv,atol=1e-12,rtol=0):
                raise ValueError('Initial fixture velocities do not match the prescribed law')
            v[self.fixture_nodes] = fv
        elif len(self.fixture_nodes):
            if velocities is not None and bool((v[self.fixture_nodes].abs()>1e-12).any()):
                raise ValueError('Nonzero initial fixture velocity requires its explicit prescribed law')
            v[self.fixture_nodes] = 0
        self.elastic(x)  # Reject nonpositive-J initial configurations; no repair.
        return DynamicState(float(time_s),x,torch.cat(auxiliary) if auxiliary else torch.zeros((0,3),dtype=torch.float64),
            v,torch.cat(aux_v) if aux_v else torch.zeros((0,3),dtype=torch.float64),centers.clone(),rotations.clone(),
            'initial physical tangent velocities; subsequent velocities are chords')

    def validate_state(self, state):
        if not isinstance(state,DynamicState) or not math.isfinite(state.time_s):
            raise ValueError('Finite DynamicState required')
        for value,shape,name in [(state.positions,self.points.shape,'state positions'),
            (state.velocities,self.points.shape,'state chord velocities'),
            (state.auxiliary_positions,(6*self.handle_count,3),'state auxiliary positions'),
            (state.auxiliary_velocities,(6*self.handle_count,3),'state auxiliary chord velocities'),
            (state.handle_centers,(self.handle_count,3),'state centers'),
            (state.handle_rotations,(self.handle_count,3,3),'state rotations')]:
            finite_array(value,shape,name)
        for R in state.handle_rotations:
            if not torch.allclose(R.T@R,torch.eye(3,dtype=torch.float64),atol=2e-12,rtol=0) or abs(float(torch.linalg.det(R))-1)>2e-12:
                raise ValueError('State rotation is not SO(3); no projection')
        q = self.coordinates(state)
        x,Y,_,_ = self.configuration(q,state.handle_rotations,self.targets(state.time_s))
        if not torch.allclose(x,state.positions,atol=1e-12,rtol=0) or not torch.allclose(Y,state.auxiliary_positions,atol=1e-12,rtol=0):
            raise ValueError('State positions violate exact patch/fixture kinematics')
        self.elastic(x)

    def _jacobians(self, state):
        J = self.mapping.A.clone()
        JA = torch.zeros((18*self.handle_count,self.size),dtype=torch.float64)
        for j,(patch,origin,dofs) in enumerate(zip(self.patches,self.origins,self.handle_dofs)):
            R = state.handle_rotations[j]
            for node in patch.tolist():
                J[3*node:3*node+3,dofs[3:]] = -skew(R@(self.points[node]-self.points[origin]))
            for k,r in enumerate(self.auxiliary_offsets[j]):
                start = 18*j+3*k
                JA[start:start+3,dofs[:3]] = torch.eye(3,dtype=torch.float64)
                JA[start:start+3,dofs[3:]] = -skew(R@r)
        return J,JA

    def preconditioner(self, state, h):
        """Rest stiffness + physical first-state mass metric cached per h.

        This metric is a preconditioner only. The objective retains the full
        finite-pose configuration-dependent inertia; no force is replaced by
        a linearization. A fixed positive mass metric permits pin/free modes.
        """
        if not self.size:
            return None
        key = float(h)
        if key in self._factor_cache:
            return self._factor_cache[key]
        J,JA = self._jacobians(state)
        Mb = J.T@(self.body_mass.repeat_interleave(3)[:,None]*J)
        Ma = JA.T@(self.auxiliary_mass.repeat_interleave(3)[:,None]*JA)
        P = J.T@self.K@J+(Mb+Ma)/(h*h)+self.drag*Mb/h
        P = .5*(P+P.T)
        factor,info = torch.linalg.cholesky_ex(P)
        if int(info) != 0 or not bool(torch.isfinite(factor).all()):
            raise InvalidConfiguration('Dynamic rest stiffness + mass failed Cholesky; no regularizer')
        self._factor_cache[key] = factor
        return factor

    def invariants(self, state):
        U,J = self.elastic(state.positions)
        Tb = .5*(self.body_mass[:,None]*state.velocities.square()).sum()
        Ta = .5*(self.auxiliary_mass[:,None]*state.auxiliary_velocities.square()).sum()
        Pb = self.body_mass[:,None]*state.velocities
        Pa = self.auxiliary_mass[:,None]*state.auxiliary_velocities
        return {'elastic_energy_J':float(U),'chord_kinetic_energy_J':float(Tb+Ta),
            'energy_J':float(U+Tb+Ta),
            'point_linear_momentum_Ns':(Pb.sum(0)+Pa.sum(0)).tolist(),
            'chord_angular_momentum_about_world_origin_Nms':(
                torch.linalg.cross(state.positions,Pb).sum(0)+torch.linalg.cross(state.auxiliary_positions,Pa).sum(0)).tolist(),
            'minimum_J':float(J.min())}

    def step(self, state, h, *, handle_forces=None, nodal_forces=None,
             relative_tolerance=1e-7, absolute_tolerance_N=1e-10, maximum_iterations=250):
        self.validate_state(state)
        h = float(h)
        if (not all(math.isfinite(float(x)) and x > 0 for x in [h,relative_tolerance,absolute_tolerance_N])
                or type(maximum_iterations) is not int or maximum_iterations < 0):
            raise ValueError('Positive finite step and nonlinear residual tolerances required')
        F = finite_array(np.zeros((self.handle_count,3)) if handle_forces is None else handle_forces,
            (self.handle_count,3),'held handle forces')
        f = finite_array(np.zeros(self.points.shape) if nodal_forces is None else nodal_forces,
            self.points.shape,'held nodal forces')
        target = self.targets(state.time_s+h)
        q = self.coordinates(state)
        factor = self.preconditioner(state,h)
        old_q = q.clone()
        applied = self.mapping.A.T@f.flatten()
        for j,dofs in enumerate(self.handle_dofs):
            applied[dofs[:3]] += F[j]
        # All components are physical Newtons; do not mix raw Nm coordinates
        # into a norm or normalize by absolute-position inertial RHS terms.
        force_scale = float(torch.sqrt(F.square().sum()+f.square().sum()))
        tolerance = max(float(absolute_tolerance_N),float(relative_tolerance)*force_scale)
        mu = self.young/(2*(1+self.poisson))
        roundoff_window = 128*np.finfo(float).eps*max(mu*float(self.volumes.sum()),1e-30)

        def evaluate(value):
            value = value.detach().requires_grad_(True)
            x,Y,c,R = self.configuration(value,state.handle_rotations,target)
            dx,dY,dc = self.increments(value,old_q,state,target)
            U,J = self.elastic(x)
            objective = (self.body_mass[:,None]*(dx-h*state.velocities).square()).sum()/(2*h*h)
            objective += (self.auxiliary_mass[:,None]*(dY-h*state.auxiliary_velocities).square()).sum()/(2*h*h)
            objective += U+self.drag*(self.body_mass[:,None]*dx.square()).sum()/(2*h)
            objective -= (F*dc).sum()+(f*dx).sum()
            gradient = torch.autograd.grad(objective,value)[0]
            if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(objective)):
                raise InvalidConfiguration('Nonfinite incremental energy or gradient')
            metrics = equilibrium_metrics(gradient.detach(),applied,self.handle_dofs,
                self.length_scale,value.detach())
            metrics['relative_scaled_generalized_residual'] = metrics['scaled_generalized_residual_norm_N']/max(force_scale,1e-30)
            return float(objective.detach()),gradient.detach(),metrics,float(J.min().detach()),(x.detach(),Y.detach(),c.detach(),R.detach())

        history,invalid = [],0
        for iteration in range(maximum_iterations+1):
            objective,gradient,metrics,minJ,configuration = evaluate(q)
            history.append({'iteration':iteration,'incremental_potential_J':objective,'minimum_J':minJ,**metrics})
            if metrics['scaled_generalized_residual_norm_N'] <= tolerance:
                break
            if iteration == maximum_iterations:
                raise RuntimeError('Dynamic nonlinear iteration cap; no replacement: '+json.dumps(history[-1]))
            direction = -torch.cholesky_solve(gradient[:,None],factor)[:,0]
            slope = float(gradient@direction)
            if not math.isfinite(slope) or slope >= 0:
                raise RuntimeError('Dynamic preconditioned direction is not descent')
            alpha,accepted = 1.,False
            for _ in range(48):
                trial = q+alpha*direction
                try:
                    candidate,_,candidate_metrics,_,_ = evaluate(trial)
                except InvalidConfiguration:
                    invalid += 1; alpha *= .5; continue
                armijo = candidate <= objective+1e-4*alpha*slope
                at_floor = abs(alpha*slope) <= roundoff_window
                falling = candidate_metrics['scaled_generalized_residual_norm_N'] < metrics['scaled_generalized_residual_norm_N']*(1-1e-4*alpha)
                if armijo or (at_floor and candidate <= objective+roundoff_window and falling):
                    q = trial.detach(); accepted = True
                    history[-1].update(accepted_step_length=alpha,accepted_by_roundoff_residual_rule=bool(not armijo))
                    break
                alpha *= .5
            if not accepted:
                raise RuntimeError('Positive-J dynamic energy/residual line search failed: '+json.dumps(history[-1]))
        x,Y,c,R = configuration
        dx,dY,dc = [v.detach() for v in self.increments(q,old_q,state,target)]
        next_coordinates = q.detach().clone()
        for dofs in self.handle_dofs:
            next_coordinates[dofs[3:]] = 0
        next_state = DynamicState(state.time_s+h,x,Y,dx/h,dY/h,c,R,
            'finite pose-map pointwise chord; absolute-position subtraction agrees to recorded roundoff',next_coordinates)
        ledger = self._balance(state,next_state,h,F,f,dx,dY,dc)
        discrepancies = [(dx-(x-state.positions)).abs().max()]
        if self.handle_count:
            discrepancies.extend([(dY-(Y-state.auxiliary_positions)).abs().max(),
                                  (dc-(c-state.handle_centers)).abs().max()])
        ledger.update(converged=True,residual_tolerance_N=tolerance,minimum_J=minJ,
            invalid_J_trials_rejected=invalid,nonlinear_history=history,
            point_velocity_semantics=next_state.velocity_semantics,
            position_increment_representation_roundoff_m=float(torch.stack(discrepancies).max()),
            native_equivalence=False,continuum_accuracy_established=False)
        return next_state,ledger

    def _balance(self, old, new, h, F, f, dx, dY, dc):
        """Discrete identities, including inertial/drag fixture reactions.

        Curved constraint chord work is measured, not set to zero. The energy
        identity does not assert that its individual time-discretization terms
        vanish, or that finite-step chord angular momentum is conserved.
        """
        x = new.positions.detach().requires_grad_(True)
        U,_ = self.elastic(x)
        grad = torch.autograd.grad(U,x)[0].detach()
        old_inv,new_inv = self.invariants(old),self.invariants(new)
        dv,dva = new.velocities-old.velocities,new.auxiliary_velocities-old.auxiliary_velocities
        body_drag = self.drag*self.body_mass[:,None]*new.velocities
        residual = self.body_mass[:,None]*dv/h+body_drag+grad-f
        residual_aux = self.auxiliary_mass[:,None]*dva/h
        reaction = residual[self.fixture_nodes]
        support_work = (reaction*dx[self.fixture_nodes]).sum()
        handle_work = (F*dc).sum()
        nodal_work = (f*dx).sum()
        damping = h*self.drag*(self.body_mass[:,None]*new.velocities.square()).sum()
        inertial_loss = .5*((self.body_mass[:,None]*dv.square()).sum()+
            (self.auxiliary_mass[:,None]*dva.square()).sum())
        elastic_defect = (grad*dx).sum()-(new_inv['elastic_energy_J']-old_inv['elastic_energy_J'])
        constraint_work = (residual*dx).sum()+(residual_aux*dY).sum()-handle_work-support_work
        defect = new_inv['energy_J']-old_inv['energy_J']-float(handle_work+nodal_work+support_work-damping)
        identity = defect-float(constraint_work-inertial_loss-elastic_defect)
        linear_balance = tensor(new_inv['point_linear_momentum_Ns'])-tensor(old_inv['point_linear_momentum_Ns'])-h*(F.sum(0)+f.sum(0)+reaction.sum(0)-body_drag.sum(0))
        endpoint_torque = torch.linalg.cross(new.handle_centers,F).sum(0)+torch.linalg.cross(new.positions,f-body_drag).sum(0)
        endpoint_torque += torch.linalg.cross(new.positions[self.fixture_nodes],reaction).sum(0)
        angular_defect = tensor(new_inv['chord_angular_momentum_about_world_origin_Nms'])-tensor(old_inv['chord_angular_momentum_about_world_origin_Nms'])-h*endpoint_torque
        return {'start_time_s':old.time_s,'end_time_s':new.time_s,'step_s':h,
            'held_handle_forces_N':F.tolist(),'held_nodal_forces_N':f.tolist(),
            'old_invariants':old_inv,'new_invariants':new_inv,
            'fixture_reactions_inertia_elastic_drag_minus_external_N':reaction.tolist(),
            'support_work_J':float(support_work),'held_handle_work_J':float(handle_work),
            'held_nodal_work_J':float(nodal_work),'body_viscous_dissipation_J':float(damping),
            'body_drag_impulse_Ns':(h*body_drag.sum(0)).tolist(),
            'implicit_velocity_increment_loss_J':float(inertial_loss),
            'backward_elastic_gradient_defect_J':float(elastic_defect),
            'constraint_chord_work_J':float(constraint_work),
            'energy_balance_time_discretization_defect_J':defect,
            'energy_algebra_identity_residual_J':identity,
            'linear_momentum_balance_error_Ns':linear_balance.tolist(),
            'endpoint_angular_momentum_discretization_defect_Nms':angular_defect.tolist()}
