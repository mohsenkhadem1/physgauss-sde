"""Independent finite-strain STATIC diagnostic, exclusively privileged TRAIN t0001.

Compressible log-J Neo-Hookean continuum, exact finite rigid patch kinematics,
and exact fixed-node elimination. This is not a PhysX reproduction, dynamic
prediction model, continuum convergence claim, or replacement dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import torch

try:
    from campaign_linear_fem import assemble_tetrahedra, condensation, checked_cholesky, tensor, skew
except ImportError:
    from experiments.campaign_linear_fem import assemble_tetrahedra, condensation, checked_cholesky, tensor, skew


class InvalidConfiguration(ValueError):
    pass


def determinant3(matrix):
    """Polynomial determinant, including differentiability at singular zero G."""
    a, b, c = matrix.unbind(-2)
    return (a * torch.linalg.cross(b, c)).sum(-1)


def rotation_matrix(omega):
    """SO(3) exponential with finite first/second derivatives at omega=0."""
    omega = tensor(omega)
    squared = omega.square().sum()
    angle = torch.sqrt(torch.clamp(squared, min=1e-12))
    a = torch.where(squared < 1e-8,
                    1-squared/6+squared.square()/120-squared**3/5040,
                    torch.sin(angle)/angle)
    b = torch.where(squared < 1e-8,
                    .5-squared/24+squared.square()/720-squared**3/40320,
                    (1-torch.cos(angle))/squared.clamp(min=1e-12))
    W = skew(omega)
    return torch.eye(3, dtype=torch.float64)+a*W+b*(W@W)


def rotation_left_jacobian(omega):
    """delta R R.T = skew(J_left(omega) delta omega)."""
    omega=tensor(omega);squared=omega.square().sum()
    angle=torch.sqrt(squared.clamp(min=1e-12))
    a=torch.where(squared<1e-8,.5-squared/24+squared.square()/720-squared**3/40320,
                  (1-torch.cos(angle))/squared.clamp(min=1e-12))
    b=torch.where(squared<1e-8,1/6-squared/120+squared.square()/5040-squared**3/362880,
                  (angle-torch.sin(angle))/(angle**3))
    W=skew(omega)
    return torch.eye(3,dtype=torch.float64)+a*W+b*(W@W)


def x_minus_log1p(x):
    # Avoid subtracting nearly equal first-order terms at small strain.
    series = sum(((-1.)**power)*x**power/power for power in range(2, 10))
    return torch.where(x.abs() < 1e-3, series, x-torch.log1p(x))


def neo_hookean_energy(positions, rest_points, tets, gradients, volumes, young, poisson,
                      minimum_J=1e-8):
    """V[mu/2*(tr(F.T F)-3)-mu*log J+lambda/2*(log J)^2].

    A polynomial det(I+G) and a stable x-log1p(x) evaluation avoid cancellation
    near the rest state. Invalid J is rejected before any logarithm; no energy
    extension, barrier substitution, inversion repair, or modulus adjustment.
    """
    if not (young > 0 and -1 < poisson < .5):
        raise ValueError('Invalid isotropic material')
    positions = tensor(positions); rest_points = tensor(rest_points)
    ids = torch.as_tensor(tets, dtype=torch.long)
    G = torch.einsum('eic,eid->ecd', (positions-rest_points)[ids], tensor(gradients))
    trace = G.diagonal(dim1=-2, dim2=-1).sum(-1)
    second = .5*(trace.square()-(G*G.transpose(-1, -2)).sum((-1, -2)))
    detG = determinant3(G)
    J_minus_one = trace+second+detG
    J = 1+J_minus_one
    if not bool(torch.isfinite(J).all()) or bool((J <= minimum_J).any()):
        raise InvalidConfiguration('Nonfinite or nonpositive/near-zero tet J; no inversion repair')
    logJ = torch.log1p(J_minus_one)
    mu = young/(2*(1+poisson))
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    density = mu*(.5*G.square().sum((-1,-2))-second-detG+x_minus_log1p(J_minus_one))+.5*lam*logJ.square()
    return (tensor(volumes)*density).sum(), J


class FiniteRigidPatchMap:
    """Same reduced coordinates as the linear map, rotations now exp(omega)."""
    def __init__(self, points, fixture_nodes, load_patches, load_nodes):
        self.points = tensor(points)
        self.A, self.handle_dofs = condensation(points, fixture_nodes, load_patches, load_nodes)
        self.patches = [torch.as_tensor(p, dtype=torch.long) for p in load_patches]
        self.origins = list(map(int, load_nodes))
        self.fixture_nodes = list(map(int, fixture_nodes))
        self.size = self.A.shape[1]

    def positions(self, z):
        z = tensor(z)
        # Replacing patch rows also removes the infinitesimal rotation term.
        result = self.points+(self.A@z).reshape(-1,3)
        for patch, origin, ids in zip(self.patches, self.origins, self.handle_dofs):
            R = rotation_matrix(z[ids[3:]])
            local = self.points[patch]-self.points[origin]
            values = self.points[origin]+z[ids[:3]]+local@R.T
            result = result.index_copy(0, patch, values)
        return result


def equilibrium_metrics(residual, applied, handle_dofs, length_scale, generalized_coordinates=None):
    """Report separate N/Nm and a declared length-scaled generalized residual."""
    residual=tensor(residual).detach();applied=tensor(applied).detach()
    if generalized_coordinates is not None:generalized_coordinates=tensor(generalized_coordinates).detach()
    physical = residual.clone(); torque_indices = []
    for ids in handle_dofs:
        if generalized_coordinates is None:raise ValueError('Finite handle rotations required for physical torque residual')
        # The gradient with respect to omega is J_left.T times world torque.
        J_left=rotation_left_jacobian(generalized_coordinates[ids[3:]])
        if abs(float(torch.linalg.det(J_left)))<1e-8:
            raise InvalidConfiguration('Rotation chart is singular; no torque-residual approximation')
        torque_indices.extend(ids[3:])
        physical[ids[3:]]=torch.linalg.solve(J_left.T,residual[ids[3:]])
    scaled=physical.clone()
    scaled[torque_indices]/=length_scale
    mask = torch.ones(len(residual), dtype=torch.bool)
    mask[torque_indices] = False
    applied_scaled = applied.clone()
    for ids in handle_dofs:
        applied_scaled[ids[3:]] /= length_scale
    force_scale = float(torch.linalg.vector_norm(applied_scaled))
    norm = float(torch.linalg.vector_norm(scaled))
    return {'scaled_generalized_residual_norm_N': norm,
            'relative_scaled_generalized_residual': norm/max(force_scale,1e-30),
            'maximum_translational_residual_N': float(residual[mask].abs().max()) if bool(mask.any()) else 0.,
            'maximum_handle_torque_residual_Nm': float(physical[torque_indices].abs().max()) if torque_indices else 0.,
            'maximum_axis_angle_gradient_Nm': float(residual[torque_indices].abs().max()) if torque_indices else 0.,
            'rotation_residual_length_scale_m': float(length_scale)}


def solve_static(kinematics, tets, gradients, volumes, young, poisson, linear_factor,
                 force, initial=None, relative_tolerance=1e-7, absolute_tolerance_N=1e-11,
                 maximum_iterations=250, length_scale=None):
    """Rest-stiffness preconditioned descent, explicit positive-J line search.

    No changed stiffness or added diagonal. A positive-definite rest matrix is
    only a preconditioner; the residual comes from the nonlinear exact energy.
    Energy-decrease acceptance is required, with a roundoff-scale energy window
    plus strictly decreasing residual near equilibrium. Every accepted iterate
    has J>minimum_J; failure returns a preserved error, never a replacement solve.
    """
    force = tensor(force)
    if any(bool((force[ids[3:]]!=0).any()) for ids in kinematics.handle_dofs):
        raise ValueError('Only zero applied handle torque supported; linear work in axis angle is not constant world torque')
    z = torch.zeros(kinematics.size,dtype=torch.float64) if initial is None else tensor(initial).clone()
    if length_scale is None:
        length_scale = float((kinematics.points.max(0).values-kinematics.points.min(0).values).max())
    if length_scale <= 0: raise ValueError('Positive residual length scale required')
    force_norm = float(torch.linalg.vector_norm(force))
    tolerance = max(absolute_tolerance_N,relative_tolerance*force_norm)
    mu = young/(2*(1+poisson))
    energy_roundoff_window = 128*np.finfo(float).eps*mu*float(tensor(volumes).sum())

    def evaluate(value):
        value = value.detach().requires_grad_(True)
        positions = kinematics.positions(value)
        elastic,J = neo_hookean_energy(positions, kinematics.points, tets, gradients, volumes,young,poisson)
        potential = elastic-force@value
        residual = torch.autograd.grad(potential,value)[0]
        if not bool(torch.isfinite(residual).all()): raise InvalidConfiguration('Nonfinite energy gradient')
        metrics = equilibrium_metrics(residual.detach(),force,kinematics.handle_dofs,length_scale,value.detach())
        return float(potential.detach()), residual.detach(),float(J.min().detach()),metrics,float(elastic.detach())

    history=[];invalid_trials=0
    for iteration in range(maximum_iterations+1):
        energy,residual,minJ,metrics,elastic = evaluate(z)
        history.append({'iteration':iteration,'potential_energy_J':energy,'elastic_energy_J':elastic,
                        'minimum_J':minJ,**metrics})
        if metrics['scaled_generalized_residual_norm_N']<=tolerance:
            return {'z':z,'positions':kinematics.positions(z).detach(),
                    'history':history,'converged':True,'residual_tolerance_N':tolerance,
                    'invalid_J_trials_rejected':invalid_trials,
                    'energy_roundoff_window_J':energy_roundoff_window,'metrics':metrics,
                    'minimum_J':minJ,'elastic_energy_J':elastic,'potential_energy_J':energy}
        if iteration == maximum_iterations:break
        direction = -torch.cholesky_solve(residual[:,None],linear_factor)[:,0]
        slope = float(residual@direction)
        if not np.isfinite(slope) or slope>=0: raise RuntimeError('Preconditioned direction is not a descent direction')
        alpha=1.;accepted=False
        for backtrack in range(45):
            trial=z+alpha*direction
            try:
                trial_energy,_,_,trial_metrics,_=evaluate(trial)
            except InvalidConfiguration:
                invalid_trials+=1;alpha*=.5;continue
            armijo=trial_energy<=energy+1e-4*alpha*slope
            # Only use the roundoff allowance when the remaining predicted
            # energy improvement is itself at the representational floor.
            at_floor=abs(alpha*slope)<=energy_roundoff_window
            residual_decrease=trial_metrics['scaled_generalized_residual_norm_N'] < metrics['scaled_generalized_residual_norm_N']*(1-1e-4*alpha)
            if armijo or (at_floor and trial_energy<=energy+energy_roundoff_window and residual_decrease):
                history[-1]['accepted_step_length']=alpha
                history[-1]['accepted_by_roundoff_residual_rule']=bool(not armijo)
                z=trial.detach();accepted=True;break
            alpha*=.5
        if not accepted:
            raise RuntimeError('Positive-J energy/residual line search failed: '+json.dumps(history[-1]))
    raise RuntimeError('Nonlinear equilibrium did not meet declared tolerance: '+json.dumps(history[-1]))


def run(output):
    torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    directory=Path('runs/gaussian-boundary-neural-v01/episodes/t0001')
    names=['reference.npz','truth/recipe.json','truth/boundaries.json']
    case=json.loads((directory/names[1]).read_text());spec=json.loads((directory/names[2]).read_text())
    if case['split']!='train' or case['episode_id']!='t0001' or case['boundary_type']!='fixed_patch' or spec['motion_amplitude_m']!=0:
        raise ValueError('Only unchanged zero-motion fixed-patch TRAIN t0001 supported')
    with np.load(directory/names[0]) as archive:
        points=archive['rest_points_m'];tets=archive['tetrahedra']
    material=case['material'];E=material['youngs_modulus_Pa'];nu=material['poissons_ratio'];rho=material['density_kg_m3']
    started=time.monotonic()
    K,mass,gradients,volumes=assemble_tetrahedra(points,tets,E,nu,rho)
    mapping=FiniteRigidPatchMap(points,spec['fixture_nodes'],spec['load_patches'],spec['load_nodes'])
    Kr=mapping.A.T@K@mapping.A
    factor,eigens=checked_cholesky(Kr,name='Unmodified reduced rest tangent')
    force=torch.zeros(mapping.size,dtype=torch.float64)
    force[mapping.handle_dofs[0][:3]]=tensor([.01,.01,.005])
    load_scales=[.001,.01,.1,.25,.5,.75,1.]
    protocol={'role':'Independent finite-strain static TRAIN t0001 diagnostic; not native PhysX or dynamic predictor',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'linear_dependency_sha256':hashlib.sha256(Path(__file__).with_name('campaign_linear_fem.py').read_bytes()).hexdigest(),
              'inputs_sha256':{str(directory/name):hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names},
              'node_count':len(points),'tetrahedra':len(tets),'reduced_dofs':mapping.size,
              'material':material,'constitutive_law':'W=mu/2*(tr(F.T F)-3)-mu*log J+lambda/2*(log J)^2; J>1e-8',
              'fixture':'Exact fixed reference positions; no penalty',
              'handles':'Both full 9-node patches: x_i=X_origin+t+exp(skew(omega))*r_i; exact finite rigid distances',
              'load':'Constant dead force (.01,.01,.005)N at first handle origin; no applied torque; handle1 unforced',
              'load_scales':load_scales,'residual_relative_tolerance':1e-7,'residual_absolute_tolerance_N':1e-11,
              'residual_definition':'All free-node and handle translation gradients in N; physical WORLD torque J_left(omega)^(-T)*axis-angle-gradient divided by maximum rest bounding-box extent. Relative to applied force norm. Axis-angle gradients also separately reported.',
              'solver':'Rest-linear-Kr Cholesky preconditioned nonlinear descent; explicit positive-J energy/residual line search; no stiffness repair',
              'damping_and_inertia':'Not present in static equilibrium; no static mass/damping effect is inferred.',
              'mass_hardware':'Same geometric rigid patches; authored masses/inertias have no gravitational/static inertial force in this diagnostic.',
              'no_native_constitutive_equivalence_claim':True,'no_continuum_convergence_claim':True,
              'local_equilibrium_only':'Continuation from rest; no global-minimum or branch-uniqueness proof.',
              'cpu_threads':2}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    rows=[];initial=None
    try:
        for scale in load_scales:
            result=solve_static(mapping,tets,gradients,volumes,E,nu,factor,force*scale,initial=initial)
            initial=result['z']
            positions=result['positions'];displacement=positions-tensor(points)
            linear_z=torch.cholesky_solve((force*scale)[:,None],factor)[:,0]
            linear_u=(mapping.A@linear_z).reshape(-1,3)
            difference=displacement-linear_u
            physical_gradient=torch.einsum('eic,eid->ecd',displacement[torch.as_tensor(tets,dtype=torch.long)],gradients)
            F=physical_gradient+torch.eye(3,dtype=torch.float64)
            green=.5*(F.transpose(-1,-2)@F-torch.eye(3,dtype=torch.float64))
            patch_errors=[]
            for patch in spec['load_patches']:
                patch_errors.append(float((torch.cdist(positions[patch],positions[patch])-torch.cdist(tensor(points)[patch],tensor(points)[patch])).abs().max()))
            rms=float(displacement.square().mean().sqrt());linear_rms=float(linear_u.square().mean().sqrt())
            row={'load_scale':scale,'converged':result['converged'],'iterations':len(result['history'])-1,
                 'coordinate_rms_displacement_m':rms,'linear_coordinate_rms_displacement_m':linear_rms,
                 'coordinate_rmse_vs_linear_m':float(difference.square().mean().sqrt()),
                 'relative_coordinate_rmse_vs_linear':float(difference.square().mean().sqrt())/max(linear_rms,1e-30),
                 'maximum_nodal_displacement_m':float(torch.linalg.vector_norm(displacement,dim=-1).max()),
                 'fixture_max_error_m':float(displacement[spec['fixture_nodes']].abs().max()),
                 'maximum_rigid_patch_pairwise_distance_error_m':max(patch_errors),
                 'maximum_absolute_green_principal_strain':float(torch.linalg.eigvalsh(green).abs().max()),
                 'minimum_J':result['minimum_J'],'elastic_energy_J':result['elastic_energy_J'],
                 'potential_energy_J':result['potential_energy_J'],'residual_tolerance_N':result['residual_tolerance_N'],
                 'invalid_J_trials_rejected':result['invalid_J_trials_rejected'],
                 'energy_roundoff_window_J':result['energy_roundoff_window_J'],**result['metrics']}
            rows.append(row)
            np.savez_compressed(output/f'scale-{scale:g}.npz',positions_m=positions.numpy(),displacements_m=displacement.numpy(),
                                generalized_coordinates=result['z'].numpy(),linear_displacements_m=linear_u.numpy())
            (output/f'scale-{scale:g}-iterations.json').write_text(json.dumps(result['history'],indent=2)+'\n')
            print(json.dumps(row),flush=True)
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error':str(error),'completed_scales':rows,'no_stiffness_repair':True},indent=2)+'\n')
        raise
    # This additional diagnostic verifies positive curvature at the final
    # equilibrium in the chosen finite-rotation chart, not global uniqueness.
    def final_potential(value):
        return neo_hookean_energy(mapping.positions(value),mapping.points,tets,gradients,volumes,E,nu)[0]-force@value
    try:
        final_tangent=torch.autograd.functional.hessian(final_potential,initial,vectorize=True)
        _,final_eigens=checked_cholesky(final_tangent,name='Final finite-strain equilibrium tangent')
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error':str(error),'completed_scales':rows,'stage':'final-tangent','no_stiffness_repair':True},indent=2)+'\n')
        raise
    result={'protocol':protocol,'reduced_rest_stiffness':eigens,'final_nonlinear_equilibrium_tangent':final_eigens,'continuation':rows,
            'all_declared_residual_gates_passed':all(row['converged'] for row in rows),
            'elapsed_s':time.monotonic()-started,
            'qualification':'Finite-strain static fixed-discretization continuum diagnostic only. No native/continuum accuracy, dynamics, posterior calibration or active-probing success claim.'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.output)
