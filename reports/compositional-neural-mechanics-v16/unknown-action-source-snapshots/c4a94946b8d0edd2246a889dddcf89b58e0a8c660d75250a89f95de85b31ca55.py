"""Privileged CPU small-strain FEM sanity reference for TRAIN t0001 only.

This is NOT a neural decoder, runtime inverse solver, PhysX reproduction or
continuum truth. Exact linear rigid-patch condensation adds no penalty spring.
Density gives lumped tet-node masses; both authored rigid masses/inertias are
added separately. No regularization repairs an underconstrained stiffness.
"""
from __future__ import annotations
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def tensor(x):
    return torch.as_tensor(x,dtype=torch.float64,device='cpu')


def assemble_tetrahedra(points,tets,young,poisson,density):
    if not (young>0 and -.99<poisson<.4999 and density>0):raise ValueError('Invalid elastic material')
    x=tensor(points);ids=torch.as_tensor(tets,dtype=torch.long)
    dm=(x[ids[:,1:]]-x[ids[:,:1]]).transpose(1,2)
    determinant=torch.linalg.det(dm)
    if not bool((determinant>0).all()):raise ValueError('Positive nondegenerate tet orientation required')
    volume=determinant/6
    gradients=torch.linalg.inv(dm)
    gradients=torch.cat([-gradients.sum(1,keepdim=True),gradients],1)
    B=torch.zeros((len(ids),6,12),dtype=torch.float64)
    for j in range(4):
        dx,dy,dz=gradients[:,j].unbind(-1)
        B[:,0,3*j]=dx;B[:,1,3*j+1]=dy;B[:,2,3*j+2]=dz
        B[:,3,3*j+1]=dz;B[:,3,3*j+2]=dy
        B[:,4,3*j]=dz;B[:,4,3*j+2]=dx
        B[:,5,3*j]=dy;B[:,5,3*j+1]=dx
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    D=torch.zeros((6,6),dtype=torch.float64);D[:3,:3]=lam
    D[:3,:3]+=torch.eye(3,dtype=torch.float64)*(2*mu)
    D[3:,3:]=torch.eye(3,dtype=torch.float64)*mu
    local=(B.transpose(1,2)@D@B)*volume[:,None,None]
    dofs=(ids[:,:,None]*3+torch.arange(3)).reshape(len(ids),12)
    size=3*len(x);K=torch.zeros((size,size),dtype=torch.float64)
    index=dofs[:,:,None]*size+dofs[:,None,:]
    K.view(-1).index_add_(0,index.flatten(),local.flatten())
    nodal_mass=torch.zeros(len(x),dtype=torch.float64)
    nodal_mass.index_add_(0,ids.flatten(),(volume[:,None].expand(-1,4)*density/4).flatten())
    return K,nodal_mass.repeat_interleave(3),gradients,volume


def skew(r):
    x,y,z=tensor(r).unbind(-1)
    zero=torch.zeros_like(x)
    return torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)


def condensation(points,fixture_nodes,load_patches,load_nodes):
    """u=A z: fixture u=0; rigid patch u_i=t_j+omega_j cross r_i."""
    points=tensor(points);fixed=set(map(int,fixture_nodes));seen=set()
    if len(load_patches)!=len(load_nodes):raise ValueError('Each rigid patch requires a handle origin')
    for patch,node in zip(load_patches,load_nodes):
        if len(set(patch))!=len(patch) or int(node) not in patch:raise ValueError('Invalid rigid patch')
        if set(patch)&(fixed|seen):raise ValueError('Overlapping fixture/rigid patches are unsupported')
        seen.update(patch)
    if any(i<0 or i>=len(points) for i in fixed|seen):raise ValueError('Constraint node outside mesh')
    free=[i for i in range(len(points)) if i not in fixed|seen]
    count=3*len(free)+6*len(load_patches)
    A=torch.zeros((3*len(points),count),dtype=torch.float64)
    for j,node in enumerate(free):A[3*node:3*node+3,3*j:3*j+3]=torch.eye(3)
    handle_dofs=[]
    for j,(patch,node) in enumerate(zip(load_patches,load_nodes)):
        start=3*len(free)+6*j;handle_dofs.append(list(range(start,start+6)))
        for i in patch:
            A[3*i:3*i+3,start:start+3]=torch.eye(3)
            A[3*i:3*i+3,start+3:start+6]=-skew(points[i]-points[node])
    return A,handle_dofs


def reduced_matrices(K,mass_diagonal,A,handle_dofs,handle_mass,handle_inertias):
    reduced_K=A.T@K@A
    M=A.T@(mass_diagonal[:,None]*A)
    for ids,m,inertia in zip(handle_dofs,handle_mass,handle_inertias):
        if m<=0 or bool((tensor(inertia)<=0).any()):raise ValueError('Positive rigid mass and inertia required')
        ids=torch.tensor(ids);M[ids[:,None],ids[None,:]]+=torch.diag(torch.cat([tensor([m,m,m]),tensor(inertia)]))
    return .5*(reduced_K+reduced_K.T),.5*(M+M.T)


def checked_cholesky(matrix,*,name='stiffness',require_relative_eigenvalue=1e-10):
    eigenvalues=torch.linalg.eigvalsh(matrix)
    cutoff=max(float(eigenvalues.abs().max())*require_relative_eigenvalue,1e-18)
    diagnostics={'minimum_eigenvalue':float(eigenvalues[0]),'maximum_eigenvalue':float(eigenvalues[-1]),
        'smallest_ten_eigenvalues':eigenvalues[:10].tolist(),'eigenvalues_at_or_below_cutoff':int((eigenvalues<=cutoff).sum()),
        'relative_eigenvalue_cutoff':require_relative_eigenvalue,'absolute_cutoff':cutoff}
    if float(eigenvalues[0])<=cutoff:
        raise ValueError(name+' is singular/ill-conditioned; no regularizer applied: '+json.dumps(diagnostics))
    factor,info=torch.linalg.cholesky_ex(matrix)
    if int(info)!=0:raise ValueError(name+' Cholesky failed; no regularizer applied: '+json.dumps(diagnostics))
    diagnostics['cholesky_info']=int(info)
    return factor,diagnostics


def pulse(t,seconds=3.):
    phase=(t/seconds-.1)/.45
    return np.sin(np.pi*phase)**2 if 0<phase<1 else 0.


def midpoint_dynamics(K,M,force,drag,hz,seconds=3.,output_hz=10,
                      initial_displacement=None,initial_velocity=None,damping_matrix=None,
                      step_observer=None):
    """Implicit midpoint, equivalent to linear average-acceleration integration.

    Force is evaluated at each interval MIDPOINT and held over that interval;
    this is not endpoint-load Newmark quadrature. u_next=u+dt*(v+v_next)/2,
    M*(v_next-v)/dt+C*(v_next+v)/2+K*(u_next+u)/2=f_mid,
    C=drag*M unless an explicit damping matrix is supplied. An analytic smooth
    pulse supplies midpoint loads. No PhysX
    elasticity damping, projection, XPBD iteration or attachment compliance is
    imitated. Each timestep matrix is factored once in float64.
    """
    if hz%output_hz or hz<=0 or drag<0:raise ValueError('Integral output stride/positive Hz/nonnegative drag required')
    dt=1/hz;size=len(force)
    u=torch.zeros(size,dtype=torch.float64) if initial_displacement is None else tensor(initial_displacement).clone()
    v=torch.zeros(size,dtype=torch.float64) if initial_velocity is None else tensor(initial_velocity).clone()
    C=drag*M if damping_matrix is None else tensor(damping_matrix)
    effective=M+(dt/2)*C+(dt*dt/4)*K
    factor,info=torch.linalg.cholesky_ex(effective)
    if int(info)!=0:raise ValueError('Effective dynamic matrix failed Cholesky; no repair')
    right_velocity=M-(dt/2)*C-(dt*dt/4)*K
    displacements=[u.clone()];velocities=[v.clone()];times=[0.];impulse=torch.zeros_like(force)
    if step_observer is not None:step_observer(0.,u,v)
    for step in range(1,round(seconds*hz)+1):
        applied=force*pulse((step-.5)*dt,seconds)
        rhs=right_velocity@v-dt*(K@u)+dt*applied
        next_v=torch.cholesky_solve(rhs[:,None],factor)[:,0]
        u=u+(dt/2)*(v+next_v);v=next_v;impulse+=dt*applied
        if step_observer is not None:step_observer(step*dt,u,v)
        if step%(hz//output_hz)==0:
            times.append(step*dt);displacements.append(u.clone());velocities.append(v.clone())
    return torch.stack(displacements),torch.stack(velocities),np.array(times),impulse


def strain_metrics(displacements,tets,gradients):
    u=tensor(displacements)
    if u.ndim==2:u=u[None]
    ids=torch.as_tensor(tets,dtype=torch.long)
    gradient=torch.einsum('teic,eid->tecd',u[:,ids],gradients)
    strain=.5*(gradient+gradient.transpose(-1,-2))
    principal=torch.linalg.eigvalsh(strain)
    F=gradient+torch.eye(3,dtype=torch.float64)
    return {'maximum_absolute_principal_small_strain':float(principal.abs().max()),
        'maximum_small_strain_frobenius':float(torch.linalg.matrix_norm(strain).max()),
        'minimum_geometric_volume_ratio':float(torch.linalg.det(F).min()),
        'small_strain_caution_above_5percent':bool(principal.abs().max()>.05)}


def run(output):
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    directory=Path('runs/gaussian-boundary-neural-v01/episodes/t0001')
    case=json.loads((directory/'truth/recipe.json').read_text())
    if case['split']!='train' or case['episode_id']!='t0001' or case['boundary_type']!='fixed_patch':
        raise ValueError('Only known fixed-patch TRAINt0001 allowed')
    spec=json.loads((directory/'truth/boundaries.json').read_text())
    with np.load(directory/'reference.npz') as z:points=z['rest_points_m'];tets=z['tetrahedra']
    if spec['motion_amplitude_m']!=0:raise ValueError('This reference eliminates zero prescribed fixture DOFs only')
    material=case['material'];peak=tensor([.01,.01,.005]);started=time.monotonic()
    K,mass,gradients,volumes=assemble_tetrahedra(points,tets,material['youngs_modulus_Pa'],material['poissons_ratio'],material['density_kg_m3'])
    A,handle_dofs=condensation(points,spec['fixture_nodes'],spec['load_patches'],spec['load_nodes'])
    inertias=[]
    for patch,node in zip(spec['load_patches'],spec['load_nodes']):
        r=points[patch]-points[node];m=spec['handle_mass_kg_each']
        inertias.append(np.maximum(m*np.mean(np.sum(r**2,axis=1)[:,None]-r**2,axis=0),.4*m*.003**2))
    Kr,M=reduced_matrices(K,mass,A,handle_dofs,[spec['handle_mass_kg_each']]*2,inertias)
    C=material['linear_damping_s_inverse']*(A.T@(mass[:,None]*A))
    force=torch.zeros(len(Kr),dtype=torch.float64);force[handle_dofs[0][:3]]=peak
    protocol={'role':'Privileged TRAINt0001 small-deformation diagnostic ONLY; no neural/runtime model change',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256':{str(directory/name):hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ['reference.npz','truth/recipe.json','truth/boundaries.json']},
        'node_count':len(points),'tetrahedra':len(tets),'reduced_dofs':len(Kr),
        'material':material,'peak_force_N':peak.tolist(),'handle_force_application':'At first rigid handle origin; zero external torque; second handle unforced but fully attached',
        'handle_mass_kg_each':spec['handle_mass_kg_each'],'handle_diagonal_inertia_kg_m2':np.asarray(inertias).tolist(),
        'body_volume_m3':float(volumes.sum()),'density_times_volume_body_mass_kg':float(volumes.sum()*material['density_kg_m3']),
        'assembled_lumped_body_mass_kg':float(mass.sum()/3),'total_body_plus_two_handles_mass_kg':float(mass.sum()/3)+2*spec['handle_mass_kg_each'],
        'condensation':'Exact LINEAR t+omega cross r; no penalty, no fictitious stiffness. Finite rotations are outside approximation.',
        'damping':'C=declared linear_damping_s_inverse*(A.T*M_body_lumped*A); no damping added to authored rigid masses/inertias. PhysX separate elasticity damping is omitted; this continuum viscous C is NOT claimed identical to PhysX damping.',
        'integration':'Implicit midpoint / linear average-acceleration equivalence, midpoint-held force; NOT endpoint Newmark loads',
        'refinement':'240/480/960 then1920/3840 only if needed; finest-pair coordinateRMSE <= max(.1micrometre,1% finest RMSmotion) AND pair-difference contraction<.7',
        'native_mass_limitation':'No installed DeformableBodyView nodal mass/inverse-mass getter found; declared rho-volume mass is not verified engine effective mass',
        'no_other_response_labels_opened':True,'no_hidden_regularization':True,'cpu_threads':2}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(output/'source.py').write_bytes(Path(__file__).read_bytes())
    try:factor,eigens=checked_cholesky(Kr,name='Reduced stiffness')
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error':str(error),'regularizer_applied':False},indent=2)+'\n');raise
    static=torch.cholesky_solve(force[:,None],factor)[:,0]
    static_u=(A@static).reshape(len(points),3)
    static_result={'coordinate_rms_displacement_m':float(static_u.square().mean().sqrt()),
        'maximum_nodal_displacement_m':float(torch.linalg.vector_norm(static_u,dim=-1).max()),
        'relative_force_balance_residual':float(torch.linalg.vector_norm(Kr@static-force)/torch.linalg.vector_norm(force)),
        **strain_metrics(static_u,tets,gradients)}
    rows=[];pairs=[];previous=None;previous_difference=None;converged=False
    for hz in [240,480,960,1920,3840]:
        all_step={'maximum_small_strain_frobenius':0.,'maximum_displacement_gradient_frobenius':0.,'maximum_nodal_displacement_m':0.}
        tet_ids=torch.as_tensor(tets,dtype=torch.long)
        def observe(t,reduced_u,reduced_v):
            nodes=(A@reduced_u).reshape(len(points),3)
            G=torch.einsum('eic,eid->ecd',nodes[tet_ids],gradients)
            strain=.5*(G+G.transpose(-1,-2))
            all_step['maximum_small_strain_frobenius']=max(all_step['maximum_small_strain_frobenius'],float(torch.linalg.matrix_norm(strain).max()))
            all_step['maximum_displacement_gradient_frobenius']=max(all_step['maximum_displacement_gradient_frobenius'],float(torch.linalg.matrix_norm(G).max()))
            all_step['maximum_nodal_displacement_m']=max(all_step['maximum_nodal_displacement_m'],float(torch.linalg.vector_norm(nodes,dim=-1).max()))
        u,v,times,impulse=midpoint_dynamics(Kr,M,force,material['linear_damping_s_inverse'],hz,damping_matrix=C,step_observer=observe)
        nodal=(u@A.T).reshape(len(times),len(points),3);nodal_v=(v@A.T).reshape(len(times),len(points),3)
        displacement_rms=float(nodal.square().mean().sqrt())
        rows.append({'hz':hz,'coordinate_rms_displacement_m':displacement_rms,
            'maximum_nodal_displacement_m':float(torch.linalg.vector_norm(nodal,dim=-1).max()),
            'impulse_handle0_Ns':impulse[handle_dofs[0][:3]].tolist(),
            'analytic_pulse_impulse_Ns':(peak*.675).tolist(),
            'fixture_max_displacement_m':float(nodal[:,spec['fixture_nodes']].abs().max()),
            'all_integration_steps':{**all_step,'strain_norm_bounds_max_absolute_principal_strain':True},
            'principal_strain_and_volume_ratio_evaluation':'31 saved physical times; full-step strain Frobenius bound also reported',
            **strain_metrics(nodal,tets,gradients)})
        np.savez_compressed(output/f'hz{hz}.npz',times_s=times,positions_m=points+nodal.numpy(),velocities_m_s=nodal_v.numpy())
        if previous is not None:
            difference=float((nodal-previous).square().mean().sqrt())
            ratio=None if previous_difference is None else difference/max(previous_difference,1e-30)
            tolerance=max(1e-7,.01*displacement_rms)
            pairs.append({'low_hz':hz//2,'high_hz':hz,'coordinate_rmse_m':difference,
                'maximum_nodal_difference_m':float(torch.linalg.vector_norm(nodal-previous,dim=-1).max()),
                'difference_contraction':ratio,'declared_tolerance_m':tolerance})
            if hz>=960 and difference<=tolerance and ratio is not None and ratio<.7:converged=True
            previous_difference=difference
        previous=nodal
        print(json.dumps({'hz':hz,'rms_displacement_m':displacement_rms,'refinement_converged':converged}),flush=True)
        if converged:break
    np.savez_compressed(output/'static.npz',positions_m=points+static_u.numpy(),displacements_m=static_u.numpy())
    result={'protocol':protocol,'reduced_stiffness_eigenvalues':eigens,'static_peak_force':static_result,
        'dynamics':rows,'refinement_pairs':pairs,'refinement_converged':converged,
        'qualification':'An independently discretized small-strain continuum reference with linear rigid attachments and declared viscous damping. It does not prove PhysX or continuum accuracy; large strains/rotations invalidate this approximation.',
        'elapsed_s':time.monotonic()-started}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
