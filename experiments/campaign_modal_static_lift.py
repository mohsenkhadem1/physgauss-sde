"""Prospective geometry-only static-lift completion of the frozen96 modes.

Up to3 canonical translation shapes complete uniform fixture translation.
No material-dependent static solve, stiffness repair, response-based dimension
selection, finite simulation, or evaluation64 trajectory access occurs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
try:
    from campaign_modal_hypothesis import FixedModalLinearDynamics
except ImportError:
    from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics


def translation_shapes(full):
    p=full.model;Q=torch.zeros((full.size,3),dtype=torch.float64)
    Q[:3*len(p.free_nodes)]=torch.eye(3,dtype=torch.float64).repeat(len(p.free_nodes),1)
    for ids in full.handle_dofs:Q[ids[:3]]=torch.eye(3,dtype=torch.float64)
    return Q


def orthogonal_completion(V,Q,M):
    """Two-pass M-orthogonalization; reject dependencies, never regularize."""
    columns=[v.clone() for v in V.T];records=[];epsilon=torch.finfo(torch.float64).eps
    for index,q in enumerate(Q.T):
        initial=float(q@M@q);r=q.clone()
        for _ in range(2):
            basis=torch.stack(columns,1);r-=basis@(basis.T@M@r)
        norm2=float(r@M@r);threshold=64*epsilon*initial
        if norm2 < -threshold:raise ValueError('Negative physical residual norm; no repair')
        added=norm2>threshold
        if added:columns.append(r/np.sqrt(norm2))
        records.append(dict(canonical_axis=index,initial_metric_norm_squared=initial,
            residual_metric_norm_squared=norm2,roundoff_dependency_threshold=threshold,added=added))
    return torch.stack(columns,1),records


def complete_static_lift(plain):
    """Return a NEW modal object; the original96-mode object stays unchanged."""
    result=copy.copy(plain);full=plain.full;p=full.model
    Q=translation_shapes(full);M0=625.*full.Mbody/p.density+full.Maux
    V,records=orthogonal_completion(plain.basis,Q,M0);result.basis=V;result.mode_count=V.shape[1]
    A=full.A;result.Mb=V.T@(full.Mbody/p.density)@V;result.Mh=V.T@full.Maux@V
    result.Ks=V.T@A.T@plain.Ks_full@A@V;result.Kv=V.T@A.T@plain.Kv_full@A@V
    result.AV=A@V;result.JAV=full.JA@V
    mass=p.body_mass.repeat_interleave(3)/p.density
    result.M_lift=V.T@A.T@(mass[:,None]*plain.S)
    result.Ks_lift=V.T@A.T@plain.Ks_full@plain.S;result.Kv_lift=V.T@A.T@plain.Kv_full@plain.S
    G=torch.zeros((full.size,3*len(full.handle_dofs)),dtype=torch.float64)
    for j,ids in enumerate(full.handle_dofs):G[ids[:3],3*j:3*j+3]=torch.eye(3,dtype=torch.float64)
    result.G=V.T@G
    displacement=A@Q+plain.S;uniform=torch.eye(3,dtype=torch.float64).repeat(len(p.points),1)
    if float((displacement-uniform).abs().max())>1e-12:raise ValueError('Canonical translation does not complete nodal lift')
    aux=full.JA@Q;expected=torch.eye(3,dtype=torch.float64).repeat(len(full.auxiliary_reference),1)
    if aux.shape!=expected.shape or (aux.numel() and float((aux-expected).abs().max())>1e-12):raise ValueError('Complete hardware mass translation lost')
    projected=V@(V.T@M0@Q);residual=float(torch.linalg.matrix_norm(Q-projected)/torch.linalg.matrix_norm(Q)) if Q.numel() else 0.
    orth=float((V.T@M0@V-torch.eye(V.shape[1],dtype=torch.float64)).abs().max())
    if orth>1e-10 or residual>1e-10:raise ValueError('Completed static-lift span/orthogonality failed')
    result.basis_diagnostics={**plain.basis_diagnostics,'vibration_modes_preserved':plain.mode_count,
        'completed_modes':result.mode_count,'canonical_translation_shapes':3,'completion_records':records,
        'completed_M0_orthonormal_max_error':orth,'canonical_translation_relative_span_error':residual,
        'nodal_uniform_translation_identity_max_error_m_per_m':float((displacement-uniform).abs().max()),
        'parameter_independent_completion':True,'original_plain_basis_unchanged':True}
    return result


def main(manifest_path,linear_root,output):
    try:
        from campaign_hypothesis_inputs import known_hardware_from_public
        from campaign_linear_hypothesis import LinearHypothesisDynamics
        from campaign_exciting_commands import make_exciting_commands
        from campaign_hypothesis_dynamics import runtime_sources
    except ImportError:
        from experiments.campaign_hypothesis_inputs import known_hardware_from_public
        from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
        from experiments.campaign_exciting_commands import make_exciting_commands
        from experiments.campaign_hypothesis_dynamics import runtime_sources
    torch.set_num_threads(2);started=time.monotonic();destination=Path(output);destination.mkdir(parents=True,exist_ok=False)
    design=json.loads(Path(manifest_path).read_text());hyp={h['id']:h for h in design['hypotheses']};public={h['id']:h for h in design['public_hardware']}
    fit=[c for c in design['cells'] if c['evaluation_role']=='fit'];assert len(fit)==32
    bases={};rows=[]
    for cell in fit:
        record=hyp[cell['hypothesis_id']];spec=record['spec'];key=record['support_draw_id']
        if key not in bases:
            full=LinearHypothesisDynamics.from_hypothesis(known_hardware_from_public(public[cell['public_hardware_id']]),spec)
            bases[key]=complete_static_lift(FixedModalLinearDynamics(full,mode_count=96))
        modal=bases[key];result=modal.predict(make_exciting_commands(cell['program_id']),spec['material'])
        reference=Path(linear_root)/'cells'/cell['id']/'predictions.npz'
        with np.load(reference,allow_pickle=False) as truth:
            q=truth['positions_m'];v=truth['velocities_point_chord_m_s'];error=float(np.sqrt(np.mean((result['positions_m'].numpy()-q)**2)))
            signal=float(np.sqrt(np.mean((q-modal.full.points.numpy())**2)));v_error=float(np.sqrt(np.mean((result['velocities_point_chord_m_s'].numpy()-v)**2)))
        np.savez_compressed(destination/(cell['id']+'.npz'),**{k:a.numpy() if isinstance(a,torch.Tensor) else a for k,a in result.items()})
        rows.append(dict(id=cell['id'],boundary_kind=spec['boundary_kind'],material=record['material_tuple_id'],
            coordinate_RMSE_vs_full_linear_m=error,full_linear_motion_RMS_m=signal,relative_coordinate_RMSE=error/max(signal,1e-30),
            velocity_coordinate_RMSE_m_s=v_error,full_linear_sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
        print(json.dumps(rows[-1]),flush=True)
    report=dict(status='FIXED96_PLUS_CANONICAL_STATIC_LIFT_FIT32_ONLY',fit_cases=32,basis_count=len(bases),
        vibration_modes=96,maximum_additional_modes=3,basis_diagnostics={k:b.basis_diagnostics for k,b in bases.items()},rows=rows,
        elapsed_s=time.monotonic()-started,evaluation64_accessed=False,response_based_basis_or_dimension_selection=False,
        finite_truth_used=False,posterior_launched=False,nonlinear_reference=False,production_qualified=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plain_modal_source_sha256=hashlib.sha256(Path(__file__).with_name('campaign_modal_hypothesis.py').read_bytes()).hexdigest(),
        frozen_physical_dependencies_sha256=runtime_sources(),input_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        limitation='Known complete hardware/support geometry and fixed analytic support law; linear small-strain approximation with explicit material coefficients. No finite,64evaluation or calibration evidence.')
    (destination/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--linear-root',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();main(a.manifest,a.linear_root,a.output)
