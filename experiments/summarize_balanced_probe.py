"""Posthoc elastic-excitation comparison on previously seen CALIBRATION15.

All four capture arrays stay closed until both balanced workers AND their
controller are terminal, the exact two declared cells are present, and source/
protocol/artifact hashes pass. No reserved values, physics calls or inference.
Applied handle wrench is reconstructed from actual step endpoint origins; it
does not include fixture reactions, drag, inertia or internal stresses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGN = dict(path='runs/operator-dev/balanced-probe-v01/design.json',
    sha256='a8336a311eec8ba862a2a6004bd0716e61ebbe440a27ae430b69d105a005146f')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def resolve(path):
    p = Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != str(path):
        raise ValueError('Canonical repository-relative path required')
    result = (ROOT/p).resolve()
    if not result.is_relative_to(ROOT): raise ValueError('Repository escape')
    return result


def checked(record):
    path = resolve(record['path'])
    if sha(path) != record['sha256']: raise ValueError('Pinned artifact changed: '+record['path'])
    return path


def record(path):
    return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))


def rigid_residual(rest,positions):
    """Uniform-node proper Kabsch fit independently at each complete frame."""
    x,y = np.asarray(rest,dtype=np.float64),np.asarray(positions,dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or y.ndim != 3 or y.shape[1:] != x.shape or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError('Finite rest[N,3] and positions[T,N,3] required')
    centered = x-x.mean(0)
    target = y-y.mean(1,keepdims=True)
    u,_,vh = np.linalg.svd(np.einsum('ni,tnj->tij',centered,target))
    correction = np.broadcast_to(np.eye(3),(len(y),3,3)).copy()
    correction[:,2,2] = np.linalg.det(vh.transpose(0,2,1)@u.transpose(0,2,1))
    rotation = vh.transpose(0,2,1)@correction@u.transpose(0,2,1)
    fitted = centered[None]@rotation.transpose(0,2,1)+y.mean(1,keepdims=True)
    residual = y-fitted
    motion = y-x
    total = float(np.sqrt(np.mean(motion**2)))
    nonrigid = float(np.sqrt(np.mean(residual**2)))
    angle = np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))
    return dict(uniform_node_coordinate_RMS_total_motion_m=total,
        uniform_node_coordinate_RMS_after_best_fit_rigid_m=nonrigid,
        residual_to_total_RMS_ratio=nonrigid/total if total else 0.,
        residual_fraction_of_total_motion_MSE=(nonrigid/total)**2 if total else 0.,
        maximum_frame_total_coordinate_RMS_m=float(np.sqrt(np.mean(motion**2,axis=(1,2))).max()),
        maximum_frame_nonrigid_coordinate_RMS_m=float(np.sqrt(np.mean(residual**2,axis=(1,2))).max()),
        maximum_fitted_rotation_angle_deg=float(np.rad2deg(angle).max()),
        definition='Uniform over all mesh nodes/XYZ/complete frames; proper rigid transform fitted separately per frame')


def strain_summary(rest,tetrahedra,positions):
    """Compatible F/J/physical covariance from the SAME supplied nodal field."""
    import torch
    try:
        from campaign_material_gaussians import MaterialGaussians
    except ImportError:
        from experiments.campaign_material_gaussians import MaterialGaussians
    reference = MaterialGaussians.from_tetrahedra(torch.as_tensor(rest,dtype=torch.float64),
        torch.as_tensor(tetrahedra,dtype=torch.int64),density=1.)
    with torch.no_grad():
        state = reference.transport(torch.as_tensor(positions,dtype=torch.float64))
        stretch = torch.linalg.svdvals(state.deformation_gradients)
        strain = stretch-1
        weights = reference.volume_weights
        volume_RMS = torch.sqrt((strain.square().mean(-1)*weights).sum(-1).mean())
        covariance_eigenvalues = torch.linalg.eigvalsh(state.covariances)
    return dict(minimum_J=float(state.jacobians.min()),maximum_J=float(state.jacobians.max()),
        maximum_absolute_J_minus_one=float((state.jacobians-1).abs().max()),
        minimum_principal_stretch=float(stretch.min()),maximum_principal_stretch=float(stretch.max()),
        maximum_absolute_principal_stretch_minus_one=float(strain.abs().max()),
        reference_volume_weighted_principal_strain_RMS=float(volume_RMS),
        minimum_physical_covariance_eigenvalue_m2=float(covariance_eigenvalues.min()),
        definition='Tet-wise F from shared vertices; reference-volume weighted RMS over frames/tets/three stretches; density1 is geometric bookkeeping only')


def distance_summary(rest,tetrahedra,positions,rest_origins,handle_centers):
    pairs = set()
    for tet in np.asarray(tetrahedra):
        for i in range(4):
            for j in range(i): pairs.add(tuple(sorted((int(tet[i]),int(tet[j])))))
    edges = np.asarray(sorted(pairs),dtype=np.int64)
    base = np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=-1)
    current = np.linalg.norm(positions[:,edges[:,1]]-positions[:,edges[:,0]],axis=-1)
    if (base<=0).any(): raise ValueError('Nonzero reference edges required')
    relative = current/base-1
    reference_distance = float(np.linalg.norm(rest_origins[1]-rest_origins[0]))
    distance = np.linalg.norm(handle_centers[:,1]-handle_centers[:,0],axis=-1)
    return dict(rest_handle_separation_m=reference_distance,
        minimum_handle_separation_m=float(distance.min()),maximum_handle_separation_m=float(distance.max()),
        maximum_absolute_handle_separation_change_m=float(np.abs(distance-reference_distance).max()),
        maximum_absolute_relative_handle_separation_change=float(np.abs(distance/reference_distance-1).max()),
        unique_mesh_edges=len(edges),minimum_edge_length_ratio=float((relative+1).min()),
        maximum_edge_length_ratio=float((relative+1).max()),
        uniform_edge_relative_length_change_RMS=float(np.sqrt(np.mean(relative**2))))


def wrench_summary(rows,expected_force,rest_origins):
    """Independently reconstruct actual applied wrench at BOTH step endpoints."""
    f = np.asarray([r['held_handle_forces_world_N'] for r in rows],dtype=np.float64)
    starts = np.asarray([r['handle_centers_start_m'] for r in rows],dtype=np.float64)
    ends = np.asarray([r['handle_centers_end_m'] for r in rows],dtype=np.float64)
    intervals = np.asarray([r['actual_interval_s'] for r in rows],dtype=np.float64)
    if f.shape != (360,2,3) or not np.array_equal(f,expected_force) or starts.shape != f.shape or ends.shape != f.shape:
        raise ValueError('Complete360 held-command/endpoint rows required')
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or not np.isfinite(intervals).all():
        raise ValueError('Finite actual step geometry/clock required')
    dt = intervals[:,1]-intervals[:,0]
    if (dt<=0).any() or not np.allclose(dt,1/120,rtol=0,atol=2e-14) or not np.array_equal(intervals[1:,0],intervals[:-1,1]):
        raise ValueError('Contiguous positive actual120Hz intervals required')
    endpoint_moment = np.stack((np.cross(starts,f),np.cross(ends,f)),axis=1)
    logged = np.asarray([r['moment_about_world_origin_start_end_Nm'] for r in rows])
    if logged.shape != endpoint_moment.shape or not np.allclose(logged,endpoint_moment,rtol=0,atol=2e-16):
        raise ValueError('Logged endpoint moments disagree with reconstructed geometry')
    force = f.sum(1)
    moment = endpoint_moment.sum(2)
    norm_force = np.linalg.norm(force,axis=-1)
    norm_moment = np.linalg.norm(moment,axis=-1)
    reference_moment = np.cross(rest_origins,f).sum(1)
    return dict(maximum_applied_handle_resultant_N=float(norm_force.max()),
        RMS_applied_handle_resultant_N=float(np.sqrt(np.mean(norm_force**2))),
        maximum_actual_endpoint_handle_moment_Nm=float(norm_moment.max()),
        RMS_actual_endpoint_handle_moment_Nm=float(np.sqrt(np.mean(norm_moment**2))),
        maximum_reference_origin_handle_moment_Nm=float(np.linalg.norm(reference_moment,axis=-1).max()),
        applied_linear_impulse_N_s=np.einsum('s,sc->c',dt,force).tolist(),
        start_endpoint_angular_impulse_quadrature_Nm_s=np.einsum('s,sc->c',dt,moment[:,0]).tolist(),
        end_endpoint_angular_impulse_quadrature_Nm_s=np.einsum('s,sc->c',dt,moment[:,1]).tolist(),
        maximum_per_handle_force_N=float(np.linalg.norm(f,axis=-1).max()),
        actual_interval_count=len(rows),actual_final_time_s=float(intervals[-1,1]),
        logged_endpoint_moments_match_reconstruction=True,
        scope='Applied handle forces only, at actual endpoints; excludes reactions/drag/internal/inertial terms; endpoint quadratures are not exact angular-impulse integrals')


def terminal_capture(protocol_record,expected_output,design):
    protocol = read(checked(protocol_record))
    if (protocol['output']!=expected_output or protocol['public_hardware']!=design['public_hardware']
            or protocol['realized_hypothesis']!=design['realized_hypothesis']
            or protocol['integration_hz']!=120): raise ValueError('Only matched CALIB15/publicH/120Hz captures permitted')
    for name,digest in protocol['source_sha256'].items():
        if design['source_sha256'].get(name)!=digest: raise ValueError('Capture physical source differs from frozen development design')
        checked(dict(path=name,sha256=digest))
    output=resolve(expected_output)
    e,s=read(output/'execution.json'),read(output/'summary.json')
    if e.get('terminal') is not True or Path('/proc',str(e['pid'])).exists(): raise ValueError('Capture still live')
    if e.get('completed') is not True or s.get('completed') is not True or (output/'failure.json').exists():
        raise ValueError('Terminal failed capture preserved; no successful-subset comparison')
    if s['steps']!=360 or s['frames']!=61 or read(output/'protocol.json')!=protocol:
        raise ValueError('Complete61/360 exact protocol capture required')
    for filename,key in [('trajectory.npz','trajectory_sha256'),('steps.jsonl','steps_sha256')]:
        if sha(output/filename)!=s[key]: raise ValueError('Terminal capture artifact changed')
    checked(protocol['known_commands'])
    return dict(protocol=protocol_record,trajectory=record(output/'trajectory.npz'),steps=record(output/'steps.jsonl'),
        summary=record(output/'summary.json'),execution=record(output/'execution.json'))


def run(controller_path,controller_sha256,output):
    import torch
    torch.set_num_threads(2)
    design=read(checked(DESIGN))
    if (design['object_index']!=15 or design['role']!='DEVELOPMENT_EXCITATION_DIAGNOSTIC'
            or design['inference_or_reserved_comparison'] is not False): raise ValueError('Exact development CALIB15 scope required')
    for name,digest in design['source_sha256'].items(): checked(dict(path=name,sha256=digest))
    prepared=read(resolve('runs/operator-dev/balanced-probe-v01/prepared.json'))
    if (prepared['design']!=DESIGN or len(prepared['cells'])!=2
            or prepared['source_sha256']!=design['source_sha256']):
        raise ValueError('Exactly two declared balanced captures and source pins required')
    controller=read(checked(dict(path=controller_path,sha256=controller_sha256)))
    if controller.get('terminal') is not True: raise ValueError('Controller not terminal; do not inspect live capture arrays')
    if (controller.get('completed') is not True or controller.get('retries')!=0
            or controller.get('raw_local') is not True or len(controller.get('records',[]))!=2):
        raise ValueError('Both terminal captures required; failed/queued slots preserved without filtering')
    actual=sorted(controller['records'],key=lambda r:r['output'])
    expected=sorted(prepared['cells'],key=lambda r:r['output'])
    for a,b in zip(actual,expected):
        if (set(a)!={'pid','exit_code','protocol','output'} or a['protocol']!=b['protocol'] or a['output']!=b['output']
                or a['exit_code']!=0 or Path('/proc',str(a['pid'])).exists()):
            raise ValueError('Exact two declared terminal/reaped cells required')
    captures={}
    if sorted(Path(row['output']).name for row in expected)!=['020N','200N']:
        raise ValueError('Both declared020N/200N amplitudes required')
    for row in expected:
        label=Path(row['output']).name
        physical=read(checked(row['protocol']))
        declared=next(c for c in design['clocks'] if c['label']==label)
        if (physical['input_design']!=DESIGN or physical['known_commands']!=declared['commands']
                or physical['public_command_manifest']!=declared['public_command_manifest']
                or physical['evaluation_role']!=design['role']):
            raise ValueError('Balanced command/design/role differs from frozen amplitudes')
        captures['balanced_'+label]=terminal_capture(row['protocol'],row['output'],design)
    calibration=read(checked(design['calibration_freeze']))
    execution=read(checked(calibration['execution_protocol']))
    plan=read(checked(execution['cohort_plan']))
    obj=next(o for o in plan['objects'] if o['index']==15)
    if obj['role']!='CALIBRATION' or obj['realized_hypothesis']!=design['realized_hypothesis']:
        raise ValueError('Only previously seen CALIBRATION15 original controls permitted')
    originals=sorted((c for c in plan['cells'] if c['object_id']==obj['object_id']),key=lambda c:c['id'])
    if [c['id'] for c in originals]!=['point_prior_015__A','point_prior_015__B']:
        raise ValueError('Both original A/B captures required')
    for label,row in zip(('A','B'),originals): captures['original_'+label]=terminal_capture(row['protocol'],row['output'],design)
    destination=resolve(output)
    if destination.exists(): raise ValueError('Fresh posthoc report directory required')
    public=read(checked(design['public_hardware']))
    rest=np.asarray(public['rest_points_m']);tets=np.asarray(public['tetrahedra'])
    origins=np.asarray([h['origin_world_m'] for h in public['handles']])
    sources={**design['source_sha256'],'experiments/summarize_balanced_probe.py':sha(Path(__file__)),
        'experiments/campaign_material_gaussians.py':sha(ROOT/'experiments/campaign_material_gaussians.py')}
    destination.mkdir(parents=True,exist_ok=False)
    declaration=dict(schema='balanced-probe-posthoc-analysis-v0.1',design=DESIGN,
        controller=dict(path=controller_path,sha256=controller_sha256),
        prepared=record(resolve('runs/operator-dev/balanced-probe-v01/prepared.json')),
        source_sha256=sources,captures=captures,role='DEVELOPMENT_EXCITATION_DIAGNOSTIC',
        object_index=15,selection_posthoc=True,reserved_values_opened=False,
        comparator='All original A/B plus both predeclared balanced amplitudes; no filtering',
        inference_or_prediction_improvement_claim=False,raw_artifacts_LOCAL=True)
    write(destination/'analysis-protocol.json',declaration)
    results={}
    try:
        for label,capture in captures.items():
            protocol=read(checked(capture['protocol']))
            with np.load(checked(protocol['known_commands']),allow_pickle=False) as raw:
                force=raw['handle_forces_world_N'].copy()
                if (np.any(raw['handle_torques_world_Nm']) or np.any(raw['body_acceleration_world_m_s2'])
                        or raw['nodal_forces_world_N'].shape!=(360,0,3)):
                    raise ValueError('This wrench diagnostic requires no other applied actuators')
            with np.load(checked(capture['trajectory']),allow_pickle=False) as raw:
                q,centers=raw['positions_m'].copy(),raw['handle_centers_m'].copy()
                intervals=raw['force_interval_actual_times_s'].copy()
                if not np.array_equal(raw['force_step_commands_N'],force) or not np.array_equal(raw['times_s'],np.arange(61)/20):
                    raise ValueError('Actual capture command/output clocks differ')
            if q.shape!=(61,len(rest),3) or centers.shape!=(61,2,3): raise ValueError('Complete finite trajectory shape required')
            keys=('held_handle_forces_world_N','handle_centers_start_m','handle_centers_end_m',
                'actual_interval_s','moment_about_world_origin_start_end_Nm')
            rows=[]
            with checked(capture['steps']).open() as stream:
                for line in stream:
                    full=json.loads(line)
                    rows.append({key:full[key] for key in keys})
            if len(rows)!=360 or not np.array_equal(intervals,np.asarray([r['actual_interval_s'] for r in rows])):
                raise ValueError('NPZ and step ledgers differ')
            results[label]=dict(rigid_motion=rigid_residual(rest,q),material_gaussian_strain=strain_summary(rest,tets,q),
                distances=distance_summary(rest,tets,q,origins,centers),applied_wrench=wrench_summary(rows,force,origins))
        for name,digest in sources.items(): checked(dict(path=name,sha256=digest))
        result=dict(completed=True,role='DEVELOPMENT_EXCITATION_DIAGNOSTIC',object_index=15,
            analysis_protocol=record(destination/'analysis-protocol.json'),cases=results,
            all4_cases_included=True,reserved_values_opened=False,
            interpretation='Posthoc excitation/strain diagnostic on one already seen object, not inference, prediction improvement, population validation or continuum/native equivalence',
            raw_artifacts_LOCAL=True)
        write(destination/'summary.json',result)
        return result
    except BaseException as exc:
        write(destination/'failure.json',dict(completed=False,error=type(exc).__name__,message=str(exc),
            complete_cases_so_far=list(results),all_cases_retained_in_protocol=True,no_subset_success_claim=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',required=True);parser.add_argument('--controller-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.controller,args.controller_sha256,args.output)
    print(json.dumps(dict(completed=result['completed'],cases=list(result['cases']),output=args.output)))
