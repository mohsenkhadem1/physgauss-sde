"""Read-only authored-force engineering audit; never launches physics.

The exact published protocol, resolved manifests, seal and saved TRAIN evidence
are checked. Public NN admission remains false: engineering command clocks are
solver-dependent. Old gates, auditors and failed evidence are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

try:
    import campaign_authored_force_case as runner
    import campaign_authored_commands as compiler
    from campaign_rigid_inertia import auxiliary_mass_points
except ImportError:
    from experiments import campaign_authored_force_case as runner
    from experiments import campaign_authored_commands as compiler
    from experiments.campaign_rigid_inertia import auxiliary_mass_points


PROTOCOL_PATH='configs/gaussian_campaign_actual_force_envelope_v01.json'
PROTOCOL_SHA='dd078bd8542cd99015d009e9806e2e09590ee855e6bdb8bafd1cc483ce3d8070'
FIXTURE_PATH='configs/gaussian_independent_diagnostic_inputs_v01.json'
FIXTURE_SHA='073380f72559c50ccc674d5f69fdff1be7e4d0b30484ad66f0233695a813847f'
COMPILER_SHA='fdbf333b9531ce11700b6be05ab916fbdb7b30c2fb96cec3762369ae05a3c475'
PREREQUISITE_SHA='53b4a43c2c86485180ee6786bc4d38b7e8897c9e3dcc737de85f7e7adfa8444d'
AUDITOR_PATH='experiments/campaign_authored_force_audit.py'


class AuditError(ValueError):pass


def require(condition,message):
    if not condition:raise AuditError(message)


def path(record):return runner.resolve_repo_path(record)
def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def read(file):return json.loads(Path(file).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(AuditError('Nonfinite JSON '+value)))


def finite_tree(value):
    if isinstance(value,dict):
        for child in value.values():finite_tree(child)
    elif isinstance(value,list):
        for child in value:finite_tree(child)
    elif isinstance(value,(int,float)):require(np.isfinite(value),'Nonfinite numeric evidence')


def close(a,b,tolerance,message):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        and np.all(np.abs(a-b)<=tolerance),message)


def exact(a,b,message):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),message)


def archive(file):
    with np.load(file,allow_pickle=False) as raw:result={key:raw[key].copy() for key in raw.files}
    for key,value in result.items():require(np.isfinite(value).all(),'Nonfinite saved array '+key)
    return result


def evidence(record,role):
    return {'path':record,'sha256':sha(path(record)),'role':role}


def child(root,name):return str(PurePosixPath(root)/name)


def compiled_expected(spec,cell):
    zero=cell['response']=='zero_external_force'
    result=compiler.compile_authored_commands(spec,force_family=spec['force_type'] if zero else cell['response'],
        physics_hz=cell['hz'],output_hz=20,information_role='known_actuator_command')
    if zero:
        result['held_handle_forces_world_N'][:]=0.
        metadata={**result['metadata'],'requested_force_family':'zero_external_force','explicit_zero_force_control':True,
            'zero_override':'All external commands exactly zero; authored support/hardware remain.'}
        for key,value in metadata.items():
            if 'impulse' in key or 'first_time_moment' in key:metadata[key]=np.zeros_like(value).tolist() if isinstance(value,list) else 0.
        result['metadata']=metadata
    return result


def verify_snapshots(root,pins):
    directory=path(child(root,'private-provenance'))
    require(read(directory/'pins.json')==pins,'Saved pin map differs from reviewed seal')
    snapshots=read(directory/'snapshots.json')
    require(len(snapshots)==len(pins),'Incomplete indexed provenance snapshots')
    require(len({item['original_path'] for item in snapshots})==len(snapshots),'Duplicate source snapshot')
    for item in snapshots:
        require(item['original_path'] in pins and item['sha256']==pins[item['original_path']],'Snapshot pin mismatch')
        saved=PurePosixPath(item['saved_path'])
        require(len(saved.parts)==1 and saved.name not in ['.','..'],'Unsafe indexed snapshot path')
        require(sha(path(child(root,'private-provenance/'+saved.name)))==item['sha256'],'Saved source/proof snapshot hash mismatch')
    require({item['original_path'] for item in snapshots}==set(pins),'Snapshot coverage differs from seal')


def static_inputs(root,cell,canonical):
    selected=next(case for case in canonical['cases'] if case['episode_id']==cell['case'])
    require(selected['recipe']['split']=='train','Only public TRAIN static input permitted')
    private=path(child(root,'private-provenance'));saved=read(private/'physical-inputs.json')
    require(saved['case']==selected['recipe'] and saved['spec']==selected['boundaries'],'Saved material/BC/static recipe differs from canonical fixture')
    arrays=archive(private/'static-reference.npz')
    require(set(arrays)=={row['key'] for row in selected['reference_static_arrays']},'Canonical static-array keys differ')
    for row in selected['reference_static_arrays']:
        a=arrays[row['key']]
        require(str(a.dtype)==row['dtype'] and list(a.shape)==row['shape'] and sha_bytes(a)==row['sha256'],'Canonical static mesh digest/shape/dtype mismatch')
    rest=arrays['rest_points_m']+np.asarray(cell['translation_m'])
    hardware=[];spec=selected['boundaries']
    for node,patch in zip(spec['load_nodes'],spec['load_patches']):
        local=rest[patch]-rest[node];mass=spec['handle_mass_kg_each']
        diagonal=np.maximum(mass*np.mean(np.sum(local**2,axis=1)[:,None]-local**2,axis=0),.4*mass*.003**2)
        hardware.append({'reference_origin_node':node,'reference_origin_m':rest[node].tolist(),'ordered_patch_nodes':patch,
            'ordered_reference_local_offsets_m':local.tolist(),'mass_kg':mass,'COM_offset_m':[0.,0.,0.],
            'full_COM_inertia_kg_m2':np.diag(diagonal).tolist(),'initial_rotation_matrix':np.eye(3).tolist()})
    require(saved['hardware']==hardware,'Full ordered patch/mass/inertia/COM/orientation differs')
    return selected['recipe'],spec,arrays,rest,hardware


def sha_bytes(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verify_contexts(root,case,spec,arrays,hardware,compiled,translation):
    # Independently rebuild the typed feature contract, without asking the
    # runner to regenerate its own output or opening original-root code files.
    points=arrays['rest_points_m']+np.asarray(translation);T=len(compiled['force_midpoints_s']);N=len(points)
    handles=[runner.context.KnownHandle(origin_world_m=h['reference_origin_m'],ordered_patch_nodes=np.asarray(h['ordered_patch_nodes']),
        mass_kg=h['mass_kg'],com_inertia_body_kg_m2=h['full_COM_inertia_kg_m2'],com_offset_body_m=h['COM_offset_m'],
        initial_rotation_body_to_world=h['initial_rotation_matrix'],origin_node_id=h['reference_origin_node']) for h in hardware]
    known=runner.context.KnownHardware(rest_points_world_m=points,tetrahedra=arrays['tetrahedra'],surface_faces=arrays['surface_faces'],
        handles=handles,coordinate_origin_world_m=points.mean(0),source_sha256={})
    forcing={'held_handle_forces_world_N':compiled['held_handle_forces_world_N'][None],
        'held_handle_torques_world_Nm':np.zeros((1,T,2,3)),'held_nodal_forces_world_N':np.zeros((1,T,N,3)),
        'held_body_acceleration_world_m_s2':np.zeros((1,T,3))}
    public={'feature_groups':{'public_known_hardware':known.features(),'public_known_commands':{
        'command_boundary_times_s':compiled['command_boundary_times_s'],'force_intervals_s':compiled['force_interval_times_s'],
        'held_interval_durations_s':np.diff(compiled['command_boundary_times_s']),'requested_output_times_s':compiled['requested_output_times_s']},
        'public_known_force_programs':forcing},'role':'public_input_only_no_material_or_boundary_labels'}
    material=case['material'];mask=np.zeros((2,N,3),dtype=bool);mask[0,spec['fixture_nodes']]=True
    support=np.zeros((T+1,2,3))
    if case['boundary_type']=='moving_patch':
        # The archived command contract evaluates each authored scalar clock;
        # vectorized libm can differ in its final bit from scalar evaluation.
        support[:,0]=np.stack([float(spec['motion_amplitude_m'])*np.sin(np.pi*np.clip(float(t)/3.,0.,1.))**2
            *np.asarray(spec['motion_direction_world'],dtype=float) for t in compiled['command_boundary_times_s']])
    groups=dict(public['feature_groups']);groups.update({
        'latent_material_hypothesis':{'homogeneous_material_parameters':np.array([material['youngs_modulus_Pa'],material['poissons_ratio'],material['density_kg_m3'],material['linear_damping_s_inverse']]),'constitutive_model_one_hot':np.array([True,False])},
        'latent_boundary_hypothesis':{'support_constraint_axis_masks':mask,'support_origins_relative_m':np.array([points[spec['fixture_nodes']].mean(0)-points.mean(0),np.zeros(3)])},
        'latent_force_hypotheses':{key:np.empty((0,*value.shape[1:])) for key,value in forcing.items()},
        'latent_support_law_hypothesis':{'prescribed_support_endpoint_translations_world_m':support,'prescribed_support_endpoint_rotations_world':np.broadcast_to(np.eye(3),(T+1,2,3,3))}})
    conditioned={'feature_groups':groups,'role':'forward_conditioned_on_explicit_hypotheses','hypothesis_role':'known_parameter_training_or_diagnostic'}
    for role,expected in [('engineering-public',public),('known-parameter-diagnostic',conditioned)]:
        directory=path(child(root,role));metadata=read(directory/'manifest.json');actual=archive(directory/'features.npz')
        require(metadata['admission']==runner.ENGINEERING_ONLY,'Solver clock must remain engineering-only')
        require(metadata['feature_archive_sha256']==sha(directory/'features.npz'),'Context archive hash mismatch')
        wanted={group+'/'+name:value for group,fields in expected['feature_groups'].items() for name,value in fields.items()}
        require(set(actual)==set(wanted),'Context feature groups/schema differ')
        for name,value in wanted.items():
            exact(actual[name],value,'Context value mismatch '+name)
            group,key=name.split('/',1);description=metadata['feature_groups'][group][key]
            require(description==runner.context.descriptor(value),'Context descriptor mismatch')
        require(metadata['role']==expected['role'] and metadata['hypothesis_role']==expected.get('hypothesis_role'),'Context information role mismatch')
        require(set(metadata['feature_groups'])==set(expected['feature_groups']),'Context manifest contains additional feature groups')


def fixture_targets(rest,spec,times):
    times=np.asarray(times);fixed=np.asarray(spec['fixture_nodes'])
    shift=spec['motion_amplitude_m']*np.sin(np.pi*np.clip(times/3.,0,1))[:,None]**2*np.asarray(spec['motion_direction_world'])
    return rest[fixed][None]+shift[:,None,:]


def held_force_work_and_moments(force,start,end,stable_increment):
    """Dead world forces at actual moving origins, no supplied pure torques."""
    force,start,end,stable_increment=[np.asarray(v,dtype=float) for v in [force,start,end,stable_increment]]
    require(all(v.shape==(2,3) and np.isfinite(v).all() for v in [force,start,end,stable_increment]),'Complete finite two-handle force/work evidence required')
    return np.einsum('hc,hc->h',force,stable_increment),np.stack([np.cross(start,force),np.cross(end,force)])


def frame_mechanics(data,rest,tets,spec,hardware,material):
    q=data['positions_m'];v=data['velocities_point_chord_m_s'];R=data['handle_rotations'];centers=data['handle_centers_m']
    require(q.shape==v.shape==(61,len(rest),3),'Full 61-frame q/v shape required')
    require(R.shape==(61,2,3,3) and centers.shape==(61,2,3),'Full rigid pose frames required')
    close(q[0],rest,1e-12,'Initial q differs from rest');close(v[0],np.zeros_like(rest),1e-12,'Initial velocity differs from zero')
    close(R[0],np.broadcast_to(np.eye(3),(2,3,3)),1e-12,'Initial rotation differs from declared rest')
    close(centers[0],rest[spec['load_nodes']],1e-12,'Initial handle origin differs from declared rest')
    close(np.swapaxes(R,-1,-2)@R,np.broadcast_to(np.eye(3),R.shape),1e-12,'Improper saved rotation')
    close(np.linalg.det(R),np.ones((61,2)),1e-12,'Saved rotation determinant differs from one')
    for j,(node,patch) in enumerate(zip(spec['load_nodes'],spec['load_patches'])):
        target=centers[:,j,None]+np.einsum('ij,tkj->tik',rest[patch]-rest[node],R[:,j])
        close(q[:,patch],target,1e-12,'Saved finite rigid patch map differs')
    close(q[:,spec['fixture_nodes']],fixture_targets(rest,spec,data['actual_state_times_s']),1e-12,'Saved fixture law differs')
    dm=np.swapaxes(rest[tets[:,1:]]-rest[tets[:,:1]],1,2);grad=np.linalg.inv(dm)
    grad=np.concatenate([-grad.sum(1,keepdims=True),grad],axis=1)
    F=np.eye(3)+np.einsum('teic,eid->tecd',q[:,tets]-rest[tets],grad);J=np.linalg.det(F)
    require(float(J.min())>=.1,'Saved endpoint J below authored 0.1 guard')
    volume=np.abs(np.linalg.det(dm))/6.;mass=np.zeros(len(rest))
    for column in range(4):np.add.at(mass,tets[:,column],material['density_kg_m3']*volume/4)
    auxiliary=data['auxiliary_positions_m'];va=data['auxiliary_velocities_point_chord_m_s']
    require(auxiliary.shape==va.shape==(61,12,3),'Complete auxiliary rigid q/v required')
    close(va[0],np.zeros((12,3)),1e-12,'Initial auxiliary velocity differs from zero')
    auxiliary_mass=[]
    for j,h in enumerate(hardware):
        offsets,mi=auxiliary_mass_points(h['mass_kg'],h['full_COM_inertia_kg_m2']);auxiliary_mass.extend(mi)
        expected=centers[:,j,None]+np.einsum('ij,tkj->tik',offsets+np.asarray(h['COM_offset_m']),R[:,j])
        close(auxiliary[:,6*j:6*j+6],expected,1e-12,'Auxiliary rigid mass pose differs')
    auxiliary_mass=np.asarray(auxiliary_mass)
    P=np.einsum('n,tnc->tc',mass,v)+np.einsum('n,tnc->tc',auxiliary_mass,va)
    kinetic=.5*(np.einsum('n,tnc,tnc->t',mass,v,v)+np.einsum('n,tnc,tnc->t',auxiliary_mass,va,va))
    mu=material['youngs_modulus_Pa']/(2*(1+material['poissons_ratio']))
    lam=material['youngs_modulus_Pa']*material['poissons_ratio']/((1+material['poissons_ratio'])*(1-2*material['poissons_ratio']))
    logJ=np.log(J);density=.5*mu*(np.sum(F*F,axis=(-1,-2))-3)-mu*logJ+.5*lam*logJ**2
    elastic=np.einsum('e,te->t',volume,density)
    stretches=np.linalg.svd(F,compute_uv=False)
    return {'minimum_saved_J':float(J.min()),'maximum_saved_J':float(J.max()),
        'minimum_saved_J_by_frame':J.min(1).tolist(),
        'minimum_saved_principal_stretch':float(stretches.min()),'maximum_saved_principal_stretch':float(stretches.max()),
        'maximum_saved_displacement_m':float(np.linalg.norm(q-rest,axis=-1).max())},P,kinetic,elastic


def verify_cell(cell,pins,canonical):
    root=cell['output'];report={'id':cell['id'],'cell':cell,'output':root,'passed':False};data=None
    try:
        require(cell['case'] in ['t0000','t0001','t0002','t0003'],'Reserved/unknown case rejected before trajectory read')
        directory=path(root)
        if not directory.exists():report['state']='missing';return report,None
        require(not any(item.is_symlink() for item in directory.rglob('*')),'Saved capture symlinks are not canonical evidence')
        if (directory/'failure.json').exists():
            report.update(state='failed_preserved',failure_sha256=sha(directory/'failure.json'));return report,None
        if not (directory/'execution.json').exists():report['state']='incomplete';return report,None
        execution=read(directory/'execution.json')
        if not execution.get('terminal') or not execution.get('completed'):
            report.update(state='incomplete',completed_steps=execution.get('completed_steps'));return report,None
        protocol=read(directory/'protocol.json');finite_tree(protocol)
        require(protocol['cell']==cell and protocol['source_input_sha256']==pins,'Saved cell/source pins differ from seal')
        require((protocol['seconds'],protocol['output_hz'],protocol['frames'])==(3.,20,61),'Saved physical/output clocks changed')
        require(protocol['admission']==runner.ENGINEERING_ONLY and protocol['native_equivalence'] is False
            and protocol['continuum_accuracy_established'] is False and protocol['native_elasticity_damping_reproduced'] is False,'Saved qualification/model semantics changed')
        verify_snapshots(root,pins)
        case,spec,arrays,rest,hardware=static_inputs(root,cell,canonical);expected=compiled_expected(spec,cell)
        compiled=archive(directory/'compiled-commands.npz')
        wanted={key:value for key,value in expected.items() if isinstance(value,np.ndarray)}
        require(set(compiled)==set(wanted),'Compiled command array schema changed')
        for key,value in wanted.items():exact(compiled[key],value,'Compiled authored command/clock differs '+key)
        metadata=protocol['compiled_force_metadata']
        require(metadata==expected['metadata'],'Compiled source/role/impulse metadata differs')
        if cell['response']=='zero_external_force':
            require(metadata['explicit_zero_force_control'] is True and metadata['requested_force_family']=='zero_external_force','Explicit zero command role missing')
            require(np.all(compiled['held_handle_forces_world_N']==0),'Zero control has a hidden force')
        verify_contexts(root,case,spec,arrays,hardware,expected,cell['translation_m'])
        summary=read(directory/'summary.json');finite_tree(summary)
        require(summary['completed'] is True and summary['frames']==61 and summary['cell_id']==cell['id'],'Incomplete summary')
        require(summary['admission']==runner.ENGINEERING_ONLY and all(summary[key] is False for key in
            ['actual_force_envelope_qualified','native_reference_qualified','continuum_accuracy_established','prediction_qualified']),'Summary overstates qualification')
        require(sha(directory/'trajectory.npz')==summary['trajectory_sha256'] and sha(directory/'steps.jsonl')==summary['steps_sha256'],'Trajectory/step digest mismatch')
        data=archive(directory/'trajectory.npz');hz=cell['hz'];n=3*hz;F=compiled['held_handle_forces_world_N']
        require(set(data)==set(REPLAY_KEYS),'Canonical trajectory array schema changed')
        require(all(value.dtype==np.dtype('float64') for value in data.values()),'All canonical trajectory arrays must retain float64 precision')
        require(execution['completed_steps']==execution['planned_steps']==n,'Incomplete physical steps')
        require(execution['cpu_threads']==2 and execution['gpu_used'] is False,'Unqualified execution backend')
        exact(data['times_s'],compiled['requested_output_times_s'],'Nominal 61-frame clocks differ')
        close(data['actual_state_times_s'],data['times_s'],1e-12,'Actual sampled solver clocks differ')
        exact(data['force_midpoints_compiled_s'],compiled['force_midpoints_s'],'Compiled force midpoint differs')
        exact(data['force_step_commands_N'],F,'Saved held commands differ from source')
        intervals=data['force_interval_actual_times_s'];require(intervals.shape==(n,2),'Full measured clock ledger missing')
        close(intervals,compiled['force_interval_times_s'],1e-12,'Actual state interval differs from compiler grid')
        exact(intervals[1:,0],intervals[:-1,1],'Actual state clocks are discontinuous')
        require(np.all(intervals[:,1]>intervals[:,0]),'Nonpositive actual state interval')
        geometry,P,kinetic,elastic=frame_mechanics(data,rest,arrays['tetrahedra'],spec,hardware,case['material'])
        rows=[json.loads(line,parse_constant=lambda value:(_ for _ in ()).throw(AuditError('Nonfinite step JSON'))) for line in (directory/'steps.jsonl').read_text().splitlines()]
        require(len(rows)==n,'Incomplete every-step ledger')
        residuals=[];thresholds=[];Js=[];all_work=[];all_moments=[];last_centers=None;max_work_error=0.;max_support_error=0.;max_energy_identity=0.;max_momentum_error=0.
        last_invariants=None
        for i,row in enumerate(rows):
            finite_tree(row);require(row['step_index']==i and row['converged'] is True,'Unaccepted or reordered step')
            exact(np.asarray(row['compiled_interval_s']),compiled['force_interval_times_s'][i],'Step compiled interval mismatch')
            require(row['compiled_midpoint_s']==compiled['force_midpoints_s'][i] and row['requested_dt_s']==row['step_s']==1/hz,'Step request clock differs')
            exact(np.asarray(row['actual_interval_s']),intervals[i],'Measured ledger intervals differ')
            exact(np.asarray([row['start_time_s'],row['end_time_s']]),intervals[i],'Core state clocks differ')
            require(row['actual_state_start_s']==intervals[i,0] and row['actual_state_midpoint_s']==intervals[i].mean(),'Actual state clock metadata differs')
            exact(np.asarray(row['held_handle_forces_world_N']),F[i],'Requested force differs')
            exact(np.asarray(row['held_handle_forces_N']),F[i],'Actual core held force differs')
            if cell['response']=='twist_couple':require(np.all(F[i].sum(0)==0),'Couple lost exact zero resultant')
            threshold=max(cell['absolute_tolerance_N'],cell['relative_tolerance']*float(np.linalg.norm(F[i])))
            close(row['residual_tolerance_N'],threshold,max(1e-25,threshold*1e-12),'Effective residual threshold differs')
            history=row['nonlinear_history'];require(1<=len(history)<=cell['maximum_iterations']+1,'Nonlinear history missing or capped')
            for j,entry in enumerate(history):
                require(entry['iteration']==j and entry['minimum_J']>0 and entry['scaled_generalized_residual_norm_N']>=0,'Invalid nonlinear history')
            residual=history[-1]['scaled_generalized_residual_norm_N'];require(residual<=threshold*(1+1e-10),'Accepted physical residual exceeds threshold')
            g=row['geometry'];require(g['minimum_volume_ratio']>=.1 and row['minimum_J']>=.1,'Accepted endpoint J below 0.1')
            require(max(g['fixture_max_error_m'],g['maximum_rigid_pose_map_error_m'],g['maximum_rotation_SO3_error'])<=1e-12,'Accepted exact constraint/rotation failure')
            start=np.asarray(row['handle_centers_start_m']);end=np.asarray(row['handle_centers_end_m']);increment=np.asarray(row['handle_center_step_increment_m'])
            require(start.shape==end.shape==increment.shape==(2,3),'Full step handle-center evidence required')
            close(increment,end-start,1e-12,'Stable increments disagree with actual moving centers')
            exact(start,data['handle_centers_m'][0] if i==0 else last_centers,'Handle centers discontinuous')
            work,moment=held_force_work_and_moments(F[i],start,end,increment)
            close(row['held_handle_work_each_J'],work,1e-15,'Independent per-handle force work differs')
            close(row['held_handle_work_J'],work.sum(),1e-15,'Core held force work differs')
            close(row['moment_about_world_origin_start_end_Nm'],moment,1e-15,'Actual moving-origin moment differs')
            close(data['handle_work_each_J'][i],work,1e-15,'NPZ per-handle work differs');close(data['moment_about_world_origin_start_end_Nm'][i],moment,1e-15,'NPZ actual-center moments differ')
            max_work_error=max(max_work_error,float(np.max(abs(np.asarray(row['held_handle_work_each_J'])-work))));last_centers=end
            reaction=np.asarray(row['fixture_reactions_inertia_elastic_drag_minus_external_N']);require(reaction.shape==(len(spec['fixture_nodes']),3),'Fixture reaction shape differs')
            target=fixture_targets(rest,spec,intervals[i]);support=float(np.sum(reaction*(target[1]-target[0])))
            close(row['support_work_J'],support,1e-12,'Independent moving-fixture work differs');max_support_error=max(max_support_error,abs(row['support_work_J']-support))
            require(row['held_nodal_work_J']==0.,'Undeclared nodal force work')
            old,new=row['old_invariants'],row['new_invariants']
            if last_invariants is not None:require(old==last_invariants,'Step invariant ledger is discontinuous')
            for inv in [old,new]:close(inv['energy_J'],inv['chord_kinetic_energy_J']+inv['elastic_energy_J'],1e-12,'Physical total energy differs from components')
            defect=new['energy_J']-old['energy_J']-(row['held_handle_work_J']+support-row['body_viscous_dissipation_J'])
            identity=defect-(row['constraint_chord_work_J']-row['implicit_velocity_increment_loss_J']-row['backward_elastic_gradient_defect_J'])
            close(identity,row['energy_algebra_identity_residual_J'],1e-12,'Energy bookkeeping identity differs')
            if 'energy_balance_time_discretization_defect_J' in row:close(defect,row['energy_balance_time_discretization_defect_J'],1e-12,'Energy time discretization defect differs')
            momentum=np.asarray(new['point_linear_momentum_Ns'])-np.asarray(old['point_linear_momentum_Ns'])-(intervals[i,1]-intervals[i,0])*(F[i].sum(0)+reaction.sum(0))+np.asarray(row['body_drag_impulse_Ns'])
            close(momentum,row['linear_momentum_balance_error_Ns'],1e-12,'Momentum bookkeeping differs')
            max_energy_identity=max(max_energy_identity,abs(identity));max_momentum_error=max(max_momentum_error,float(np.linalg.norm(momentum)))
            if (i+1)%(hz//20)==0:
                frame=(i+1)//(hz//20);exact(end,data['handle_centers_m'][frame],'Step center/frame mismatch')
                close([row['minimum_J'],g['minimum_volume_ratio']],np.repeat(geometry['minimum_saved_J_by_frame'][frame],2),1e-10,'Runtime/frame independent minimum J differs')
                close(new['point_linear_momentum_Ns'],P[frame],1e-11,'Frame physical mass/momentum reconstruction differs')
                close(new['chord_kinetic_energy_J'],kinetic[frame],1e-11+1e-9*abs(kinetic[frame]),'Frame physical kinetic energy differs')
                close(new['elastic_energy_J'],elastic[frame],1e-11+1e-9*abs(elastic[frame]),'Independent saved-frame log-J energy differs')
            residuals.append(residual);thresholds.append(threshold);Js.append(g['minimum_volume_ratio']);all_work.append(work);all_moments.append(moment)
            last_invariants=new
        dt=intervals[:,1]-intervals[:,0]
        for key,value in [('actual_clock_impulse_Ns',np.einsum('t,thc->hc',dt,F)),
            ('actual_clock_absolute_impulse_Ns',np.einsum('t,thc->hc',dt,abs(F))),
            ('actual_clock_first_time_moment_Ns2',np.einsum('t,thc->hc',dt*intervals.mean(1),F))]:close(summary[key],value,1e-12,'Actual clock impulse/moment differs '+key)
        close(summary['handle_work_each_total_J'],np.sum(all_work,axis=0),1e-12,'Total handle work differs')
        roles=[('execution.json','verified_terminal_execution'),('protocol.json','verified_cell_protocol'),('summary.json','verified_cell_summary'),('trajectory.npz','verified_full_61frame_response'),
            ('steps.jsonl','verified_every_step_ledger'),('compiled-commands.npz','verified_authored_command_grid'),('private-provenance/snapshots.json','verified_source_snapshot_index'),
            ('private-provenance/pins.json','verified_saved_source_pin_map'),('private-provenance/static-reference.npz','verified_canonical_static_mesh'),('private-provenance/physical-inputs.json','verified_privileged_static_inputs'),
            ('engineering-public/features.npz','verified_engineering_context_not_production'),('engineering-public/manifest.json','verified_engineering_context_manifest'),
            ('known-parameter-diagnostic/features.npz','verified_privileged_context'),('known-parameter-diagnostic/manifest.json','verified_privileged_context_manifest')]
        roles += [('private-provenance/'+item['saved_path'],'verified_source_snapshot') for item in read(path(child(root,'private-provenance/snapshots.json')))]
        report.update(state='verified',passed=True,source_evidence_records=[evidence(child(root,name),role) for name,role in roles],geometry=geometry,
            minimum_runtime_J=min(Js),maximum_accepted_residual_N=max(residuals),effective_threshold_range_N=[min(thresholds),max(thresholds)],
            absolute_floor_dominated_steps=int(np.sum(np.asarray(thresholds)==cell['absolute_tolerance_N'])),
            maximum_handle_work_error_J=max_work_error,maximum_fixture_work_error_J=max_support_error,
            maximum_energy_bookkeeping_identity_J=max_energy_identity,maximum_momentum_bookkeeping_error_Ns=max_momentum_error,
            energy_conservation_claim=False,contact_model='not_modeled',continuous_contact_verified=False)
        return report,data
    except (AuditError,ValueError,KeyError,IndexError,TypeError,OSError,StopIteration,np.linalg.LinAlgError) as error:
        report.update(state='invalid',error=str(error));return report,None


def audit_cell(seal_path,cell_id):
    """Stable read-only single-cell API for the staged execution supervisor.

    A passing result verifies this exact cell, never its temporal qualification.
    The supervisor may persist the bounded JSON return value. No files are made.
    """
    try:
        _,_,cell,pins=runner.validate_seal(seal_path,cell_id)
        require(sha(path(FIXTURE_PATH))==FIXTURE_SHA,'Canonical TRAIN fixture changed')
        report,_=verify_cell(cell,pins,read(path(FIXTURE_PATH)))
        report['prelaunch_seal']=evidence(str(seal_path),'reviewed_execution_seal')
        report['single_cell_temporal_or_dataset_qualification']=False
        return report
    except (ValueError,KeyError,TypeError,OSError) as error:
        return {'id':cell_id,'passed':False,'state':'invalid_seal','error':str(error)}


def coordinate_rmse(a,b):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and a.shape[0]==61 and a.shape[-1]==3,'Complete common 61-frame coordinates required')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite temporal comparison')
    return float(np.sqrt(np.mean((a-b)**2)))


def temporal_metric(trajectories,tolerance):
    errors=[coordinate_rmse(a,b) for a,b in zip(trajectories[:-1],trajectories[1:])]
    passed=len(errors)>=2 and errors[-1]<tolerance and (errors[-1]<errors[-2] or all(e<2e-6 for e in errors[-2:]))
    return {'adjacent_coordinate_rmse_m':errors,'passed':bool(passed),'metric':'full_q_coordinate_RMSE_all_61_frames'}


def analytic_static_zero(canonical,case_id):
    """Exact undeformed solution, explicitly NOT a captured/reset trajectory."""
    case=next(c for c in canonical['cases'] if c['episode_id']==case_id)
    require(case['recipe']['boundary_type']!='moving_patch','Moving zero must be freshly simulated')
    points,tets,faces,spec=compiler.forces_at.__globals__['geometry'](case['recipe'])
    arrays={'rest_points_m':points,'tetrahedra':tets,'surface_faces':faces}
    require(spec==case['boundaries'],'Generated analytic static-zero spec changed')
    for descriptor in case['reference_static_arrays']:
        array=arrays[descriptor['key']]
        require(list(array.shape)==descriptor['shape'] and str(array.dtype)==descriptor['dtype']
            and sha_bytes(array)==descriptor['sha256'],'Analytic rest geometry differs from canonical fixture')
    return np.broadcast_to(np.asarray(points,dtype=float),(61,*points.shape)).copy()


def unique_records(records):
    result={}
    for record in records:
        old=result.get(record['path'])
        require(old is None or old['sha256']==record['sha256'],'Evidence file changed between comparisons')
        result[record['path']]=record
    return list(result.values())


def temporal_candidates(protocol,reports,data,canonical,analytic_record):
    accepted={};zeros={};rows={r['cell']['id']:r for r in reports}
    for scenario in protocol['scenarios']:
        name=scenario['id'];rates=scenario['initial_rates_hz'];moving=scenario['source_case']=='t0003'
        ids=[name+'__temporal__hz'+str(hz) for hz in rates]
        zero_ids=['t0003__zero__temporal__hz'+str(hz) for hz in rates] if moving else []
        entry={'passed':False,'physics_hz':rates[-1],'output_hz':20,'frames':61,'all_actual_rates_hz':rates,
            'geometry_residual_checks_passed':False,'state':'pending_or_failed_cells',
            'required_temporal_cell_ids':ids+zero_ids}
        accepted[name]=entry
        if not all(id in data for id in ids+zero_ids):continue
        driven=[data[id]['positions_m'] for id in ids]
        zero=[data[id]['positions_m'] for id in zero_ids] if moving else [analytic_static_zero(canonical,scenario['source_case'])]*len(rates)
        signal=coordinate_rmse(driven[-1],zero[-1]);tolerance=max(2e-6,.05*signal)
        dmetric=temporal_metric(driven,tolerance);zmetric=temporal_metric(zero,tolerance)
        records=[record for id in ids+zero_ids for record in rows[id]['source_evidence_records']]
        if not moving:records.append(analytic_record)
        entry.update(state='evaluated',passed=dmetric['passed'] and zmetric['passed'],geometry_residual_checks_passed=True,
            force_induced_signal_m=signal,tolerance_m=tolerance,full_driven_temporal_metrics=dmetric,
            full_zero_temporal_metrics=zmetric,all_adjacent_errors_m={'driven':dmetric['adjacent_coordinate_rmse_m'],'zero':zmetric['adjacent_coordinate_rmse_m']},
            source_evidence_records=unique_records(records),zero_reference_kind='fresh_61frame_captured_moving_zero' if moving else 'analytic_rest_NOT_capture_NOT_reset')
    families=[id for id,entry in accepted.items() if id.startswith('t0003__') and entry['passed']]
    for hz in sorted({accepted[id]['physics_hz'] for id in families}):
        selected=[id for id in families if accepted[id]['physics_hz']==hz]
        zeros[str(hz)]={'passed':True,'physics_hz':hz,'output_hz':20,'frames':61,'geometry_residual_checks_passed':True,
            'passing_family_ids':selected,'source_evidence_records':unique_records([r for id in selected for r in accepted[id]['source_evidence_records']])}
    return accepted,{'t0003__zero_external_force':zeros}


REPLAY_KEYS=('positions_m','velocities_point_chord_m_s','auxiliary_positions_m','auxiliary_velocities_point_chord_m_s',
    'handle_centers_m','handle_rotations','times_s','actual_state_times_s','force_interval_actual_times_s',
    'force_midpoints_compiled_s','force_step_commands_N','handle_work_each_J','moment_about_world_origin_start_end_Nm')


def replay_metric(a,b):
    differences={};bitwise={}
    for key in REPLAY_KEYS:
        require(key in a and key in b and a[key].shape==b[key].shape,'Replay full-array schema differs '+key)
        differences[key]=float(np.max(np.abs(a[key]-b[key])))
        bitwise[key]=bool(np.array_equal(a[key],b[key]) and a[key].dtype==b[key].dtype)
    return {'passed':all(error<=1e-12 for error in differences.values()),'maximum_absolute_difference_by_array':differences,'bitwise_equal_by_array':bitwise}


def matched_comparisons(protocol,candidates,zero_candidates,reports,data):
    indexed={r['cell']['id']:r['cell'] for r in reports};comparisons=[];required=set()
    groups=[(id,e['physics_hz'],e['tolerance_m']) for id,e in candidates.items() if e['passed']]
    for hz,entry in zero_candidates['t0003__zero_external_force'].items():
        groups.append(('t0003__zero_external_force',int(hz),min(candidates[id]['tolerance_m'] for id in entry['passing_family_ids'])))
    templates=protocol['initial_cell_plan']
    for scenario,hz,tolerance in groups:
        moving_zero=scenario.endswith('__zero_external_force')
        baseline=('t0003__zero' if moving_zero else scenario)+'__temporal__hz'+str(hz)
        matched=[t for t in templates if t['stage']=='matched_after_temporal' and t['scenario_id']==scenario]
        resolved={t['id']:t['id']+'__hz'+str(hz) for t in matched};required.update(resolved.values())
        accuracy_kind='zero_absolute_floor' if moving_zero else 'nonlinear_tolerance'
        accuracy=sorted([t for t in matched if t['kind']==accuracy_kind],key=lambda t:t['absolute_tolerance_N'] if moving_zero else t['relative_tolerance'],reverse=True)
        accuracy_ids=[resolved[accuracy[0]['id']],baseline,resolved[accuracy[-1]['id']]]
        comp={'kind':accuracy_kind,'scenario_id':scenario,'hz':hz,'ids':accuracy_ids,'passed':False,'state':'pending','tolerance_m':tolerance}
        if all(id in data for id in accuracy_ids):
            errors=[coordinate_rmse(data[a]['positions_m'],data[b]['positions_m']) for a,b in zip(accuracy_ids[:-1],accuracy_ids[1:])]
            comp.update(state='evaluated',adjacent_coordinate_rmse_m=errors,passed=errors[-1]<tolerance,
                floor_dominance_by_cell={id:next(r['absolute_floor_dominated_steps'] for r in reports if r['id']==id) for id in accuracy_ids})
        comparisons.append(comp)
        for kind in ['translated','reset_replay']:
            row=next(t for t in matched if t['kind']==kind);id=resolved[row['id']]
            comp={'kind':kind,'scenario_id':scenario,'hz':hz,'ids':[baseline,id],'passed':False,'state':'pending','tolerance_m':tolerance if kind=='translated' else 1e-12}
            if baseline in data and id in data:
                comp['state']='evaluated'
                if kind=='translated':
                    error=coordinate_rmse(data[baseline]['positions_m'],data[id]['positions_m']-np.asarray(indexed[id]['translation_m']))
                    comp.update(coordinate_rmse_m=error,passed=error<tolerance)
                else:comp.update(replay_metric(data[baseline],data[id]))
            comparisons.append(comp)
    return comparisons,required


def audit_stage(seal_path,*,analytic_reference_record):
    """Verify one sealed stage plus its pinned temporal predecessor, no writes."""
    seal=read(path(seal_path));manifest=read(runner.checked_record(seal['resolved_manifest']))
    require(bool(manifest['cells']),'Empty resolved stage')
    protocol,_,_,pins=runner.validate_seal(seal_path,manifest['cells'][0]['id'])
    require(sha(path(FIXTURE_PATH))==FIXTURE_SHA and sha(path('experiments/campaign_authored_commands.py'))==COMPILER_SHA,'Canonical fixture/compiler changed')
    canonical=read(path(FIXTURE_PATH));seals=[str(seal_path)]
    if manifest['stage']=='matched_after_temporal':
        for row in manifest['cells']:
            proof=read(runner.checked_record(row['temporal_candidate_proof']))
            require(proof['auditor_source_sha256']==sha(path(AUDITOR_PATH)),'Candidate auditor revision changed')
            for previous in proof['temporal_stage_seals']:
                runner.checked_record(previous)
                if previous['path'] not in seals:seals.append(previous['path'])
        require(len(seals)==2,'Exactly one complete original temporal stage required; conditional extensions need a new reviewed audit')
    reports=[];data={};outputs=set();ids=set();source_seals=[];temporal_seals=[]
    for source_seal in seals:
        value=read(path(source_seal));m=read(runner.checked_record(value['resolved_manifest']))
        p,_,_,source_pins=runner.validate_seal(source_seal,m['cells'][0]['id'])
        # Stage seals/manifests/proofs legitimately differ; the physical source,
        # compiler, context and auditor bytes must stay identical.
        require(p==protocol and all(source_pins.get(key)==digest for key,digest in runner.dependencies().items())
            and source_pins[AUDITOR_PATH]==pins[AUDITOR_PATH],'Stage physical source/protocol pins changed')
        if source_seal!=str(seal_path):require(m['stage']=='initial_temporal','Predecessor must be original temporal stage')
        source_seals.append(evidence(source_seal,'verified_stage_execution_seal'))
        if m['stage']=='initial_temporal':temporal_seals.append(source_seals[-1])
        for row in m['cells']:
            require(row['id'] not in ids and row['output'] not in outputs,'Cross-stage duplicate cell/output')
            ids.add(row['id']);outputs.add(row['output']);report,arrays=verify_cell(row,source_pins,canonical);reports.append(report)
            if report['passed']:data[row['id']]=arrays
    require(analytic_reference_record['sha256']==sha(path(analytic_reference_record['path'])),'Analytic reference declaration hash mismatch')
    analytic=read(path(analytic_reference_record['path']))
    require(analytic==analytic_declaration(protocol),'Analytic static-zero declaration changed')
    candidates,zeros=temporal_candidates(protocol,reports,data,canonical,analytic_reference_record)
    comparisons,matched_required=matched_comparisons(protocol,candidates,zeros,reports,data)
    temporal_required={r['id'] for r in protocol['initial_cell_plan'] if r['stage']=='initial_temporal'}
    all_temporal=all(e['passed'] for e in candidates.values())
    qualified=all_temporal and (temporal_required|matched_required)==ids and len(data)==len(ids) and all(c['passed'] for c in comparisons)
    return {'schema':'actual-authored-force-engineering-audit-v0.1','protocol_sha256':PROTOCOL_SHA,
        'auditor_source_sha256':sha(path(AUDITOR_PATH)),'temporal_stage_seals':temporal_seals,
        'stage_seals':source_seals,'small_force_prerequisite_sha256':PREREQUISITE_SHA,
        'required_initial_physical_cells_if_one_moving_rate':119,'indexed_unique_cells':len(ids),'verified_unique_cells':len(data),
        'cells':reports,'accepted_candidates':candidates,'accepted_zero_candidates':zeros,'matched_comparisons':comparisons,
        'all_16_temporal_candidates_passed':all_temporal,'actual_force_declared_discrete_engineering_envelope_qualified':qualified,
        'production_neural_dataset_admitted':False,'native_reference_qualified':False,'continuum_accuracy_established':False,
        'prediction_qualified':False,'inference_qualified':False,'active_policy_qualified':False,
        'limits':['Only four public TRAIN layouts, authored force families/materials, fixed mesh and stated discrete model.',
            'Factorized temporal/accuracy/translation/replay controls do not establish all interactions.',
            'BC-dependent solver/command clocks and source metadata are engineering provenance, never public neural features.',
            'Production requires a common known actuator clock independent of latent BC and prospective dataset admission.',
            'No native elasticity damping, contact, continuous nonintersection or real-material/continuum qualification.',
            'Analytic static-zero references are not captured trajectories or reset tests; old failed evidence remains unchanged.']}


def analytic_declaration(protocol):
    return {'kind':'analytic_static_zero_NOT_CAPTURE_NOT_RESET','fixture_sha256':FIXTURE_SHA,
        'physical_source_sha256':protocol['frozen_source_sha256'],'requested_output_times_s':(np.arange(61)/20).tolist(),
        'grids':[{'case':'t000'+str(case),'hz':hz,'frames':61,'solution':'exact rest q and zero velocity, no gravity/external force; static fixed constraints'}
            for case in range(3) for hz in [120,240,480]],'frames_upsampled_from_prior':False}


def run(seal_path,output):
    output=str(output);target=path(output);require(not target.exists(),'Fresh audit output required')
    sidecar=output.removesuffix('.json')+'-analytic-static-zero.json';require(not path(sidecar).exists(),'Fresh analytic reference record required')
    protocol=read(path(PROTOCOL_PATH));target.parent.mkdir(parents=True,exist_ok=True)
    path(sidecar).write_text(json.dumps(analytic_declaration(protocol),indent=2,allow_nan=False)+'\n')
    result=audit_stage(str(seal_path),analytic_reference_record=evidence(sidecar,'analytic_static_zero_NOT_CAPTURE_NOT_RESET'))
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prelaunch-seal',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.prelaunch_seal,args.output)
    print(json.dumps({key:result[key] for key in ['indexed_unique_cells','verified_unique_cells','all_16_temporal_candidates_passed','actual_force_declared_discrete_engineering_envelope_qualified']}))
