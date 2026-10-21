"""Independent, read-only audit of the factorized TRAIN dynamic gate matrix.

Reads saved protocols, canonical static inputs, step ledgers and complete
trajectories. Never launches a model, loads reserved observations, replaces a
failed cell, or qualifies native Isaac. A missing supplemental study remains
incomplete. Numeric audit tolerances below concern floating-point bookkeeping,
not a replacement of the published physical trajectory tolerance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

GATE_SHA='962457955d72a97dd0d8e48a41aff0c6ead821931031343e081cf837c561b2a0'
FLOOR_SHA='ab4d663af320b1e1863d07239b473982b340992b2b6e6ff7554c9a9350fc358e'
GATE_PATH=Path('configs/gaussian_campaign_independent_dynamic_gate_v01.json')
INPUT_PATH=Path('configs/gaussian_independent_diagnostic_inputs_v01.json')
INPUT_SHA='073380f72559c50ccc674d5f69fdff1be7e4d0b30484ad66f0233695a813847f'
FAMILIES={'t0000':'point_pin','t0001':'fixed_patch','t0002':'two_patches','t0003':'moving_patch'}
RESPONSES=['zero_external_force','diagonal_pulse_handle0']
CORE=['campaign_nonlinear_dynamics.py','campaign_nonlinear_fem.py','campaign_linear_fem.py','campaign_rigid_inertia.py']
REPLAY_ATOL=1e-12


class AuditError(ValueError):pass
def require(value,message):
    if not value:raise AuditError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def reject_constant(value):raise AuditError('Nonfinite JSON constant '+value)
def read(path):return json.loads(Path(path).read_text(),parse_constant=reject_constant)
def finite(value,name):
    array=np.asarray(value,dtype=float);require(np.isfinite(array).all(),'Nonfinite '+name);return array
def close(a,b,atol,message):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.all(np.abs(a-b)<=atol),message)
def finite_tree(value):
    if isinstance(value,dict):
        for child in value.values():finite_tree(child)
    elif isinstance(value,list):
        for child in value:finite_tree(child)
    elif isinstance(value,(int,float)):
        require(np.isfinite(value),'Nonfinite numeric ledger value')


def slot(cell):
    return (cell['case'],cell['response'],int(cell['hz']),float(cell['relative_tolerance']),
        tuple(map(float,cell['translation_m'])),cell['kind'],float(cell.get('absolute_tolerance_N',1e-10)))


def required_slots(supplemental=False):
    rows=[]
    for case in FAMILIES:
        for response in RESPONSES:
            for hz in [120,240,480]:rows.append((case,response,hz,1e-7,(0.,0.,0.),'temporal',1e-10))
            if response=='diagonal_pulse_handle0':
                for tolerance in [1e-5,1e-9]:rows.append((case,response,480,tolerance,(0.,0.,0.),'nonlinear_tolerance',1e-10))
            rows.append((case,response,480,1e-7,(0.,0.,-.25),'translated',1e-10))
            rows.append((case,response,480,1e-7,(0.,0.,0.),'reset_replay',1e-10))
    if supplemental:
        for floor in [1e-8,1e-12]:rows.append(('t0003','zero_external_force',480,1e-7,(0.,0.,0.),'absolute_floor',floor))
    return rows


def coverage(cells,supplemental=False):
    actual=[slot(c) for c in cells];needed=required_slots(supplemental)
    return {'expected_cells':len(needed),'indexed_cells':len(actual),
        'missing_slots':[list(x) for x in needed if x not in actual],
        'unexpected_slots':[list(x) for x in actual if x not in needed],
        'duplicate_slots':[list(x) for x in set(actual) if actual.count(x)>1],
        'duplicate_output_paths':len({str(Path(c['output']).resolve()) for c in cells})!=len(cells),
        'factorized_dimensions':True,
        'interaction_limit':'Time/refinement, nonlinear accuracy, translation and replay are sampled as declared separate slices; their full Cartesian interactions are not tested.'}


def pulse_forces(gate,response,midpoints):
    peak=np.asarray([gate['force']['peak_world_force_N_handle0'],gate['force']['peak_world_force_N_handle1']],dtype=float)
    if response=='zero_external_force':return np.zeros((len(midpoints),2,3))
    require(response=='diagonal_pulse_handle0','Unknown response')
    phase=(np.asarray(midpoints)/3.-.1)/.45
    amplitude=np.where((phase>0)&(phase<1),np.sin(np.pi*np.clip(phase,0,1))**2,0.)
    return amplitude[:,None,None]*peak


def fixture_positions(rest,spec,times):
    fixed=np.asarray(spec['fixture_nodes'],dtype=int)
    times=np.asarray(times,dtype=float)
    shift=np.zeros((len(times),3))
    if spec['boundary_type']=='moving_patch':
        shift=float(spec['motion_amplitude_m'])*np.sin(np.pi*np.clip(times/3.,0,1))[:,None]**2*np.asarray(spec['motion_direction_world'])
    return rest[fixed][None]+shift[:,None,:]


def geometry_frame_checks(data,rest,tets,spec):
    q=data['positions_m'];v=data['velocities_point_chord_m_s'];centers=data['handle_centers_m'];R=data['handle_rotations']
    require(q.shape==(31,len(rest),3) and v.shape==q.shape,'Full q/v frame shape')
    require(centers.shape==(31,2,3) and R.shape==(31,2,3,3),'Full rigid pose frame shape')
    close(q[0],rest,1e-12,'Initial positions differ from rest');close(v[0],np.zeros_like(rest),1e-12,'Initial velocity not zero')
    close(np.swapaxes(R,-1,-2)@R,np.broadcast_to(np.eye(3),R.shape),2e-12,'Invalid rigid rotation')
    close(np.linalg.det(R),np.ones((31,2)),2e-12,'Improper rotation')
    patch_error=0.
    for h,(node,patch) in enumerate(zip(spec['load_nodes'],spec['load_patches'])):
        expected=centers[:,h,None,:]+np.einsum('ij,tkj->tik',rest[patch]-rest[node],R[:,h])
        patch_error=max(patch_error,float(np.max(abs(q[:,patch]-expected))))
    require(patch_error<=1e-12,'Finite rigid patch positions fail')
    fixed=np.asarray(spec['fixture_nodes'],dtype=int)
    target=fixture_positions(rest,spec,data['actual_state_times_s'])
    close(q[:,fixed],target,1e-12,'Prescribed fixture positions fail')
    dm=np.swapaxes(rest[tets[:,1:]]-rest[tets[:,:1]],1,2)
    gradients=np.linalg.inv(dm);gradients=np.concatenate([-gradients.sum(1,keepdims=True),gradients],axis=1)
    G=np.einsum('teic,eid->tecd',q[:,tets]-rest[tets],gradients)
    F=G+np.eye(3);J=np.linalg.det(F)
    require(np.min(J)>0,'Nonpositive saved-frame volume')
    green=.5*(np.swapaxes(F,-1,-2)@F-np.eye(3))
    return {'maximum_rigid_patch_position_error_m':patch_error,
        'minimum_saved_frame_J':float(J.min()),'maximum_saved_frame_J':float(J.max()),
        'maximum_saved_Green_principal_strain':float(abs(np.linalg.eigvalsh(green)).max()),
        'maximum_saved_nodal_displacement_m':float(np.linalg.norm(q-rest,axis=-1).max()),
        'maximum_saved_chord_speed_m_s':float(np.linalg.norm(v,axis=-1).max())}


def audit_cell(cell,gate,canonical,approved_runners,expected_dependencies,supplemental_dependencies=None):
    root=Path(cell['output']);result={'id':cell.get('id'), 'slot':list(slot(cell)),'output':str(root),'passed':False}
    data=None
    try:
        require(cell['case'] in FAMILIES,'Reserved/unknown case index rejected before reading trajectory')
        if not root.exists():result['state']='missing';return result,None
        require((root/'protocol.json').exists(),'Missing protocol')
        protocol=read(root/'protocol.json')
        require(protocol['episode_id']==cell['case'] and protocol['boundary_kind']==FAMILIES[cell['case']],'Protocol TRAIN identity mismatch')
        require(protocol['predictive_reserved_responses_opened'] is False,'Reserved-data role violation')
        if (root/'failure.json').exists():
            result.update(state='failed_preserved',failure_sha256=sha(root/'failure.json'));return result,None
        if not (root/'execution.json').exists():result['state']='incomplete';return result,None
        execution=read(root/'execution.json')
        if not execution.get('terminal') or not execution.get('completed'):
            result.update(state='incomplete',completed_steps=execution.get('completed_steps'));return result,None
        hz=cell['hz'];n=3*hz;rtol=cell['relative_tolerance'];floor=cell.get('absolute_tolerance_N',1e-10)
        require(protocol['physics_hz']==hz and protocol['seconds']==3. and protocol['output_hz']==10,'Protocol physical clocks mismatch')
        require(protocol['response']==cell['response'] and protocol['relative_nonlinear_residual_tolerance']==rtol,'Force/accuracy metadata mismatch')
        require(protocol['absolute_residual_floor_N']==floor and protocol['nonlinear_iteration_cap']==250,'Effective floor/cap mismatch')
        close(protocol['translation_m'],cell['translation_m'],0.,'Translation metadata mismatch')
        by_name={}
        for original,digest in protocol['source_input_dependency_and_gate_sha256'].items():
            name=Path(original).name;require(name not in by_name,'Ambiguous provenance basename')
            require(sha(root/'provenance'/name)==digest,'Saved source/input hash mismatch '+name);by_name[name]=digest
        require(by_name.get(GATE_PATH.name)==GATE_SHA,'Wrong published gate')
        require(by_name.get(INPUT_PATH.name)==INPUT_SHA,'Canonical public static fixture is not verified')
        require(by_name.get('campaign_nonlinear_dynamics_case.py') in approved_runners,'Undeclared runner source revision')
        require(all(name in by_name for name in CORE),'Incomplete core provenance')
        require(all(by_name.get(name)==digest for name,digest in expected_dependencies.items()),'Core/physical dependency differs from frozen manifest')
        if cell['kind']=='absolute_floor':
            require(by_name.get('gaussian_campaign_dynamic_floor_refinement_v01.json')==FLOOR_SHA,'Saved supplemental protocol differs')
            require(supplemental_dependencies is not None and 'campaign_dynamic_floor_protocol.py' in supplemental_dependencies,'Supplemental helper revision must be declared')
            require(all(by_name.get(name)==digest for name,digest in supplemental_dependencies.items()),'Supplemental helper/source differs from declaration')
            metadata=protocol['supplemental_accuracy_protocol']
            require(metadata['protocol_sha256']==FLOOR_SHA and metadata['parent_gate_sha256']==GATE_SHA
                and metadata['absolute_residual_floor_N']==floor and metadata['baseline_absolute_residual_floor_N_reused']==1e-10
                and metadata['physical_model_unchanged'] is True,'Supplemental request metadata mismatch')
        selected=next(c for c in canonical['cases'] if c['episode_id']==cell['case'])
        require(protocol['declared_material']==selected['recipe']['material'],'Material differs from declared TRAIN static input')
        require(protocol['body_only_viscous_damping_s_inverse']==selected['recipe']['material']['linear_damping_s_inverse'],'Body damping mismatch')
        require(protocol['native_elasticity_damping_reproduced'] is False,'Changed constitutive/damping semantics')
        static_path=root/'provenance/static-reference.npz'
        require(sha(static_path)==protocol['saved_regenerated_static_reference_sha256'],'Saved static mesh file hash mismatch')
        with np.load(static_path) as raw:static={key:raw[key].copy() for key in ['rest_points_m','tetrahedra','surface_faces']}
        for row in selected['reference_static_arrays']:
            array=np.ascontiguousarray(static[row['key']])
            require(str(array.dtype)==row['dtype'] and list(array.shape)==row['shape'],'Static array dtype/shape mismatch')
            require(hashlib.sha256(array.tobytes()).hexdigest()==row['sha256'],'Canonical static array digest mismatch')
        rest=static['rest_points_m']+np.asarray(cell['translation_m']);spec=selected['boundaries'];tets=static['tetrahedra']
        require(len(protocol['hardware'])==2,'Both handles required')
        for h,(node,patch) in zip(protocol['hardware'],zip(spec['load_nodes'],spec['load_patches'])):
            require(h['reference_origin_node']==node and h['ordered_patch_nodes']==patch,'Hardware origins/order mismatch')
            close(h['reference_origin_m'],rest[node],0.,'Hardware world origin mismatch')
            local=rest[patch]-rest[node];close(h['ordered_reference_local_offsets_m'],local,0.,'Hardware local offsets mismatch')
            mass=spec['handle_mass_kg_each'];inertia=np.maximum(mass*np.mean(np.sum(local**2,axis=1)[:,None]-local**2,axis=0),.4*mass*.003**2)
            require(h['mass_kg']==mass,'Hardware mass mismatch')
            close(h['full_COM_inertia_kg_m2'],np.diag(inertia),1e-20,'Hardware inertia mismatch')
            close(h['COM_offset_m'],np.zeros(3),0.,'Undeclared COM offset');close(h['initial_rotation_matrix'],np.eye(3),0.,'Undeclared initial rotation')
        summary=read(root/'summary.json')
        require(summary['completed'] and execution['completed_steps']==n and summary['frames']==31,'Incomplete execution/frames')
        require(sha(root/'trajectory.npz')==summary['trajectory_sha256'] and sha(root/'steps.jsonl')==summary['steps_sha256'],'Trajectory/ledger digest mismatch')
        with np.load(root/'trajectory.npz') as raw:data={key:raw[key].copy() for key in raw.files}
        for key,value in data.items():finite(value,key)
        nominal=np.arange(31)/10
        close(data['times_s'],nominal,0.,'Nominal sample times');close(data['actual_state_times_s'],nominal,1e-12,'Actual sample clock')
        intervals=data['force_interval_actual_times_s'];require(intervals.shape==(n,2),'Full step interval shape')
        close(intervals[:,0],np.arange(n)/hz,1e-12,'Step start clocks');close(intervals[:,1],np.arange(1,n+1)/hz,1e-12,'Step end clocks')
        close(intervals[1:,0],intervals[:-1,1],0.,'Step clock discontinuity');close(data['requested_step_s'],np.full(n,1/hz),0.,'Requested step size')
        midpoints=intervals.mean(1);close(data['force_midpoints_actual_s'],midpoints,1e-15,'Held force midpoint')
        expected=pulse_forces(gate,cell['response'],midpoints)
        close(data['force_step_commands_N'],expected,2e-15,'Held force command')
        dt=intervals[:,1]-intervals[:,0];impulse=np.sum(expected*dt[:,None,None],axis=0)
        analytic=np.zeros((2,3)) if cell['response']=='zero_external_force' else .675*np.asarray([gate['force']['peak_world_force_N_handle0'],gate['force']['peak_world_force_N_handle1']])
        close(impulse,analytic,1e-12,'Analytic pulse impulse');close(summary['command_impulse_Ns'],impulse,1e-14,'Summary impulse mismatch')
        geom=geometry_frame_checks(data,rest,tets,spec)
        rows=[json.loads(line,parse_constant=reject_constant) for line in (root/'steps.jsonl').read_text().splitlines()]
        require(len(rows)==n,'Incomplete full-step ledger')
        final_residual=[];thresholds=[];support_error=[];identity=[];runtime_J=[];runtime_strain=[]
        momentum_errors=[];runtime_speed=[];runtime_displacement=[];handle_work_errors=[];last_centers=None
        work_totals={key:0. for key in ['support_work_J','held_handle_work_J','held_nodal_work_J','body_viscous_dissipation_J','constraint_chord_work_J','implicit_velocity_increment_loss_J']}
        for i,row in enumerate(rows):
            finite_tree(row)
            require(row['step_index']==i and row['converged'] is True,'Unaccepted step')
            close([row['start_time_s'],row['end_time_s']],intervals[i],0.,'Ledger clocks')
            close(np.asarray(row['step_s']),np.asarray(1/hz),0.,'Ledger requested step')
            close(row['held_handle_forces_N'],expected[i],2e-15,'Ledger held force')
            require(row['nodal_force_command']=='all_zero','Undeclared nodal force')
            tolerance=max(floor,rtol*float(np.linalg.norm(expected[i])))
            require(abs(row['residual_tolerance_N']-tolerance)<=max(1e-25,1e-12*tolerance),'Effective residual tolerance mismatch')
            history=row['nonlinear_history'];require(len(history)>=1 and len(history)-1<=250,'Missing/capped nonlinear history')
            require(row['nonlinear_iterations']==len(history)-1==data['nonlinear_iterations'][i],'Nonlinear iteration bookkeeping mismatch')
            residual=float(history[-1]['scaled_generalized_residual_norm_N'])
            require(np.isfinite(residual) and residual>=0 and residual<=tolerance*(1+1e-10),'Residual acceptance failure')
            for entry in history:
                require(np.isfinite(entry['incremental_potential_J']) and entry['minimum_J']>0,'Invalid nonlinear history/J')
            g=row['geometry'];require(row['minimum_J']>0 and g['minimum_volume_ratio']>0,'Nonpositive runtime J')
            require(g['maximum_rigid_patch_distance_error_m']<=1e-12 and g['fixture_max_error_m']<=1e-12,'Runtime exact constraint error')
            target=fixture_positions(rest,spec,intervals[i]);delta=target[1]-target[0]
            reaction=finite(row['fixture_reactions_inertia_elastic_drag_minus_external_N'],'fixture reaction')
            require(reaction.shape==(len(spec['fixture_nodes']),3),'Fixture reaction shape')
            close(np.asarray(row['support_work_J']),np.asarray(np.sum(reaction*delta)),1e-12,'Independent fixture reaction work')
            support_error.append(abs(row['support_work_J']-float(np.sum(reaction*delta))))
            center_keys=['handle_centers_start_m','handle_centers_end_m','handle_center_step_increment_m']
            if any(key in row for key in center_keys):
                require(all(key in row for key in center_keys),'Incomplete per-step center evidence')
                start,end,increment=[finite(row[key],key) for key in center_keys]
                require(start.shape==end.shape==increment.shape==(2,3),'Per-step handle center shape')
                close(increment,end-start,1e-15,'Stable center increment disagrees with world poses')
                close(start,data['handle_centers_m'][0] if i==0 else last_centers,0.,'Per-step handle center discontinuity')
                if (i+1)%(hz//10)==0:close(end,data['handle_centers_m'][(i+1)//(hz//10)],0.,'Per-step center/sample mismatch')
                last_centers=end
                held_work=float(np.sum(expected[i]*increment))
                close(np.asarray(row['held_handle_work_J']),np.asarray(held_work),1e-15,'Independent held-force handle work')
                handle_work_errors.append(abs(held_work-row['held_handle_work_J']))
            old=row['old_invariants'];new=row['new_invariants']
            defect=new['energy_J']-old['energy_J']-(row['held_handle_work_J']+row['held_nodal_work_J']+row['support_work_J']-row['body_viscous_dissipation_J'])
            reconstructed=defect-(row['constraint_chord_work_J']-row['implicit_velocity_increment_loss_J']-row['backward_elastic_gradient_defect_J'])
            require(np.isfinite(reconstructed) and abs(reconstructed-row['energy_algebra_identity_residual_J'])<=1e-12,'Energy ledger algebra mismatch')
            momentum=(np.asarray(new['point_linear_momentum_Ns'])-np.asarray(old['point_linear_momentum_Ns'])
                -dt[i]*(expected[i].sum(0)+reaction.sum(0))+np.asarray(row['body_drag_impulse_Ns']))
            close(momentum,row['linear_momentum_balance_error_Ns'],1e-12,'Independent linear momentum ledger reconstruction')
            momentum_errors.append(float(np.linalg.norm(momentum)))
            runtime_speed.append(g['maximum_point_chord_speed_m_s']);runtime_displacement.append(g['maximum_nodal_displacement_m'])
            for key in work_totals:work_totals[key]+=row[key]
            final_residual.append(residual);thresholds.append(tolerance);identity.append(abs(reconstructed));runtime_J.append(g['minimum_volume_ratio']);runtime_strain.append(g['maximum_absolute_Green_principal_strain'])
        require(len(handle_work_errors) in [0,n],'Per-step centers missing from some steps')
        result.update(state='verified',passed=True,source_sha256={name:by_name[name] for name in CORE},
            physical_input_sha256={key:value for key,value in by_name.items() if key in [INPUT_PATH.name,'isaac_dataset.py','boundary_block.py']},
            runner_sha256=by_name['campaign_nonlinear_dynamics_case.py'],trajectory_sha256=summary['trajectory_sha256'],steps_sha256=summary['steps_sha256'],
            geometry=geom,minimum_runtime_J=min(runtime_J),maximum_runtime_Green_principal_strain=max(runtime_strain),
            maximum_accepted_physical_residual_N=max(final_residual),effective_threshold_min_N=min(thresholds),effective_threshold_max_N=max(thresholds),
            absolute_floor_dominated_steps=int(np.sum(np.asarray(thresholds)==floor)),steps=n,
            maximum_fixture_work_reconstruction_error_J=max(support_error),maximum_energy_algebra_identity_residual_J=max(identity),
            maximum_linear_momentum_ledger_error_Ns=max(momentum_errors),maximum_runtime_point_chord_speed_m_s=max(runtime_speed),
            maximum_runtime_nodal_displacement_m=max(runtime_displacement),reported_work_and_discretization_sums=work_totals,
            external_handle_work_independently_reconstructed_from_saved_step_positions=bool(handle_work_errors),
            maximum_handle_work_reconstruction_error_J=max(handle_work_errors,default=None),
            energy_conservation_claim=False,
            work_scope=('Fixture and held-force handle work independently reconstructed from recorded increments.' if handle_work_errors else
                'Fixture work independently reconstructed; handle work is source/mechanics-test/ledger verified because per-step handle centers are not persisted.')+' Algebra identity is bookkeeping, not conservation.')
        return result,data
    except (AuditError,ValueError,KeyError,IndexError,OSError,TypeError,StopIteration) as error:
        result.update(state='invalid',error=str(error));return result,None


def coordinate_rmse(a,b):
    a,b=np.asarray(a),np.asarray(b);require(a.shape==b.shape,'Matched full-trajectory shape required')
    return float(np.sqrt(np.mean((a-b)**2)))


def temporal_metrics(trajectories,zero_finest,diagonal_finest):
    signal=coordinate_rmse(diagonal_finest,zero_finest);tolerance=max(2e-6,.05*signal)
    differences=[coordinate_rmse(trajectories[i+1],trajectories[i]) for i in range(2)]
    floor_exception=all(x<2e-6 for x in differences)
    return {'coordinate_rmse_m':differences,'finest_force_induced_coordinate_RMS_m':signal,'tolerance_m':tolerance,
        'both_differences_below_2micrometre_floor':floor_exception,
        'passed':bool(differences[-1]<tolerance and (differences[-1]<differences[0] or floor_exception))}


def mechanics_audit(directory,core_expected):
    directory=Path(directory)
    try:
        proof=read(directory/'validation.json')
        for original,digest in proof['source_and_gate_sha256'].items():require(sha(directory/'provenance'/Path(original).name)==digest,'Mechanics source proof mismatch')
        for original,digest in proof['test_result_sha256'].items():require(sha(directory/Path(original).name)==digest,'Mechanics XML digest mismatch')
        tree=ET.parse(directory/'tests.xml');tests=tree.findall('.//testcase')
        require(len(tests)==proof['tests'] and len(tests)>=31,'Mechanics suite incomplete')
        require(not any(t.find('failure') is not None or t.find('error') is not None or t.find('skipped') is not None for t in tests),'Mechanics test not passed')
        require(proof['failures']==proof['errors']==0,'Mechanics failure count')
        spin=[p.attrib['value'] for p in tree.findall('.//property') if p.attrib.get('name')=='free_spin_refinement']
        require(len(spin)==1,'Missing free-spin refinement record');spin=json.loads(spin[0]);require(spin==proof['free_spin_refinement'],'Spin proof/XML mismatch')
        errors=finite(spin['relative_energy_and_chord_angular_momentum_errors'],'spin errors')
        require(errors.shape==(3,2) and np.all(errors[1:]<errors[:-1]),'Spin errors do not refine')
        source={Path(k).name:v for k,v in proof['source_and_gate_sha256'].items()}
        require(all(source.get(k)==v for k,v in core_expected.items()),'Mechanics/core revision mismatch')
        return {'passed':True,'tests':len(tests),'validation_sha256':sha(directory/'validation.json'),'spin_refinement':spin}
    except (AuditError,ValueError,KeyError,OSError,ET.ParseError) as error:return {'passed':False,'error':str(error)}


def run(manifest_path,output,*,supplemental_manifest=None,mechanics_directory='runs/operator-dev/nonlinear-dynamics-mechanics-v01'):
    require(sha(GATE_PATH)==GATE_SHA and sha(INPUT_PATH)==INPUT_SHA,'Published gate/static-input fixture changed')
    gate=read(GATE_PATH);canonical=read(INPUT_PATH);manifest=read(manifest_path)
    require(manifest['gate_sha256']==GATE_SHA,'Manifest gate mismatch')
    cells=list(manifest['cells']);approved=set(manifest['approved_runner_source_sha256'])
    expected_dependencies={Path(name).name:digest for name,digest in manifest['frozen_source_input_and_gate_sha256'].items()
        if Path(name).name in CORE+['isaac_dataset.py','boundary_block.py',INPUT_PATH.name]}
    require(all(name in expected_dependencies for name in CORE),'Manifest must pin all physical core sources')
    supplemental_present=supplemental_manifest is not None;supplemental_dependencies=None
    if supplemental_present:
        supplemental=read(supplemental_manifest)
        require(supplemental['parent_gate_sha256']==GATE_SHA and supplemental['floor_protocol_sha256']==FLOOR_SHA,'Supplemental protocol linkage mismatch')
        require(sha('configs/gaussian_campaign_dynamic_floor_refinement_v01.json')==FLOOR_SHA,'Supplemental floor protocol changed')
        supplemental_dependencies=supplemental['supplemental_dependency_sha256']
        require('campaign_dynamic_floor_protocol.py' in supplemental_dependencies,'Supplemental helper revision must be pinned')
        cells+=supplemental['cells'];approved.update(supplemental['approved_runner_source_sha256'])
    coverage_report=coverage(cells,supplemental=True)
    results=[];loaded={};source_family=None;input_family=None
    for cell in cells:
        report,data=audit_cell(cell,gate,canonical,approved,expected_dependencies,supplemental_dependencies);results.append(report)
        if report['passed']:
            if source_family is None:source_family=report['source_sha256'];input_family=report['physical_input_sha256']
            if source_family!=report['source_sha256'] or input_family!=report['physical_input_sha256']:
                report.update(passed=False,state='invalid',error='Cross-cell physical source/input hashes differ')
            else:loaded[slot(cell)]=data
    def get(case,response,hz=480,tol=1e-7,shift=(0.,0.,0.),kind='temporal',floor=1e-10):
        return loaded.get((case,response,hz,tol,shift,kind,floor))
    comparisons=[];static_zero=[]
    for case in FAMILIES:
        zero=get(case,RESPONSES[0]);diagonal=get(case,RESPONSES[1])
        if zero is None or diagonal is None:continue
        tolerance=max(2e-6,.05*coordinate_rmse(diagonal['positions_m'],zero['positions_m']))
        for response in RESPONSES:
            levels=[get(case,response,hz) for hz in [120,240,480]]
            if all(x is not None for x in levels):comparisons.append({'case':case,'response':response,'kind':'temporal',**temporal_metrics([x['positions_m'] for x in levels],zero['positions_m'],diagonal['positions_m'])})
            base=get(case,response)
            for kind,shift in [('translated',(0.,0.,-.25)),('reset_replay',(0.,0.,0.))]:
                other=get(case,response,shift=shift,kind=kind)
                if other is None:continue
                difference=coordinate_rmse(base['positions_m'],other['positions_m']-np.asarray(shift))
                replay=kind=='reset_replay';limit=REPLAY_ATOL if replay else tolerance
                arrays_equal={key:bool(np.array_equal(base[key],other[key])) for key in ['positions_m','velocities_point_chord_m_s','handle_centers_m','handle_rotations','force_step_commands_N']}
                replay_error=max(float(abs(base[key]-other[key]).max()) for key in arrays_equal) if replay else None
                comparisons.append({'case':case,'response':response,'kind':kind,'coordinate_rmse_m':difference,'tolerance_m':limit,
                    'replay_arrays_bitwise':arrays_equal if replay else None,'passed':bool(replay_error<=limit if replay else difference<tolerance)})
        levels=[get(case,RESPONSES[1],tol=t,kind='temporal' if t==1e-7 else 'nonlinear_tolerance') for t in [1e-5,1e-7,1e-9]]
        if all(x is not None for x in levels):
            diffs=[coordinate_rmse(levels[i+1]['positions_m'],levels[i]['positions_m']) for i in range(2)]
            comparisons.append({'case':case,'response':RESPONSES[1],'kind':'nonlinear_tolerance','coordinate_rmse_m':diffs,'tolerance_m':tolerance,'passed':bool(diffs[-1]<tolerance)})
        if case!='t0003':
            passed=bool(np.max(abs(zero['positions_m']-zero['positions_m'][0]))<=1e-12 and np.max(abs(zero['velocities_point_chord_m_s']))<=1e-12)
            static_zero.append({'case':case,'analytic_zero_rest_passed':passed,'nominal_zero_tolerance_duplicates_required':False})
        else:
            levels=[get(case,RESPONSES[0],kind='temporal' if f==1e-10 else 'absolute_floor',floor=f) for f in [1e-8,1e-10,1e-12]]
            if all(x is not None for x in levels):
                diffs=[coordinate_rmse(levels[i+1]['positions_m'],levels[i]['positions_m']) for i in range(2)]
                comparisons.append({'case':case,'response':RESPONSES[0],'kind':'absolute_floor','floors_N':[1e-8,1e-10,1e-12],'coordinate_rmse_m':diffs,'tolerance_m':tolerance,'passed':bool(diffs[-1]<tolerance)})
    mechanics=mechanics_audit(mechanics_directory,source_family or {})
    complete=not any(coverage_report[key] for key in ['missing_slots','unexpected_slots','duplicate_slots','duplicate_output_paths'])
    # Eight temporal, eight translation, eight replay, four diagonal tolerance,
    # and one moving-zero absolute-floor comparison are all indispensable.
    qualified=bool(complete and len(results)==50 and all(r['passed'] for r in results) and len(comparisons)==29 and all(c['passed'] for c in comparisons) and len(static_zero)==3 and all(z['analytic_zero_rest_passed'] for z in static_zero) and mechanics['passed'])
    result={'audit_source_sha256':sha(__file__),'manifest_sha256':sha(manifest_path),'gate_sha256':GATE_SHA,'coverage':coverage_report,
        'cells':results,'comparisons':comparisons,'analytic_static_zero_controls':static_zero,'mechanics':mechanics,
        'independent_fixed_discretization_dynamic_gate_passed':qualified,'native_reference_qualified':False,
        'tested_operating_envelope':{'responses':RESPONSES,'peak_world_handle0_force_N':[.01,.01,.005],
            'peak_world_handle1_force_N':[0.,0.,0.],'profile':'Only the specified3second sin-squared midpoint-held pulse or zero external force, with the declared support laws/materials/initial rest.',
            'original_recipe_amplitudes_and_other_force_profiles_qualified':False,
            'bulk_original_amplitude_dataset_generation_qualified':False,
            'prospective_actual_force_operating_envelope_refinement_still_required':True},
        'continuum_accuracy_established':False,'prediction_gate_passed':False,'uncertainty_calibrated':False,'active_policy_qualified':False,
        'qualification':'Factorized TRAIN engineering gate only. Full zero and diagonal trajectories determine temporal errors; zero subtraction is used solely for the force-induced signal denominator. Failed/missing cells remain visible. Energy algebra is not a conservation test.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--supplemental-manifest');parser.add_argument('--mechanics-directory',default='runs/operator-dev/nonlinear-dynamics-mechanics-v01')
    args=parser.parse_args();result=run(args.manifest,args.output,supplemental_manifest=args.supplemental_manifest,mechanics_directory=args.mechanics_directory)
    print(json.dumps({'cells_verified':sum(c['passed'] for c in result['cells']),'indexed_cells':len(result['cells']),
        'required_cells':result['coverage']['expected_cells'],'comparisons_available':len(result['comparisons']),
        'gate_passed':result['independent_fixed_discretization_dynamic_gate_passed']}))
