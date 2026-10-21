"""Offline evidence tests: analytic rest fixture, no physical integration/job.

The fixture exercises the cell verifier, not an admitted stage/physical capture.
Strict seal/51-row tests remain independent runner tests. Tampering rehashes the
outer container so rejection must come from mechanics/schema/provenance checks.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from experiments import campaign_authored_force_audit as audit
from experiments import campaign_authored_force_case as runner


def write(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value,allow_nan=False)+'\n')


def load(file):return json.loads(file.read_text())


def save_npz(file,arrays):np.savez_compressed(file,**arrays)


def build_rest_artifact(root):
    """Hand-authored exact static-zero ledger; does not invoke model.step."""
    canonical=load(runner.ROOT/audit.FIXTURE_PATH)
    selected=next(c for c in canonical['cases'] if c['episode_id']=='t0001')
    recipe,spec=selected['recipe'],selected['boundaries']
    points,tets,faces,generated=audit.compiler.forces_at.__globals__['geometry'](recipe)
    assert generated==spec
    arrays={'rest_points_m':points,'tetrahedra':tets,'surface_faces':faces}
    cell={'id':'offline_analytic_rest_NOT_CAPTURE','case':'t0001','response':'zero_external_force','hz':120,
        'output_hz':20,'relative_tolerance':1e-7,'absolute_tolerance_N':1e-10,'translation_m':[0,0,0],
        'maximum_iterations':250,'output':'runs/offline-analytic-rest'}
    folder=root/cell['output'];folder.mkdir(parents=True)
    # Copies let snapshot tampering be tested without modifying real source.
    pins={audit.FIXTURE_PATH:audit.sha(runner.ROOT/audit.FIXTURE_PATH)}
    target=root/audit.FIXTURE_PATH;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(runner.ROOT/audit.FIXTURE_PATH,target)
    private=folder/'private-provenance';private.mkdir()
    name='00-'+Path(audit.FIXTURE_PATH).name;shutil.copyfile(target,private/name)
    write(private/'pins.json',pins);write(private/'snapshots.json',[{'original_path':audit.FIXTURE_PATH,'saved_path':name,'sha256':pins[audit.FIXTURE_PATH]}])
    save_npz(private/'static-reference.npz',arrays)
    hardware=runner.hardware_descriptor(points.astype(float),spec)
    write(private/'physical-inputs.json',{'case':recipe,'spec':spec,'hardware':hardware,'static_lineage':{'role':'offline exact rest test'}})
    compiled=audit.compiled_expected(spec,cell)
    built=runner.make_contexts(recipe,spec,arrays,hardware,compiled,np.zeros(3))
    runner.save_context(folder,built[-2],built[-1])
    save_npz(folder/'compiled-commands.npz',{k:v for k,v in compiled.items() if isinstance(v,np.ndarray)})
    write(folder/'protocol.json',{'cell':cell,'source_input_sha256':pins,'seconds':3.,'output_hz':20,'frames':61,
        'admission':runner.ENGINEERING_ONLY,'native_equivalence':False,'continuum_accuracy_established':False,
        'native_elasticity_damping_reproduced':False,'compiled_force_metadata':compiled['metadata']})
    centers=points[spec['load_nodes']].astype(float);aux=[]
    for h in hardware:
        offsets,_=audit.auxiliary_mass_points(h['mass_kg'],h['full_COM_inertia_kg_m2']);aux.extend(np.asarray(h['reference_origin_m'])+offsets)
    n=360;intervals=compiled['force_interval_times_s'];F=compiled['held_handle_forces_world_N'];zero3=[0.,0.,0.]
    invariants={'energy_J':0.,'elastic_energy_J':0.,'chord_kinetic_energy_J':0.,'point_linear_momentum_Ns':zero3}
    rows=[]
    for i in range(n):
        rows.append({'step_index':i,'converged':True,'compiled_interval_s':intervals[i].tolist(),
            'compiled_midpoint_s':float(compiled['force_midpoints_s'][i]),'requested_dt_s':1/120,'step_s':1/120,
            'actual_interval_s':intervals[i].tolist(),'start_time_s':float(intervals[i,0]),'end_time_s':float(intervals[i,1]),
            'actual_state_start_s':float(intervals[i,0]),'actual_state_midpoint_s':float(intervals[i].mean()),
            'held_handle_forces_world_N':F[i].tolist(),'held_handle_forces_N':F[i].tolist(),'residual_tolerance_N':1e-10,
            'nonlinear_history':[{'iteration':0,'minimum_J':1.,'scaled_generalized_residual_norm_N':0.}],
            'minimum_J':1.,'geometry':{'minimum_volume_ratio':1.,'fixture_max_error_m':0.,'maximum_rigid_pose_map_error_m':0.,'maximum_rotation_SO3_error':0.},
            'handle_centers_start_m':centers.tolist(),'handle_centers_end_m':centers.tolist(),'handle_center_step_increment_m':np.zeros((2,3)).tolist(),
            'held_handle_work_each_J':[0.,0.],'held_handle_work_J':0.,'moment_about_world_origin_start_end_Nm':np.zeros((2,2,3)).tolist(),
            'fixture_reactions_inertia_elastic_drag_minus_external_N':np.zeros((len(spec['fixture_nodes']),3)).tolist(),
            'support_work_J':0.,'held_nodal_work_J':0.,'old_invariants':invariants,'new_invariants':invariants,
            'body_viscous_dissipation_J':0.,'constraint_chord_work_J':0.,'implicit_velocity_increment_loss_J':0.,
            'backward_elastic_gradient_defect_J':0.,'energy_algebra_identity_residual_J':0.,'body_drag_impulse_Ns':zero3,'linear_momentum_balance_error_Ns':zero3})
    (folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    trajectory={'times_s':compiled['requested_output_times_s'],'actual_state_times_s':compiled['requested_output_times_s'],
        'positions_m':np.broadcast_to(points,(61,*points.shape)).copy(),'velocities_point_chord_m_s':np.zeros((61,*points.shape)),
        'auxiliary_positions_m':np.broadcast_to(aux,(61,12,3)).copy(),'auxiliary_velocities_point_chord_m_s':np.zeros((61,12,3)),
        'handle_centers_m':np.broadcast_to(centers,(61,2,3)).copy(),'handle_rotations':np.broadcast_to(np.eye(3),(61,2,3,3)).copy(),
        'force_interval_actual_times_s':intervals,'force_midpoints_compiled_s':compiled['force_midpoints_s'],'force_step_commands_N':F,
        'handle_work_each_J':np.zeros((n,2)),'moment_about_world_origin_start_end_Nm':np.zeros((n,2,2,3))}
    save_npz(folder/'trajectory.npz',trajectory)
    write(folder/'summary.json',{'completed':True,'frames':61,'cell_id':cell['id'],'admission':runner.ENGINEERING_ONLY,
        'actual_force_envelope_qualified':False,'native_reference_qualified':False,'continuum_accuracy_established':False,'prediction_qualified':False,
        'trajectory_sha256':audit.sha(folder/'trajectory.npz'),'steps_sha256':audit.sha(folder/'steps.jsonl'),
        'actual_clock_impulse_Ns':np.zeros((2,3)).tolist(),'actual_clock_absolute_impulse_Ns':np.zeros((2,3)).tolist(),
        'actual_clock_first_time_moment_Ns2':np.zeros((2,3)).tolist(),'handle_work_each_total_J':[0.,0.]})
    write(folder/'execution.json',{'terminal':True,'completed':True,'completed_steps':n,'planned_steps':n,'cpu_threads':2,'gpu_used':False})
    return cell,pins,canonical


@pytest.fixture
def artifact(tmp_path,monkeypatch):
    fixture=build_rest_artifact(tmp_path);monkeypatch.setattr(runner,'ROOT',tmp_path)
    return tmp_path,fixture


def refresh_hashes(folder):
    summary=load(folder/'summary.json')
    summary.update(trajectory_sha256=audit.sha(folder/'trajectory.npz'),steps_sha256=audit.sha(folder/'steps.jsonl'))
    write(folder/'summary.json',summary)


def test_exact_rest_evidence_verifies_but_is_not_a_stage_or_capture(artifact):
    root,(cell,pins,canonical)=artifact
    report,_=audit.verify_cell(cell,pins,canonical)
    assert report['passed'],report
    assert report['energy_conservation_claim'] is False
    assert report['contact_model']=='not_modeled'
    assert report['absolute_floor_dominated_steps']==360
    assert len(report['source_evidence_records'])>=12


@pytest.mark.parametrize('field,value,message',[
    ('requested_dt_s',1/240,'clock'),('residual_tolerance_N',1e-8,'threshold'),
    ('minimum_J',.09,'0.1'),('support_work_J',1e-5,'fixture work'),
    ('held_handle_work_each_J',[1e-5,0.],'force work'),
    ('moment_about_world_origin_start_end_Nm',np.ones((2,2,3)).tolist(),'moment'),
    ('handle_center_step_increment_m',np.ones((2,3)).tolist(),'increments'),
    ('held_handle_forces_N',np.ones((2,3)).tolist(),'held force'),
    ('body_drag_impulse_Ns',[1e-4,0.,0.],'Momentum')])
def test_rehashed_step_tamper_is_rejected(artifact,field,value,message):
    root,(cell,pins,canonical)=artifact;folder=root/cell['output'];rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    rows[12][field]=value;(folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));refresh_hashes(folder)
    report,_=audit.verify_cell(cell,pins,canonical)
    assert not report['passed'] and message.lower() in report['error'].lower(),report


@pytest.mark.parametrize('kind',['history_residual','history_J','history_cap','fixture','rigid_patch','saved_J','clock','context','hardware','snapshot','command'])
def test_structural_and_physical_corruption_is_rejected(artifact,kind):
    root,(cell,pins,canonical)=artifact;folder=root/cell['output']
    if kind.startswith('history'):
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        if kind=='history_residual':rows[1]['nonlinear_history'][-1]['scaled_generalized_residual_norm_N']=1e-6
        if kind=='history_J':rows[1]['nonlinear_history'][-1]['minimum_J']=0.
        if kind=='history_cap':rows[1]['nonlinear_history']*=252
        (folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif kind in ['fixture','rigid_patch','saved_J','clock']:
        values=audit.archive(folder/'trajectory.npz');spec=canonical['cases'][1]['boundaries']
        if kind=='clock':values['actual_state_times_s'][3]+=1e-5
        elif kind=='fixture':values['positions_m'][3,spec['fixture_nodes'][0],0]+=1e-5
        elif kind=='rigid_patch':values['positions_m'][3,spec['load_patches'][0][-1],0]+=1e-5
        else:values['positions_m'][3]*=-1
        save_npz(folder/'trajectory.npz',values)
    elif kind=='context':
        group=folder/'engineering-public';values=audit.archive(group/'features.npz');values['latent_material_hypothesis/truth']=np.array([5.]);save_npz(group/'features.npz',values)
        m=load(group/'manifest.json');m['feature_archive_sha256']=audit.sha(group/'features.npz');write(group/'manifest.json',m)
    elif kind=='hardware':
        file=folder/'private-provenance/physical-inputs.json';value=load(file);value['hardware'][0]['full_COM_inertia_kg_m2'][0][1]=1e-8;write(file,value)
    elif kind=='snapshot':(folder/'private-provenance'/('00-'+Path(audit.FIXTURE_PATH).name)).write_text('{}')
    else:
        values=audit.archive(folder/'compiled-commands.npz');values['held_handle_forces_world_N'][1,0,0]=1e-4;save_npz(folder/'compiled-commands.npz',values)
    refresh_hashes(folder);report,_=audit.verify_cell(cell,pins,canonical)
    assert not report['passed'],(kind,report)


def test_failed_and_incomplete_evidence_stays_unqualified(artifact):
    root,(cell,pins,canonical)=artifact;folder=root/cell['output']
    write(folder/'failure.json',{'message':'preserved deliberate failure'})
    report,_=audit.verify_cell(cell,pins,canonical);assert report['state']=='failed_preserved' and not report['passed']
    (folder/'failure.json').unlink();status=load(folder/'execution.json');status['completed']=False;write(folder/'execution.json',status)
    report,_=audit.verify_cell(cell,pins,canonical);assert report['state']=='incomplete'


def test_temporal_uses_full_response_not_zero_subtracted_errors():
    q=np.zeros((61,4,3));common=[q+30e-6,q+15e-6,q+8e-6];driven=[x+1e-6 for x in common]
    assert all(np.allclose(d-z,1e-6) for d,z in zip(driven,common))
    assert not audit.temporal_metric(driven,2e-6)['passed']
    assert audit.temporal_metric([q+2e-6,q+1e-6,q+.5e-6],2e-6)['passed']
    assert not audit.temporal_metric([q,q+.5e-6,q+3e-6],2e-6)['passed']


def test_replay_checks_velocities_rotations_and_commands_not_only_q():
    a={key:np.zeros((61,2,3)) for key in audit.REPLAY_KEYS};b=deepcopy(a)
    assert audit.replay_metric(a,b)['passed']
    for key in audit.REPLAY_KEYS:
        b=deepcopy(a);b[key].flat[-1]=2e-12
        assert not audit.replay_metric(a,b)['passed'],key


def test_static_zero_is_analytic_and_moving_zero_is_never_fabricated():
    canonical=load(runner.ROOT/audit.FIXTURE_PATH)
    rest=audit.analytic_static_zero(canonical,'t0000');assert rest.shape[0]==61
    with pytest.raises(ValueError,match='freshly'):audit.analytic_static_zero(canonical,'t0003')
    declaration=audit.analytic_declaration(load(runner.PROTOCOL_PATH))
    assert len(declaration['grids'])==9 and declaration['frames_upsampled_from_prior'] is False
    assert 'NOT_CAPTURE_NOT_RESET' in declaration['kind']


def test_public_api_invalid_seal_fails_without_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    write(tmp_path/'seal.json',{'status':'DRAFT'})
    before=set(tmp_path.rglob('*'));report=audit.audit_cell('seal.json','arbitrary')
    assert report['passed'] is False and report['state']=='invalid_seal' and set(tmp_path.rglob('*'))==before


@pytest.mark.parametrize('bad',['/etc/passwd','../outside','runs/../outside','runs//item','./runs/item'])
def test_absolute_or_noncanonical_records_rejected(tmp_path,monkeypatch,bad):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    with pytest.raises(ValueError):audit.path(bad)


def test_fresh_process_relocated_root_reads_only_relocated_evidence(artifact,tmp_path):
    root,(cell,pins,canonical)=artifact
    # Copy only source and explicit fixture, then run a new interpreter from the
    # restored root. The deny-read hook forbids original workspace files.
    original=Path(__file__).resolve().parents[1]
    for relative in set(runner.PINNED)|{'experiments/campaign_authored_force_case.py',audit.AUDITOR_PATH}:
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/relative,target)
    write(root/'offline-test-request.json',{'cell':cell,'pins':pins})
    script="""
import builtins,json,pathlib,sys
original=pathlib.Path(sys.argv[1]).resolve(); restored=pathlib.Path.cwd().resolve()
def hook(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  p=pathlib.Path(args[0]).resolve()
  if p.is_relative_to(original) and not p.is_relative_to(original/'.venv') and not p.is_relative_to(restored):
   raise RuntimeError('Forbidden original-root read: '+str(p))
sys.addaudithook(hook)
from experiments import campaign_authored_force_audit as a
request=json.load(open('offline-test-request.json'));canonical=json.load(open(a.FIXTURE_PATH))
report,_=a.verify_cell(request['cell'],request['pins'],canonical)
assert report['passed'],report
print('relocated_only_verified')
"""
    result=subprocess.run([sys.executable,'-c',script,str(original)],cwd=root,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr+result.stdout
    assert 'relocated_only_verified' in result.stdout


def test_stage_predecessor_uses_its_own_seal_and_manifest_pin_map(tmp_path,monkeypatch):
    """Regression: source bytes match while stage-specific provenance differs."""
    original=runner.ROOT;protocol=load(runner.PROTOCOL_PATH)
    for relative in [audit.FIXTURE_PATH,audit.AUDITOR_PATH,'experiments/campaign_authored_commands.py']:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/relative,target)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    source={'physical.py':'same-physical-sha'};auditor_sha=audit.sha(tmp_path/audit.AUDITOR_PATH)
    pins1={**source,audit.AUDITOR_PATH:auditor_sha,'stage1-seal.json':'stage1hash','stage1-manifest.json':'firsthash'}
    pins2={**source,audit.AUDITOR_PATH:auditor_sha,'stage2-seal.json':'stage2hash','stage2-manifest.json':'secondhash'}
    write(tmp_path/'stage1-manifest.json',{'stage':'initial_temporal','cells':[{'id':'first','output':'runs/first'}]})
    write(tmp_path/'stage1-seal.json',{'resolved_manifest':{'path':'stage1-manifest.json','sha256':audit.sha(tmp_path/'stage1-manifest.json')}})
    write(tmp_path/'proof.json',{'auditor_source_sha256':auditor_sha,'temporal_stage_seals':[
        {'path':'stage1-seal.json','sha256':audit.sha(tmp_path/'stage1-seal.json')}]})
    write(tmp_path/'stage2-manifest.json',{'stage':'matched_after_temporal','cells':[{'id':'second','output':'runs/second',
        'temporal_candidate_proof':{'path':'proof.json','sha256':audit.sha(tmp_path/'proof.json')}}]})
    write(tmp_path/'stage2-seal.json',{'resolved_manifest':{'path':'stage2-manifest.json','sha256':audit.sha(tmp_path/'stage2-manifest.json')}})
    write(tmp_path/'analytic.json',audit.analytic_declaration(protocol));seen={}
    def validate(seal,id):
        manifest=load(tmp_path/('stage1-manifest.json' if seal=='stage1-seal.json' else 'stage2-manifest.json'))
        return protocol,manifest,manifest['cells'][0],pins1 if seal=='stage1-seal.json' else pins2
    def verify(row,pins,canonical):
        seen[row['id']]=pins
        assert pins==(pins1 if row['id']=='first' else pins2)
        return {'cell':row,'id':row['id'],'passed':False,'state':'missing'},None
    monkeypatch.setattr(runner,'validate_seal',validate);monkeypatch.setattr(runner,'dependencies',lambda:source)
    monkeypatch.setattr(audit,'verify_cell',verify)
    report=audit.audit_stage('stage2-seal.json',analytic_reference_record=audit.evidence('analytic.json','analytic'))
    assert seen=={'first':pins1,'second':pins2}
    assert report['actual_force_declared_discrete_engineering_envelope_qualified'] is False
    assert report['temporal_stage_seals'][0]['path']=='stage1-seal.json'


def test_matched_controls_require_every_driven_family_and_moving_zero():
    protocol=load(runner.PROTOCOL_PATH);q=np.zeros((61,4,3));candidates={}
    zeros={'t0003__zero_external_force':{'960':{'passing_family_ids':['t0003__push_pulse']}}}
    for scenario in protocol['scenarios']:
        candidates[scenario['id']]={'passed':True,'physics_hz':scenario['initial_rates_hz'][-1],'tolerance_m':2e-6}
    data={};reports=[]
    for scenario,entry in candidates.items():
        id=scenario+'__temporal__hz'+str(entry['physics_hz']);data[id]={key:q.copy() for key in audit.REPLAY_KEYS}
        reports.append({'cell':{'id':id},'id':id,'absolute_floor_dominated_steps':0})
    baseline='t0003__zero__temporal__hz960';data[baseline]={key:q.copy() for key in audit.REPLAY_KEYS}
    reports.append({'cell':{'id':baseline},'id':baseline,'absolute_floor_dominated_steps':2880})
    for template in protocol['initial_cell_plan']:
        if template['stage']!='matched_after_temporal':continue
        hz=960 if template['case']=='t0003' else 480;id=template['id']+'__hz'+str(hz)
        arrays={key:q.copy() for key in audit.REPLAY_KEYS}
        if template['kind']=='translated':arrays['positions_m']+=np.asarray(template['translation_m'])
        data[id]=arrays;reports.append({'cell':{**template,'id':id,'hz':hz},'id':id,'absolute_floor_dominated_steps':1})
    comparisons,required=audit.matched_comparisons(protocol,candidates,zeros,reports,data)
    assert len(required)==68 and len(comparisons)==51 and all(c['passed'] for c in comparisons)
    # Missing only one family's replay must remain visible as a failed slot.
    missing=next(id for id in required if 'fresh_reset' in id and 'shear' in id);data.pop(missing)
    comparisons,_=audit.matched_comparisons(protocol,candidates,zeros,reports,data)
    assert not all(c['passed'] for c in comparisons)
    assert any(missing in c['ids'] and c['state']=='pending' for c in comparisons)


def test_known_parameter_and_public_contexts_for_all_four_layouts(tmp_path):
    # Typed-query comparison only; no response generation.
    for case_id in ['t0000','t0001','t0002','t0003']:
        case,spec,arrays,*_=runner.load_case_inputs(case_id,None,runner.ROOT/audit.FIXTURE_PATH)
        hardware=runner.hardware_descriptor(arrays['rest_points_m'].astype(float),spec)
        cell={'hz':120,'response':'twist_couple'};compiled=audit.compiled_expected(spec,cell)
        known,query,boundary,public,conditioned=runner.make_contexts(case,spec,arrays,hardware,compiled,np.zeros(3))
        folder=tmp_path/case_id;folder.mkdir();runner.save_context(folder,public,conditioned)
        # Temporarily set only path resolution; compilation and saved descriptors
        # above were generated before this override.
        old=runner.ROOT
        try:
            runner.ROOT=tmp_path
            audit.verify_contexts(case_id,case,spec,arrays,hardware,compiled,[0,0,0])
        finally:runner.ROOT=old


def test_moving_origin_couple_is_not_a_fixed_reference_torque():
    force=np.array([[0.,0.,2.],[0.,0.,-2.]])
    start=np.array([[1.,0.,0.],[0.,1.,0.]])
    increment=np.array([[.2,0.,.1],[0.,-.3,-.2]])
    work,moment=audit.held_force_work_and_moments(force,start,start+increment,increment)
    np.testing.assert_allclose(work,[.2,.4],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(force.sum(0),[0,0,0])
    np.testing.assert_allclose(moment[0].sum(0),[-2.,-2.,0.],rtol=0,atol=1e-15)
    np.testing.assert_allclose(moment[1].sum(0),[-1.4,-2.4,0.],rtol=0,atol=1e-15)
    assert not np.array_equal(moment[0],moment[1])
    translation=np.array([.4,-.7,.2])
    shifted_work,shifted_moment=audit.held_force_work_and_moments(force,start+translation,start+increment+translation,increment)
    np.testing.assert_array_equal(shifted_work,work)
    np.testing.assert_allclose(shifted_moment.sum(1),moment.sum(1),rtol=0,atol=1e-15)


def test_candidate_proof_producer_matches_strict_runner_consumer(tmp_path,monkeypatch):
    """Comparison-contract test only; already-verified inputs are synthetic."""
    protocol=load(runner.PROTOCOL_PATH);q=np.zeros((61,4,3));reports=[];data={}
    write(tmp_path/'evidence.json',{'role':'synthetic comparison evidence, not capture'})
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    record=audit.evidence('evidence.json','synthetic_verified_comparison_input')
    monkeypatch.setattr(audit,'analytic_static_zero',lambda canonical,case_id:q.copy())
    for scenario in protocol['scenarios']:
        for i,hz in enumerate(scenario['initial_rates_hz']):
            id=scenario['id']+'__temporal__hz'+str(hz)
            reports.append({'cell':{'id':id},'id':id,'source_evidence_records':[record]})
            data[id]={'positions_m':q+1e-4+(3-i)*1e-6}
    for hz in [240,480,960]:
        id='t0003__zero__temporal__hz'+str(hz)
        reports.append({'cell':{'id':id},'id':id,'source_evidence_records':[record]});data[id]={'positions_m':q.copy()}
    candidates,zeros=audit.temporal_candidates(protocol,reports,data,{},record)
    proof={'accepted_candidates':candidates,'accepted_zero_candidates':zeros}
    assert len(candidates)==16 and all(c['passed'] for c in candidates.values())
    for scenario in protocol['scenarios']:
        hz=scenario['initial_rates_hz'][-1]
        assert runner.validate_candidate(protocol,proof,scenario['id'],hz)['passed']
    assert runner.validate_candidate(protocol,proof,'t0003__zero_external_force',960)['passed']
    # A common moving-zero drift must fail BOTH roles, despite cancellation in
    # force-induced trajectories; do not silently remove prior rate failures.
    for i,hz in enumerate([240,480,960]):
        id='t0003__zero__temporal__hz'+str(hz);data[id]['positions_m']=q+[30e-6,15e-6,8e-6][i]
    failed,_=audit.temporal_candidates(protocol,reports,data,{},record)
    assert not any(c['passed'] for id,c in failed.items() if id.startswith('t0003__'))
    assert failed['t0003__push_pulse']['all_actual_rates_hz']==[240,480,960]


@pytest.mark.parametrize('kind',['float32','runtime_J'])
def test_precision_and_runtime_geometry_must_match_saved_frames(artifact,kind):
    root,(cell,pins,canonical)=artifact;folder=root/cell['output']
    if kind=='float32':
        values=audit.archive(folder/'trajectory.npz');values['velocities_point_chord_m_s']=values['velocities_point_chord_m_s'].astype(np.float32);save_npz(folder/'trajectory.npz',values)
    else:
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        rows[5]['minimum_J']=.5
        (folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    refresh_hashes(folder);report,_=audit.verify_cell(cell,pins,canonical)
    assert not report['passed'] and ('float64' in report['error'] if kind=='float32' else 'minimum J' in report['error'])
