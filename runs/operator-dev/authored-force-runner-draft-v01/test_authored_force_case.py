"""Draft runner contracts only. No TRAIN dynamics integration or GPU execution."""
from copy import deepcopy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import campaign_authored_force_case as runner
from experiments import campaign_full_hardware_context as context

PROTOCOL=Path('configs/gaussian_campaign_actual_force_envelope_v01.json')


def protocol():return runner.read(PROTOCOL)


def stage1(tmp_path):
    d=protocol(); rows=[]
    for template in d['initial_cell_plan']:
        if template['stage']=='initial_temporal':
            rows.append({**deepcopy(template),'template_id':template['id'],'output':str(Path('runs')/tmp_path.name/template['id'])})
    return d,{'schema':'actual-authored-force-resolved-stage-v0.1','status':'FROZEN_PUBLISHED_RESOLVED_EXECUTION_MANIFEST','stage':'initial_temporal','cells':rows,'protocol_sha256':runner.sha(PROTOCOL)}


def setup_case(case_id, family='push_pulse', hz=120):
    d=protocol();case,spec,arrays,*_=runner.load_case_inputs(case_id,None,runner.ROOT/d['source_fixture'])
    cell={'response':family,'hz':hz}
    compiled=runner.compile_cell(spec,cell)
    hardware=runner.hardware_descriptor(arrays['rest_points_m'].astype(float),spec)
    built=runner.make_contexts(case,spec,arrays,hardware,compiled,np.zeros(3))
    return case,spec,arrays,hardware,compiled,built


def relocate_root(tmp_path,monkeypatch):
    original=runner.ROOT
    extra=[runner.repo_record(runner.SOURCE),runner.repo_record(runner.PROTOCOL_PATH),
        protocol()['proof_pin_pending_before_launch']['small_force_complete60_audit_path']]
    for relative in list(runner.PINNED)+extra:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original/relative,target)
    source=tmp_path/runner.repo_record(runner.SOURCE)
    protocol_path=tmp_path/runner.repo_record(runner.PROTOCOL_PATH)
    auditor=tmp_path/'experiments/campaign_authored_force_audit.py'
    auditor.write_text('# Synthetic offline seal test; no executable auditor implementation.\n')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'SOURCE',source)
    monkeypatch.setattr(runner,'PROTOCOL_PATH',protocol_path)
    monkeypatch.setattr(runner,'AUDITOR_PATH',auditor)
    return original


def test_frozen_physics_compiler_and_old_runner_unchanged():
    pins=runner.dependencies()
    for path,wanted in runner.PINNED.items():assert runner.sha(runner.ROOT/path)==wanted
    assert runner.repo_record(runner.SOURCE) in pins


def test_initial_stage_requires_all_exact_51_rows_and_fresh_unique_outputs(tmp_path):
    d,m=stage1(tmp_path);assert len(m['cells'])==51
    for row in m['cells']:assert runner.validate_cell(d,m,row['id'])==row
    for key,value in [('hz',960),('relative_tolerance',1e-3),('maximum_iterations',500),('response','zero_external_force'),('translation_m',[1,0,0])]:
        altered=deepcopy(m);altered['cells'][0][key]=value
        with pytest.raises(ValueError):runner.validate_cell(d,altered,altered['cells'][0]['id'])
    with pytest.raises(ValueError):runner.validate_cell(d,{**m,'cells':m['cells'][:-1]},m['cells'][0]['id'])
    altered=deepcopy(m);altered['cells'][1]['output']=altered['cells'][0]['output']
    with pytest.raises(ValueError):runner.validate_cell(d,altered,m['cells'][0]['id'])


def test_draft_has_no_launch_authority_and_creates_no_output(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    seal=tmp_path/'seal.json';seal.write_text(json.dumps({'physical_launch_authorized':False}))
    output=tmp_path/'no-output'
    with pytest.raises(ValueError,match='Draft alone'):runner.run('no-output',prelaunch_seal='seal.json',cell_id='anything')
    assert not output.exists()


def test_matched_rate_is_resolved_only_from_exact_passing_proof(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    d=protocol();template=next(r for r in d['initial_cell_plan'] if r['stage']=='matched_after_temporal')
    proof=tmp_path/'candidate.json';evidence=tmp_path/'evidence.json';evidence.write_text('{}')
    metrics={'adjacent_coordinate_rmse_m':[1e-6,5e-7],'passed':True}
    candidate={'passed':True,'physics_hz':480,'output_hz':20,'frames':61,'geometry_residual_checks_passed':True,
        'all_actual_rates_hz':[120,240,480],'force_induced_signal_m':1e-4,'tolerance_m':5e-6,
        'full_driven_temporal_metrics':metrics,'full_zero_temporal_metrics':metrics,
        'all_adjacent_errors_m':{'driven':[1e-6,5e-7],'zero':[1e-6,5e-7]},
        'source_evidence_records':[{'path':evidence.name,'sha256':runner.sha(evidence),'role':'synthetic contract-test evidence only'}]}
    proof.write_text(json.dumps({'protocol_sha256':runner.sha(PROTOCOL),'accepted_candidates':{template['scenario_id']:candidate}}))
    row={**deepcopy(template),'template_id':template['id'],'id':template['id']+'__hz480','hz':480,'output':'result',
        'temporal_candidate_proof':{'path':proof.name,'sha256':runner.sha(proof)}}
    manifest={'schema':'actual-authored-force-resolved-stage-v0.1','status':'FROZEN_PUBLISHED_RESOLVED_EXECUTION_MANIFEST','stage':'matched_after_temporal','protocol_sha256':runner.sha(PROTOCOL),'cells':[row]}
    assert runner.validate_cell(d,manifest,row['id'])['hz']==480
    changed=deepcopy(manifest);changed['cells'][0]['hz']=960;changed['cells'][0]['id']=template['id']+'__hz960'
    with pytest.raises(ValueError):runner.validate_cell(d,changed,changed['cells'][0]['id'])
    proof.write_text('{}')
    with pytest.raises(ValueError,match='mismatch'):runner.validate_cell(d,manifest,row['id'])


@pytest.mark.parametrize('case_id',['t0000','t0001','t0002','t0003'])
def test_context_separates_known_hardware_force_and_truth_with_fixed_support_capacity(case_id,tmp_path):
    case,spec,arrays,hardware,compiled,built=setup_case(case_id)
    known,query,boundary,public,conditioned=built
    runner.validate_supported_context(known,query,boundary,spec)
    assert query.support_count==2 and len(query.force_programs)==1 and query.handle_count==2
    assert not boundary.masks[1].any()
    assert set(public['feature_groups'])=={'public_known_hardware','public_known_commands','public_known_force_programs'}
    assert all('latent_' not in k for k in public['feature_groups'])
    assert 'latent_material_hypothesis' in conditioned['feature_groups']
    assert 'latent_boundary_hypothesis' in conditioned['feature_groups']
    assert 'latent_support_law_hypothesis' in conditioned['feature_groups']
    assert conditioned['hypothesis_role']=='known_parameter_training_or_diagnostic'
    runner.save_context(tmp_path,public,conditioned)
    public_manifest=runner.read(tmp_path/'engineering-public/manifest.json')
    assert not {'recipe_sha256','source_spec_canonical_sha256','case','boundary_type','material'} & public_manifest.keys()
    assert runner.ENGINEERING_ONLY==public_manifest['admission']
    source_records=public['provenance']['known_hardware']['caller_declared_source_sha256_not_opened_or_verified']
    assert source_records=={}
    np.testing.assert_array_equal(context.combined_forward_forcing(query)['features']['held_handle_forces_world_N'],compiled['held_handle_forces_world_N'])


def test_unsupported_com_rotation_axis_and_external_force_are_rejected():
    _,spec,arrays,hardware,compiled,built=setup_case('t0001')
    known,query,boundary,*_=built
    h=known.handles[0]
    for changed in [{'com_offset_body_m':np.array([.001,0,0])}, {'initial_rotation_body_to_world':np.diag([-1,-1,1])}]:
        clone=context.KnownHandle(**{key:getattr(h,key) for key in h.__dataclass_fields__}|changed)
        changed_known=deepcopy(known);changed_known.handles=(clone,*known.handles[1:])
        with pytest.raises(ValueError):runner.validate_supported_context(changed_known,query,boundary,spec)
    masks=boundary.masks.copy();masks[0,spec['fixture_nodes'][0],2]=False
    wrong=context.BoundaryHypothesis(constraint_axis_masks=masks,support_origins_relative_m=boundary.origins)
    with pytest.raises(ValueError):runner.validate_supported_context(known,query,wrong,spec)
    program=query.force_programs[0];program.torques=np.ones_like(program.torques)
    with pytest.raises(ValueError):runner.validate_supported_context(known,query,boundary,spec)


def test_zero_control_and_all_families_use_exact_source_float32_promoted_commands():
    _,spec,_,_,_,_=setup_case('t0000')
    for family in ['push_pulse','pull_hold','shear_oscillation','twist_couple']:
        compiled=runner.compile_cell(spec,{'response':family,'hz':120})
        direct=np.stack([runner.compiler.forces_at(dict(spec,force_type=family),t) for t in compiled['force_midpoints_s']]).astype(np.float64)
        np.testing.assert_array_equal(direct,compiled['held_handle_forces_world_N'])
        assert len(compiled['requested_output_times_s'])==61
        if family=='twist_couple':assert not compiled['held_handle_forces_world_N'].sum(1).any()
    zero=runner.compile_cell(spec,{'response':'zero_external_force','hz':120})
    assert not zero['held_handle_forces_world_N'].any()
    assert not np.array(zero['metadata']['continuous_unquantized_impulse_Ns']).any()


def test_no_physics_step_when_initialization_fails_and_terminal_failure_is_preserved(tmp_path,monkeypatch):
    relocate_root(tmp_path,monkeypatch)
    d,m=stage1(tmp_path);cell=m['cells'][0];output=runner.resolve_repo_path(cell['output'])
    monkeypatch.setattr(runner,'validate_prelaunch',lambda *args:(d,cell,{}))
    class Broken:
        def __init__(self,*args,**kwargs):raise RuntimeError('intentional initializer failure')
    monkeypatch.setattr(runner,'NonlinearDynamics',Broken)
    with pytest.raises(RuntimeError,match='intentional'):runner.run(cell['output'],prelaunch_seal='unused',cell_id=cell['id'])
    execution=runner.read(output/'execution.json');failure=runner.read(output/'failure.json')
    assert execution['terminal'] and not execution['completed'] and execution['completed_steps']==0
    assert failure['message']=='intentional initializer failure' and failure['step_request'] is None
    assert not (output/'trajectory.npz').exists()


def test_candidate_pass_boolean_without_full_evidence_is_rejected():
    with pytest.raises((ValueError,KeyError)):
        runner.validate_candidate(protocol(),{'accepted_candidates':{'t0001__push_pulse':{'passed':True,'physics_hz':480}}},'t0001__push_pulse',480)


def test_traceback_copies_only_bounded_detached_named_failure_values(tmp_path):
    # Synthetic code uses frozen filename solely to exercise traceback extraction;
    # it performs no solver evaluation and never edits or executes the core file.
    namespace={'torch':torch}
    code=compile("def step():\n q=torch.tensor([1.],requires_grad=True)\n history=[{'iteration':i,'minimum_J':1.} for i in range(300)]\n unrelated_secret='must not serialize'\n raise RuntimeError('control failure')\n",str(runner.ROOT/'experiments/campaign_nonlinear_dynamics.py'),'exec')
    exec(code,namespace)
    try:namespace['step']()
    except RuntimeError as error:proof=runner.capture_traceback_failure(error,tmp_path)
    assert len(proof['nonlinear_history'])==251 and 'UNACCEPTED' in proof['semantics']
    with np.load(tmp_path/'unaccepted-core-locals.npz') as raw:
        assert raw.files==['q'];np.testing.assert_array_equal(raw['q'],[1.])
    assert 'secret' not in json.dumps(proof)
    assert runner.failure_safe({'bad':float('nan')})=={'bad':'nan'}


def test_reviewed_seal_is_portable_and_reads_only_relocated_files(tmp_path,monkeypatch):
    original=relocate_root(tmp_path,monkeypatch)
    d,m=stage1(tmp_path);m['source_sha256']=runner.dependencies()
    m['authored_auditor_source_sha256']=runner.sha(runner.AUDITOR_PATH)
    manifest=tmp_path/'manifest.json';runner.write(manifest,m)
    resources=tmp_path/'resources.json';runner.write(resources,{'threads_per_worker':2,'max_parallel_workers':8,'gpu_used':False})
    prior=runner.ROOT/d['proof_pin_pending_before_launch']['small_force_complete60_audit_path']
    rec=lambda path:{'path':runner.repo_record(path),'sha256':runner.sha(path)}
    seal={'schema':'actual-authored-force-prelaunch-seal-v0.1','status':'REVIEWED_PUBLISHED_EXECUTION_SEAL',
        'physical_launch_authorized':True,'sources_and_stage_manifest_published':True,
        'protocol':rec(runner.PROTOCOL_PATH),'resolved_manifest':rec(manifest),'source_sha256':m['source_sha256'],
        'authored_auditor_source':rec(runner.AUDITOR_PATH),'small_force_complete60_audit':rec(prior),'resource_declaration':rec(resources)}
    seal_path=tmp_path/'validation-only-seal.json';runner.write(seal_path,seal)
    original_sha,original_read=runner.sha,runner.read
    def local_sha(path):
        assert Path(path).resolve().is_relative_to(tmp_path),path
        return original_sha(path)
    def local_read(path):
        assert Path(path).resolve().is_relative_to(tmp_path),path
        return original_read(path)
    monkeypatch.setattr(runner,'sha',local_sha);monkeypatch.setattr(runner,'read',local_read)
    protocol_read,manifest_read,cell,pins=runner.validate_seal(seal_path.name,m['cells'][0]['id'])
    assert len(manifest_read['cells'])==51 and protocol_read['frames']==61
    assert all(not Path(path).is_absolute() for path in pins)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:monkeypatch.setenv(name,'2')
    output=runner.resolve_repo_path(cell['output']);assert not output.exists()
    runner.validate_prelaunch(seal_path.name,cell['id'],cell['output']);assert not output.exists()
    with pytest.raises(ValueError,match='fresh'):runner.validate_prelaunch(seal_path.name,cell['id'],'different-output')
    # Changing a relocated source fails; the intact original cannot rescue it.
    local_source=tmp_path/'experiments/campaign_linear_fem.py';local_source.write_text('altered local source')
    assert (original/'experiments/campaign_linear_fem.py').read_text()!='altered local source'
    with pytest.raises(ValueError,match='dependency'):runner.validate_seal(seal_path.name,cell['id'])


def test_repo_paths_reject_absolute_parent_noncanonical_and_symlink_escapes(tmp_path):
    for path in ['/tmp/x','../x','nested/../../x','./nested','nested//x']:
        with pytest.raises(ValueError):runner.resolve_repo_path(path,repo_root=tmp_path)
    (tmp_path/'escape').symlink_to(tmp_path.parent,target_is_directory=True)
    with pytest.raises(ValueError):runner.resolve_repo_path('escape/x',repo_root=tmp_path)
    assert runner.resolve_repo_path('safe/a',repo_root=tmp_path)==tmp_path/'safe/a'


def test_shared_zero_candidate_is_bound_to_passing_moving_families_and_rate(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    evidence=tmp_path/'proof-evidence.json';evidence.write_text('{}')
    records=[{'path':evidence.name,'sha256':runner.sha(evidence),'role':'synthetic proof validator test'}]
    metrics={'adjacent_coordinate_rmse_m':[1e-6,5e-7],'passed':True}
    candidate={'passed':True,'physics_hz':960,'output_hz':20,'frames':61,'geometry_residual_checks_passed':True,
        'all_actual_rates_hz':[240,480,960],'force_induced_signal_m':1e-4,'tolerance_m':5e-6,
        'full_driven_temporal_metrics':metrics,'full_zero_temporal_metrics':metrics,
        'all_adjacent_errors_m':{'driven':[1e-6,5e-7],'zero':[1e-6,5e-7]},'source_evidence_records':records}
    zero={'passed':True,'physics_hz':960,'output_hz':20,'frames':61,'geometry_residual_checks_passed':True,
        'passing_family_ids':['t0003__push_pulse'],'source_evidence_records':records}
    proof={'accepted_candidates':{'t0003__push_pulse':candidate},'accepted_zero_candidates':{'t0003__zero_external_force':{'960':zero}}}
    assert runner.validate_candidate(protocol(),proof,'t0003__zero_external_force',960)==zero
    with pytest.raises(ValueError):runner.validate_candidate(protocol(),proof,'t0003__zero_external_force',480)
    proof['accepted_candidates']['t0003__push_pulse']['full_zero_temporal_metrics']={'adjacent_coordinate_rmse_m':[1e-4,2e-4],'passed':True}
    with pytest.raises(ValueError):runner.validate_candidate(protocol(),proof,'t0003__zero_external_force',960)


def test_endpoint_guard_enforces_discrete_J_residual_and_full_pose_constraints():
    class Model:
        def validate_state(self,state):pass
    metrics={'minimum_volume_ratio':1.,'fixture_max_error_m':0.,'maximum_rigid_pose_map_error_m':0.,'maximum_rotation_SO3_error':0.}
    row={'converged':True,'residual_tolerance_N':1e-10,'nonlinear_history':[{'minimum_J':1.,'scaled_generalized_residual_norm_N':1e-12}]}
    runner.endpoint_guard(Model(),None,metrics,row)
    for key,value in [('minimum_volume_ratio',.099),('fixture_max_error_m',2e-12),('maximum_rigid_pose_map_error_m',2e-12),('maximum_rotation_SO3_error',2e-12)]:
        with pytest.raises(ValueError):runner.endpoint_guard(Model(),None,{**metrics,key:value},row)
    failed=deepcopy(row);failed['nonlinear_history'][0]['scaled_generalized_residual_norm_N']=2e-10
    with pytest.raises(ValueError):runner.endpoint_guard(Model(),None,metrics,failed)


def test_failed_first_step_preserves_exact_request_last_state_and_partial_frames(tmp_path,monkeypatch):
    from types import SimpleNamespace
    relocate_root(tmp_path,monkeypatch)
    d,m=stage1(tmp_path);cell=m['cells'][0];output=runner.resolve_repo_path(cell['output'])
    monkeypatch.setattr(runner,'validate_prelaunch',lambda *args:(d,cell,{}))
    class FailedStep:
        def __init__(self,points,*args,**kwargs):self.points=torch.as_tensor(points)
        def initial_state(self):
            return SimpleNamespace(time_s=0.,positions=self.points.clone(),velocities=torch.zeros_like(self.points),
                auxiliary_positions=torch.zeros((12,3)),auxiliary_velocities=torch.zeros((12,3)),
                handle_centers=torch.zeros((2,3)),handle_rotations=torch.eye(3).repeat(2,1,1),reduced_position_coordinates=None)
        def coordinates(self,state):return torch.zeros(12)
        def step(self,*args,**kwargs):raise RuntimeError('intentional first-step failure')
    monkeypatch.setattr(runner,'NonlinearDynamics',FailedStep)
    monkeypatch.setattr(runner,'endpoint_metrics',lambda *args:{'minimum_volume_ratio':1.})
    monkeypatch.setattr(runner,'endpoint_guard',lambda *args:None)
    with pytest.raises(RuntimeError,match='first-step'):runner.run(cell['output'],prelaunch_seal='unused',cell_id=cell['id'])
    failure=runner.read(output/'failure.json');execution=runner.read(output/'execution.json')
    assert execution['terminal'] and not execution['completed'] and execution['completed_steps']==0
    assert failure['step_request']['step_index']==0 and failure['step_request']['actual_state_start_s']==0.
    assert failure['step_request']['compiled_interval_s']==[0.,1./cell['hz']]
    with np.load(output/'last-accepted-state.npz') as raw:assert raw['time_s']==0.
    with np.load(output/'partial-trajectory.npz') as raw:np.testing.assert_array_equal(raw['actual_state_times_s'],[0.])
    assert not (output/'trajectory.npz').exists()
