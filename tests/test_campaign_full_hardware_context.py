"""Analytic explicit inputs only: no episodes, observations, training or solve."""
import copy
from pathlib import Path

import numpy as np
import pytest

from experiments import campaign_full_hardware_context as context
from experiments.campaign_rigid_inertia import auxiliary_mass_points


POINTS=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.]])
TETS=np.array([[0,1,2,3],[0,4,3,2]])
FACES=np.array([[0,1,2],[0,1,3],[1,2,3],[0,3,4],[0,2,4],[2,3,4]])
ORIGIN=np.array([.2,-.3,.5])
INERTIA=1e-6*np.array([[4.,1.,0.],[1.,3.,0.],[0.,0.,5.]])
SOURCE={'explicit_literal_test_inputs':'a'*64}


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def handle(*,patch=(0,1,2),shift=(0.,0.,0.),inertia=INERTIA,com=(.01,.02,.03),rotation=None):
    return context.KnownHandle(origin_world_m=np.asarray(shift),ordered_patch_nodes=np.asarray(patch),
        mass_kg=.002,com_inertia_body_kg_m2=inertia,com_offset_body_m=com,
        initial_rotation_body_to_world=rz(.3) if rotation is None else rotation,origin_node_id=0)


def hardware(*,shift=(0.,0.,0.),patch=(0,1,2),handles=None):
    shift=np.asarray(shift)
    return context.KnownHardware(rest_points_world_m=POINTS+shift,tetrahedra=TETS,surface_faces=FACES,
        handles=[handle(patch=patch,shift=shift)] if handles is None else handles,
        coordinate_origin_world_m=ORIGIN+shift,source_sha256=SOURCE)


def query(*,handles=1,supports=1,force_role='known_actuator_command',support_role='candidate_boundary_hypothesis',
          extra_force_programs=(),**changes):
    translations=np.zeros((3,supports,3));rotations=np.broadcast_to(np.eye(3),(3,supports,3,3)).copy()
    if supports:
        translations[1,:,0]=.01;translations[2,:,1]=.02
        rotations[1]=rz(.2);rotations[2]=rz(.4)
    force_values=dict(held_handle_forces_world_N=np.broadcast_to([.01,.02,0.],(2,handles,3)),
        held_handle_torques_world_Nm=np.broadcast_to([0.,0.,.001],(2,handles,3)),
        held_nodal_forces_world_N=np.zeros((2,5,3)),held_body_acceleration_world_m_s2=np.zeros((2,3)),
        information_role=force_role,source_sha256=SOURCE)
    for key in set(changes).intersection(force_values):force_values[key]=changes.pop(key)
    values=dict(boundary_times_s=np.array([0.,.4,1.]),force_programs=[context.ForceProgram(**force_values),*extra_force_programs],
        support_translations_world_m=translations,support_rotations_world=rotations,
        support_law_role=support_role,support_law_source_sha256=SOURCE,
        requested_output_times_s=np.array([0.,.4,1.]),source_sha256=SOURCE)
    values.update(changes)
    return context.HeldCommandQuery(**values)


def material(E=3000.):
    return context.MaterialHypothesis(youngs_modulus_Pa=E,poissons_ratio=.35,density_kg_m3=700.,
        body_linear_drag_s_inverse=.6,constitutive_model='log_J_neo_hookean')


def boundary(node=4):
    mask=np.zeros((1,5,3),dtype=bool);mask[0,node]=True
    return context.BoundaryHypothesis(constraint_axis_masks=mask,support_origins_relative_m=[POINTS[3]-ORIGIN])


def conditioned(known=None,commands=None,mat=None,bc=None):
    return context.conditioned_forward_context(known_hardware=hardware() if known is None else known,
        command_query=query() if commands is None else commands,material_hypothesis=material() if mat is None else mat,
        boundary_hypothesis=boundary() if bc is None else bc,hypothesis_role='candidate_from_prior_or_posterior',
        hypothesis_source_sha256=SOURCE)


def test_different_complete_patches_with_same_center_remain_distinguishable():
    first=hardware(patch=(0,1,2));second=hardware(patch=(0,1,3))
    a,b=first.features(),second.features()
    np.testing.assert_array_equal(a['handle_origins_relative_m'],b['handle_origins_relative_m'])
    assert not np.array_equal(a['patch_node_masks'],b['patch_node_masks'])
    assert not np.array_equal(a['ordered_patch_local_offsets_body_m'],b['ordered_patch_local_offsets_body_m'])
    assert first.provenance()['manifest_sha256']!=second.provenance()['manifest_sha256']
    reordered=hardware(patch=(2,0,1)).features()
    np.testing.assert_array_equal(a['patch_node_masks'],reordered['patch_node_masks'])
    assert not np.array_equal(a['ordered_patch_node_ids'],reordered['ordered_patch_node_ids'])
    R=a['initial_handle_rotations_body_to_world'][0]
    reconstructed=a['handle_origins_relative_m'][0]+a['ordered_patch_local_offsets_body_m'][0]@R.T
    np.testing.assert_allclose(reconstructed,a['rest_points_relative_m'][a['ordered_patch_node_ids'][0]],atol=3e-16,rtol=0)


def test_full_offdiagonal_inertia_and_com_affect_context_and_physical_kinetic_energy():
    h=handle();diagonal=handle(inertia=np.diag(np.diag(INERTIA)));centered=handle(com=(0,0,0))
    v=np.array([.1,-.2,.05]);w=np.array([1.,2.,3.])
    energy=lambda item:context.rigid_kinetic_energy(item,origin_velocity_world_m_s=v,angular_velocity_world_rad_s=w)
    assert abs(energy(h)-energy(diagonal))>1e-8
    assert abs(energy(h)-energy(centered))>1e-8
    offsets,masses=auxiliary_mass_points(h.mass_kg,h.com_inertia_body_kg_m2)
    world=(offsets+h.com_offset_body_m)@h.initial_rotation_body_to_world.T
    expected=.5*np.sum(masses[:,None]*(v+np.cross(w,world))**2)
    np.testing.assert_allclose(energy(h),expected,atol=1e-18,rtol=0)
    features=hardware(handles=[h]).features()
    np.testing.assert_array_equal(features['handle_COM_inertias_body_kg_m2'][0],INERTIA)
    np.testing.assert_array_equal(features['handle_COM_offsets_body_m'][0],h.com_offset_body_m)
    assert not np.array_equal(features['handle_inertias_about_origin_body_kg_m2'][0],INERTIA)


def test_all_model_features_and_derived_support_targets_are_translation_consistent():
    original=hardware();shifted=hardware(shift=(.375,-.625,4.25))
    for key,value in original.features().items():
        np.testing.assert_allclose(value,shifted.features()[key],atol=1e-12,rtol=0,err_msg=key)
    a=conditioned(known=original);b=conditioned(known=shifted)
    for group,fields in a['feature_groups'].items():
        for key,value in fields.items():np.testing.assert_allclose(value,b['feature_groups'][group][key],atol=1e-12,rtol=0)
    targets=lambda known:context.support_targets_relative(known_hardware=known,boundary_hypothesis=boundary(),command_query=query())
    np.testing.assert_allclose(targets(original)['targets_relative_m'],targets(shifted)['targets_relative_m'],atol=1e-12,rtol=0)
    assert original.provenance()['absolute_frame_input_arrays']!=shifted.provenance()['absolute_frame_input_arrays']


def test_public_inference_api_never_accepts_or_includes_material_boundary_or_response_labels():
    known=hardware();commands=query()
    public=context.inference_public_context(known_hardware=known,command_query=commands)
    assert set(public['feature_groups'])=={'public_known_hardware','public_known_commands','public_known_force_programs'}
    for extra in ['material_hypothesis','boundary_hypothesis','future_rgb','response_positions_m','truth']:
        with pytest.raises(TypeError):context.inference_public_context(known_hardware=known,command_query=commands,**{extra:material()})
    low=conditioned(mat=material(2000.));high=conditioned(mat=material(9000.))
    for group in public['feature_groups']:
        for key,value in public['feature_groups'][group].items():
            np.testing.assert_array_equal(value,low['feature_groups'][group][key])
            np.testing.assert_array_equal(value,high['feature_groups'][group][key])
    assert not np.array_equal(low['feature_groups']['latent_material_hypothesis']['homogeneous_material_parameters'],high['feature_groups']['latent_material_hypothesis']['homogeneous_material_parameters'])
    assert low['provenance']['known_hardware']==high['provenance']['known_hardware']
    assert low['provenance']['hypotheses']!=high['provenance']['hypotheses']


def passive_program(value,source):
    return context.ForceProgram(held_handle_forces_world_N=np.full((2,1,3),value),
        held_handle_torques_world_Nm=np.full((2,1,3),2*value),
        held_nodal_forces_world_N=np.full((2,5,3),3*value),held_body_acceleration_world_m_s2=np.full((2,3),4*value),
        information_role='candidate_force_hypothesis',source_sha256={'sampled_passive_program':source*64})


def test_latent_force_and_support_programs_never_enter_public_features_or_provenance():
    first=query(extra_force_programs=[passive_program(.03,'b')])
    translations=first.support_translation.copy();translations[1:]*=3
    second=query(extra_force_programs=[passive_program(-.08,'c')],support_translations_world_m=translations,
        support_law_source_sha256={'sampled_support_program':'d'*64})
    a=context.inference_public_context(known_hardware=hardware(),command_query=first)
    b=context.inference_public_context(known_hardware=hardware(),command_query=second)
    assert a['provenance']==b['provenance']
    for group,fields in a['feature_groups'].items():
        for key,value in fields.items():np.testing.assert_array_equal(value,b['feature_groups'][group][key])
    assert 'public_known_support_program' not in a['feature_groups']
    assert 'latent_force_hypotheses' not in a['feature_groups']
    forward_a=conditioned(commands=first);forward_b=conditioned(commands=second)
    assert not np.array_equal(forward_a['feature_groups']['latent_force_hypotheses']['held_handle_forces_world_N'],
        forward_b['feature_groups']['latent_force_hypotheses']['held_handle_forces_world_N'])
    assert not np.array_equal(forward_a['feature_groups']['latent_support_law_hypothesis']['prescribed_support_endpoint_translations_world_m'],
        forward_b['feature_groups']['latent_support_law_hypothesis']['prescribed_support_endpoint_translations_world_m'])
    # The known probe stays available even with an unknown additive history.
    np.testing.assert_array_equal(a['feature_groups']['public_known_force_programs']['held_handle_forces_world_N'][0],first.force_programs[0].forces)
    combined=context.combined_forward_forcing(first)
    np.testing.assert_allclose(combined['features']['held_handle_forces_world_N'],first.force_programs[0].forces+.03,atol=1e-17,rtol=0)
    assert 'not_public' in combined['role']


def test_unknown_passive_forcing_can_be_completely_absent_from_public_input_while_known_actuators_are_explicit():
    unknown=query(force_role='candidate_force_hypothesis')
    public=context.inference_public_context(known_hardware=hardware(),command_query=unknown)
    assert public['feature_groups']['public_known_force_programs']['held_handle_forces_world_N'].shape==(0,2,1,3)
    known=query(support_role='known_actuator_command')
    commanded=context.inference_public_context(known_hardware=hardware(),command_query=known)
    np.testing.assert_array_equal(commanded['feature_groups']['public_known_support_program']['prescribed_support_endpoint_translations_world_m'],known.support_translation)
    assert commanded['feature_groups']['public_known_force_programs']['held_handle_forces_world_N'].shape==(1,2,1,3)
    with pytest.raises(ValueError,match='information role'):query(force_role='from_truth_recipe')
    with pytest.raises(ValueError,match='information role'):query(support_role='from_truth_recipe')


def test_known_gravity_stays_a_command_without_oracle_density_scaled_nodal_forces():
    commands=query(held_body_acceleration_world_m_s2=np.tile([0,0,-9.81],(2,1)))
    dense=context.MaterialHypothesis(youngs_modulus_Pa=3000.,poissons_ratio=.35,density_kg_m3=1900.,
        body_linear_drag_s_inverse=.6,constitutive_model='log_J_neo_hookean')
    a=conditioned(commands=commands);b=conditioned(commands=commands,mat=dense)
    for key,value in a['feature_groups']['public_known_force_programs'].items():
        np.testing.assert_array_equal(value,b['feature_groups']['public_known_force_programs'][key])
    np.testing.assert_array_equal(a['feature_groups']['public_known_force_programs']['held_nodal_forces_world_N'],0.)


def test_every_provenance_manifest_hash_covers_its_nested_sources_and_hypotheses():
    result=conditioned(commands=query(extra_force_programs=[passive_program(.02,'b')]))
    def verify(value):
        if isinstance(value,dict):
            if 'manifest_sha256' in value:
                assert value['manifest_sha256']==context.digest({key:child for key,child in value.items() if key!='manifest_sha256'})
            for child in value.values():verify(child)
        elif isinstance(value,list):
            for child in value:verify(child)
    verify(result['provenance'])


def test_explicit_command_units_impulse_support_law_and_feature_schema():
    known=hardware();commands=query();bc=boundary();result=conditioned(known,commands,bc=bc)
    features=result['feature_groups']['public_known_commands']
    forces=result['feature_groups']['public_known_force_programs']['held_handle_forces_world_N'][0]
    np.testing.assert_allclose(np.sum(forces*features['held_interval_durations_s'][:,None,None],axis=0),[[.01,.02,0]],atol=1e-18,rtol=0)
    targets=context.support_targets_relative(known_hardware=known,boundary_hypothesis=bc,command_query=commands)
    point=POINTS[4]-ORIGIN;origin=POINTS[3]-ORIGIN
    expected=origin+rz(.4)@(point-origin)+[0,.02,0]
    np.testing.assert_allclose(targets['targets_relative_m'][-1,0,4],expected,atol=3e-16,rtol=0)
    np.testing.assert_array_equal(targets['constraint_axis_masks'],bc.masks)
    for group,fields in result['feature_groups'].items():
        assert set(fields)==set(result['schema']['groups'][group])
        assert all(spec['shape'] and spec['units'] for spec in result['schema']['groups'][group].values())
    assert 'no_material' in context.inference_public_context(known_hardware=known,command_query=commands)['role']


def test_no_input_source_file_or_observation_is_implicitly_opened(monkeypatch):
    original=Path.read_bytes;opened=[]
    allowed={'campaign_full_hardware_context.py','campaign_rigid_inertia.py'}
    def read_bytes(path):
        opened.append(path.name)
        if path.name not in allowed:pytest.fail('Implicit external read: '+str(path))
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read_bytes)
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:pytest.fail('Implicit array/archive loading'))
    result=conditioned()
    assert opened and set(opened)==allowed
    assert result['provenance']['known_hardware']['caller_declared_source_sha256_not_opened_or_verified']==SOURCE


def test_feature_return_values_are_copies_and_inputs_are_snapshotted():
    points=POINTS.copy();known=context.KnownHardware(rest_points_world_m=points,tetrahedra=TETS,surface_faces=FACES,
        handles=[handle()],coordinate_origin_world_m=ORIGIN,source_sha256=SOURCE)
    before=known.features();points[:]=100;returned=known.features();returned['patch_node_masks'][:]=False
    returned['handle_COM_inertias_body_kg_m2'][:]=0
    for key,value in before.items():np.testing.assert_array_equal(value,known.features()[key])
    with pytest.raises(ValueError):known.points[0]=0


@pytest.mark.parametrize('changes',[
    dict(com_inertia_body_kg_m2=np.diag([1.,1.,3.])),
    dict(com_inertia_body_kg_m2=[[1,.1,0],[0,1,0],[0,0,1]]),
    dict(com_offset_body_m=[0,np.nan,0]),dict(initial_rotation_body_to_world=np.diag([1,1,-1])),
    dict(ordered_patch_nodes=[0,0,1]),dict(mass_kg=0.),
])
def test_invalid_hardware_is_rejected_without_repair(changes):
    values=dict(origin_world_m=[0,0,0],ordered_patch_nodes=[0,1,2],mass_kg=.002,
        com_inertia_body_kg_m2=INERTIA,com_offset_body_m=[0,0,0],initial_rotation_body_to_world=np.eye(3))
    values.update(changes)
    with pytest.raises(ValueError):context.KnownHandle(**values)


def test_incomplete_mesh_origin_mismatch_and_ambiguous_patch_coupling_fail():
    with pytest.raises(ValueError,match='faces'):
        context.KnownHardware(rest_points_world_m=POINTS,tetrahedra=TETS,surface_faces=FACES[:-1],handles=[],coordinate_origin_world_m=ORIGIN,source_sha256=SOURCE)
    with pytest.raises(ValueError,match='origin'):
        hardware(handles=[handle(shift=(.1,0,0))])
    with pytest.raises(ValueError,match='Overlapping'):
        hardware(handles=[handle(),handle(patch=(0,3,4))])


@pytest.mark.parametrize('changes',[
    dict(boundary_times_s=[0.,.4,.4]),dict(boundary_times_s=[.1,.4,1.]),
    dict(requested_output_times_s=[0.,.5,1.]),dict(held_handle_forces_world_N=np.zeros((2,1,2))),
    dict(support_translations_world_m=np.ones((3,1,3))),
])
def test_invalid_or_underspecified_command_program_fails(changes):
    with pytest.raises(ValueError):query(**changes)


def test_boundary_masks_are_latent_explicit_and_conflicts_or_dimension_mismatch_fail():
    with pytest.raises(ValueError,match='boolean'):
        context.BoundaryHypothesis(constraint_axis_masks=np.ones((1,5,3)),support_origins_relative_m=[[0,0,0]])
    with pytest.raises(ValueError,match='two support'):
        context.BoundaryHypothesis(constraint_axis_masks=np.ones((2,5,3),dtype=bool),support_origins_relative_m=np.zeros((2,3)))
    with pytest.raises(ValueError,match='overlap'):conditioned(bc=boundary(0))
    with pytest.raises(ValueError,match='dimensions'):conditioned(commands=query(handles=0))


def test_free_object_zero_handle_and_zero_support_shapes_are_complete():
    known=hardware(handles=[]);commands=query(handles=0,supports=0)
    bc=context.BoundaryHypothesis(constraint_axis_masks=np.zeros((0,5,3),dtype=bool),support_origins_relative_m=np.zeros((0,3)))
    result=conditioned(known,commands,bc=bc)
    assert result['feature_groups']['public_known_hardware']['patch_node_masks'].shape==(0,5)
    assert result['feature_groups']['latent_boundary_hypothesis']['support_constraint_axis_masks'].shape==(0,5,3)
    assert context.support_targets_relative(known_hardware=known,boundary_hypothesis=bc,command_query=commands)['targets_relative_m'].shape==(3,0,5,3)


def test_ragged_patch_padding_and_axis_selective_boundary_groups_preserve_full_membership():
    second=context.KnownHandle(origin_world_m=POINTS[3],ordered_patch_nodes=[3,4],mass_kg=.003,
        com_inertia_body_kg_m2=INERTIA,com_offset_body_m=[0,0,0],initial_rotation_body_to_world=np.eye(3),origin_node_id=3)
    features=hardware(handles=[handle(),second]).features()
    np.testing.assert_array_equal(features['ordered_patch_node_ids'],[[0,1,2],[3,4,-1]])
    np.testing.assert_array_equal(features['ordered_patch_valid_mask'],[[True,True,True],[True,True,False]])
    np.testing.assert_array_equal(features['ordered_patch_local_offsets_body_m'][1,-1],0)
    masks=np.zeros((2,5,3),dtype=bool);masks[0,4,0]=True;masks[1,4,1:]=True
    bc=context.BoundaryHypothesis(constraint_axis_masks=masks,support_origins_relative_m=np.tile(POINTS[3]-ORIGIN,(2,1)))
    result=conditioned(commands=query(supports=2),bc=bc)
    np.testing.assert_array_equal(result['feature_groups']['latent_boundary_hypothesis']['support_constraint_axis_masks'],masks)
