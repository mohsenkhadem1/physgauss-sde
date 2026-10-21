"""Pure public-query/counterfactual admission contract tests, no physics."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('admission_contract_draft',Path(__file__).with_name('admission_contract.py'))
draft=importlib.util.module_from_spec(spec);spec.loader.exec_module(draft)
H=np.array([[.12,.04,.25],[-.12,-.04,.25]])


def test_no_hidden_material_or_boundary_arguments_exist():
    m=draft.menu(public_handle_origins_m=H)
    assert len(m['actions'])==12
    for forbidden in ['material','boundary_type','support_nodes','passive_force','solver_hz','recipe']:
        with pytest.raises(TypeError):draft.menu(public_handle_origins_m=H,**{forbidden:None})


def test_identical_menu_across_latent_crossproduct_and_common_amplitudes():
    reference=draft.fingerprint(draft.menu(public_handle_origins_m=H))
    for latent_bc in range(4):
        for latent_material in [1500,5000,20000]:
            assert draft.fingerprint(draft.menu(public_handle_origins_m=H))==reference
    for peak in [.01,.02]:
        m=draft.menu(public_handle_origins_m=H,common_peak_N=peak)
        for a in m['actions'][1:]:np.testing.assert_allclose(np.linalg.norm(a['peak_world_N'][0]),peak,rtol=0,atol=1e-17)


def test_exact_common_commands_fixed_schema_and_couple_resultant():
    m=draft.menu(public_handle_origins_m=H)
    for a in m['actions']:
        c=draft.compile_known_action(m,a['id'])
        assert c['known_handle_forces_N'].shape==(1,360,2,3)
        assert c['known_handle_forces_N'].dtype==np.float64
        assert c['requested_output_times_s'].shape==(61,)
        assert c['command_boundary_times_s'].shape==(361,)
        assert not np.any(c['known_handle_pure_torques_Nm'])
        if a['family']=='twist_couple':np.testing.assert_array_equal(c['known_handle_forces_N'].sum(2),0)


def test_private_refinement_repeats_exact_commands_preserves_impulse():
    m=draft.menu(public_handle_origins_m=H)
    c=draft.compile_known_action(m,'shear_oscillation__direction0')
    original=c['known_handle_forces_N']
    for hz in [120,240,480,960,1920]:
        expanded=draft.held_force_at_private_midpoints(c,hz)
        np.testing.assert_array_equal(expanded.reshape(1,360,hz//120,2,3),np.repeat(original[:,:,None],hz//120,axis=2))
        np.testing.assert_allclose(expanded.sum(1)/hz,original.sum(1)/120,rtol=0,atol=1e-16)
    for hz in [0,20,121,False]:
        with pytest.raises(ValueError):draft.held_force_at_private_midpoints(c,hz)


def test_translation_changes_neither_action_basis_nor_command_clock():
    a=draft.menu(public_handle_origins_m=H)
    b=draft.menu(public_handle_origins_m=H+[0,0,-.25])
    assert draft.fingerprint(a)==draft.fingerprint(b)


def rows():
    m=draft.menu(public_handle_origins_m=H);signature=draft.fingerprint(m)
    return [{'public_hardware_id':'publicH0','menu_replicate':0,'hidden_BC':b,'hidden_material':e,
        'menu_sha256':signature,'command_clock_sha256':'clock','camera_program_sha256':'camera','feature_schema_sha256':'fixed',
        'known_capacities':{'H':2,'S':2,'K':1},'public_recipe_fields':[]} for b in range(4) for e in [1500,5000,20000]]


@pytest.mark.parametrize('field',['menu_sha256','command_clock_sha256','camera_program_sha256','feature_schema_sha256'])
def test_hidden_condition_specific_public_bytes_are_rejected(field):
    r=rows();assert draft.check_counterfactual_manifest(r)
    r[-1][field]='BC-specific changed public value'
    with pytest.raises(ValueError,match='hidden condition'):draft.check_counterfactual_manifest(r)


def test_hidden_ids_or_active_capacity_side_channels_rejected():
    r=rows();r[0]['public_recipe_fields']=['recipe_sha256']
    with pytest.raises(ValueError,match='Recipe'):draft.check_counterfactual_manifest(r)
    r=rows();r[0]['known_capacities']['S']=1
    with pytest.raises(ValueError,match='capacities'):draft.check_counterfactual_manifest(r)


def test_no_amplitude_rescaling_or_silent_undeclared_tier():
    for peak in [.0025,.03,.12,.24]:
        with pytest.raises(ValueError,match='common tiers'):draft.menu(public_handle_origins_m=H,common_peak_N=peak)


def test_no_silent_missing_category_or_duplicate_counterfactual():
    r=rows();r=[row for row in r if row['hidden_BC']!=0]
    with pytest.raises(ValueError,match='crossproduct'):draft.check_counterfactual_manifest(r)
    r=rows();r.append(deepcopy(r[0]))
    with pytest.raises(ValueError,match='Duplicate'):draft.check_counterfactual_manifest(r)
