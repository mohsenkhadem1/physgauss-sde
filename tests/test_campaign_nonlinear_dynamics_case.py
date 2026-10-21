"""Protocol/input/force controls; no TRAIN time integration is launched."""
import json
from pathlib import Path
import numpy as np
import pytest

from experiments.campaign_nonlinear_dynamics_case import (
    GATE_PATH,GATE_SHA256,sha,validate_request,boundary_motion,force_command,
    hardware_descriptor,load_case_inputs,run)


def test_frozen_gate_rejects_reserved_roles_rates_and_response_changes_before_output(tmp_path):
    gate=json.loads(GATE_PATH.read_text());assert sha(GATE_PATH)==GATE_SHA256
    for episode in ['t0000','t0001','t0002','t0003']:
        for hz in [120,240,480]:
            assert validate_request(gate,episode,hz,'diagonal_pulse_handle0',1e-7,[0,0,0]).shape==(3,)
    for episode,hz,response,tol,shift in [('v0000',120,'diagonal_pulse_handle0',1e-7,[0,0,0]),
        ('t0001',60,'diagonal_pulse_handle0',1e-7,[0,0,0]),
        ('t0001',120,'poke_0',1e-7,[0,0,0]),
        ('t0001',120,'diagonal_pulse_handle0',1e-4,[0,0,0]),
        ('t0001',120,'diagonal_pulse_handle0',1e-7,[0,.1,0])]:
        with pytest.raises(ValueError):validate_request(gate,episode,hz,response,tol,shift)
    output=tmp_path/'must-not-exist'
    with pytest.raises(ValueError,match='TRAIN'):
        run(output,episode='v0000')
    assert not output.exists()


def test_midpoint_held_force_impulse_at_all_predeclared_rates():
    gate=json.loads(GATE_PATH.read_text())
    exact=.675*np.array([[.01,.01,.005],[0.,0.,0.]])
    for hz in [120,240,480]:
        command=np.array([force_command(gate,'diagonal_pulse_handle0',(k+.5)/hz) for k in range(3*hz)])
        np.testing.assert_allclose(command.sum(0)/hz,exact,atol=3e-17,rtol=0)
        np.testing.assert_array_equal([force_command(gate,'zero_external_force',t) for t in [0,.5,1.,3.]],0)


def test_moving_support_known_amplitude_and_initial_derivative():
    spec={'seconds':3.,'motion_amplitude_m':.008,'motion_direction_world':[0.,-.8,.6]}
    shift,v=boundary_motion(spec,0.);np.testing.assert_array_equal(shift,0);np.testing.assert_array_equal(v,0)
    np.testing.assert_allclose(boundary_motion(spec,1.5)[0],np.array([0.,-.8,.6])*.008,atol=1e-17,rtol=0)
    t=.7;epsilon=1e-5
    finite_difference=(boundary_motion(spec,t+epsilon)[0]-boundary_motion(spec,t-epsilon)[0])/(2*epsilon)
    np.testing.assert_allclose(finite_difference,boundary_motion(spec,t)[1],atol=1e-11,rtol=0)


def test_public_static_fixture_regenerates_all_four_TRAIN_cases_without_dataset():
    fixture=Path('configs/gaussian_independent_diagnostic_inputs_v01.json')
    families=[]
    for episode in ['t0000','t0001','t0002','t0003']:
        case,spec,arrays,inputs,lineage=load_case_inputs(episode,'/nonexistent-bulk-dataset',fixture)
        assert case['split']=='train' and case['episode_id']==episode
        assert arrays['rest_points_m'].shape==(455,3)
        assert arrays['tetrahedra'].shape==(1728,4)
        assert lineage['boundary_spec_exactly_regenerated'] is True
        families.append(spec['boundary_type'])
        hardware=hardware_descriptor(arrays['rest_points_m'],spec)
        assert len(hardware)==2
        for h in hardware:
            ids=h['ordered_patch_nodes'];origin=h['reference_origin_node']
            np.testing.assert_array_equal(h['ordered_reference_local_offsets_m'],arrays['rest_points_m'][ids]-arrays['rest_points_m'][origin])
            assert np.linalg.eigvalsh(h['full_COM_inertia_kg_m2']).min()>0
    assert families==['point_pin','fixed_patch','two_patches','moving_patch']


def test_corrupt_public_static_fixture_hash_and_boundary_spec_are_rejected(tmp_path):
    original=json.loads(Path('configs/gaussian_independent_diagnostic_inputs_v01.json').read_text())
    bad=json.loads(json.dumps(original));bad['cases'][0]['reference_static_arrays'][0]['sha256']='0'*64
    path=tmp_path/'corrupt-hash.json';path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='hash mismatch'):load_case_inputs('t0000',None,path)
    bad=json.loads(json.dumps(original));bad['cases'][0]['boundaries']['fixture_nodes']=[0]
    path=tmp_path/'corrupt-spec.json';path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='specification'):load_case_inputs('t0000',None,path)
