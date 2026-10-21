"""Prospective exact matched960 guard; no physical integrations or outputs."""
import json
from pathlib import Path
import pytest
from experiments import campaign_dynamic_960_protocol as matched
from experiments import campaign_nonlinear_dynamics_case as runner


def declaration():return json.loads(matched.PROTOCOL_PATH.read_text())
def request(cell=None,**changes):
    cell=declaration()['new_cells'][0] if cell is None else cell
    values=dict(study_cell_id=cell['id'],episode='t0003',hz=960,response=cell['response'],
        relative_tolerance=cell['relative_tolerance'],absolute_tolerance_N=cell['absolute_tolerance_N'],
        maximum_iterations=250,translation=cell['translation_m'],inputs_json='configs/gaussian_independent_diagnostic_inputs_v01.json',
        core_path='experiments/campaign_nonlinear_dynamics.py',gate_path='configs/gaussian_campaign_independent_dynamic_gate_v01.json')
    values.update(changes);return values


@pytest.mark.parametrize('cell',declaration()['new_cells'],ids=lambda cell:cell['id'])
def test_exact_eight_controls_require_verified_candidate_and_preserve_initial_failure(cell):
    metadata=matched.validate_matched_960(matched.PROTOCOL_PATH,**request(cell))
    assert metadata['study_cell_id']==cell['id'] and metadata['kind']==cell['kind']
    assert metadata['candidate_temporal_pass_verified'] is True
    assert metadata['initial_temporal_failure_preserved'] is True
    assert metadata['physical_model_unchanged'] is True
    assert metadata['reference_qualification_implied'] is False
    paths=metadata['provenance_dependency_paths'];assert len({Path(p).name for p in paths})==len(paths)
    assert all(Path(p).is_file() for p in paths)


@pytest.mark.parametrize('changes',[
    dict(study_cell_id=None),dict(study_cell_id='moving960-zero-reuse'),dict(episode='v0003'),dict(episode='t0001'),
    dict(hz=480),dict(hz=1920),dict(hz=960.),dict(response='zero_external_force'),dict(relative_tolerance=1e-7),
    dict(relative_tolerance=float('nan')),dict(absolute_tolerance_N=1e-12),dict(absolute_tolerance_N=float('inf')),
    dict(maximum_iterations=251),dict(maximum_iterations=250.),dict(translation=[0,0,-.25]),
    dict(translation=[0,float('nan'),0]),dict(inputs_json=None),
])
def test_rejects_undeclared_combinations(changes):
    with pytest.raises(ValueError):matched.validate_matched_960(matched.PROTOCOL_PATH,**request(**changes))


def test_all_proof_source_and_physical_dependency_bytes_are_pinned(monkeypatch,tmp_path):
    d=declaration();bad=tmp_path/'changed.json';bad.write_text(matched.PROTOCOL_PATH.read_text()+' ')
    with pytest.raises(ValueError,match='protocol hash'):matched.validate_matched_960(bad,**request())
    original=matched.digest
    paths=list(d['physical_source_input_sha256'])+list(d['proof_source_sha256'])+[p['path'] for p in d['completed_proofs'].values()]
    for changed in paths:
        with monkeypatch.context() as patch:
            patch.setattr(matched,'digest',lambda p,changed=changed:'0'*64 if Path(p)==Path(changed) else original(p))
            with pytest.raises(ValueError):matched.validate_matched_960(matched.PROTOCOL_PATH,**request())


@pytest.mark.parametrize('proof,field,value',[
    ('candidate_audit','conditional_960_temporal_candidate_passed',False),
    ('candidate_audit','initial_moving_temporal_gate_passed',True),
    ('candidate_audit','initial_comparisons_independently_recomputed',False),
    ('candidate_audit','overall_independent_reference_qualified',True),
    ('candidate_audit','initial_120_240_480_comparisons_preserved',[]),
    ('candidate_audit','temporal_manifest_sha256','0'*64),
    ('floor_complete_audit','cells',[]),
])
def test_verified_proof_decisions_required_even_after_hash_check(monkeypatch,proof,field,value):
    # Exercise semantic rejection separately from the independent immutable-byte gate.
    original=matched.read;target=Path(declaration()['completed_proofs'][proof]['path'])
    def altered(p):
        data=original(p)
        if Path(p)==target:data[field]=value
        return data
    monkeypatch.setattr(matched,'read',altered)
    with pytest.raises(ValueError):matched.validate_matched_960(matched.PROTOCOL_PATH,**request())


class BeforeInputs(Exception):pass


@pytest.mark.parametrize('cell',declaration()['new_cells'],ids=lambda cell:cell['id'])
def test_runner_accepts_exact_eight_before_static_input_boundary(tmp_path,monkeypatch,cell):
    def stop(*a,**kw):raise BeforeInputs
    monkeypatch.setattr(runner,'load_case_inputs',stop)
    output=tmp_path/'no-output';values=request(cell)
    for key in ['core_path','gate_path']:values.pop(key)
    with pytest.raises(BeforeInputs):runner.run(output,matched_960_protocol=matched.PROTOCOL_PATH,**values)
    assert not output.exists()


@pytest.mark.parametrize('changes',[
    dict(study_cell_id=None),dict(matched_960_protocol=None),dict(hz=1920),dict(episode='t0000'),
    dict(relative_tolerance=1e-7),dict(absolute_tolerance_N=1e-12),dict(study_cell_id='invalid'),
    dict(temporal_extension_protocol='configs/gaussian_campaign_dynamic_temporal_extension_v01.json'),
    dict(supplemental_floor_protocol='configs/gaussian_campaign_dynamic_floor_refinement_v01.json'),
])
def test_runner_rejects_before_inputs_or_output(tmp_path,monkeypatch,changes):
    def forbidden(*a,**kw):pytest.fail('Undeclared request reached static inputs')
    monkeypatch.setattr(runner,'load_case_inputs',forbidden)
    output=tmp_path/'no-output';values=request()
    for key in ['core_path','gate_path']:values.pop(key)
    values['matched_960_protocol']=matched.PROTOCOL_PATH;values.update(changes)
    with pytest.raises(ValueError):runner.run(output,**values)
    assert not output.exists()
