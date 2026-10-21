"""Exact prospective request gate for eight matched TRAIN moving960 controls.

No physical solve or reserved observation is opened. Publication/launch approval
remains an orchestration requirement; validation does not qualify a reference.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

PROTOCOL_PATH=Path('configs/gaussian_campaign_dynamic_960_validation_v02.json')
PROTOCOL_SHA256='b7c6692a34d5b67c6b99f8ff6124bc9520e20da6afa715facecaa392d9f331fc'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def validate_matched_960(protocol_path,*,study_cell_id,episode,hz,response,relative_tolerance,
                         absolute_tolerance_N,maximum_iterations,translation,inputs_json,core_path,gate_path):
    protocol_path=Path(protocol_path)
    require(digest(protocol_path)==PROTOCOL_SHA256,'Matched960 protocol hash differs from declaration')
    declaration=read(protocol_path)
    require(declaration['status']=='prospectively_declared_before_eight_matched_runs','Prospective matched960 declaration required')
    require(inputs_json is not None and digest(inputs_json)==declaration['static_fixture_sha256'],'Exact public TRAIN static fixture required')
    require(digest(core_path)==declaration['core_sha256'],'Frozen physical core required')
    require(digest(gate_path)==declaration['parent_gate_sha256'],'Frozen parent gate required')
    dependencies=[protocol_path]
    for group in ['physical_source_input_sha256','proof_source_sha256']:
        for path,wanted in declaration[group].items():
            require(digest(path)==wanted,'Pinned matched960 source/input differs: '+path)
            if group=='proof_source_sha256':dependencies.append(Path(path))
    proofs={}
    for key,item in declaration['completed_proofs'].items():
        path=Path(item['path']);require(digest(path)==item['sha256'],'Pinned completed proof differs: '+key)
        proofs[key]=read(path);dependencies.append(path)
    candidate=proofs['candidate_audit'];floor=proofs['floor_complete_audit'];trigger=proofs['initial_failure_trigger']
    require(candidate['conditional_960_temporal_candidate_passed'] is True
        and candidate['initial_moving_temporal_gate_passed'] is False
        and candidate['initial_comparisons_independently_recomputed'] is True
        and candidate['temporal_manifest_state']=='verified'
        and candidate['matched_960_operating_point_checks_completed'] is False
        and candidate['overall_independent_reference_qualified'] is False,'Verified candidate with preserved initial failure required')
    original=[c for c in trigger['comparisons'] if c['case']=='t0003' and c['kind']=='temporal']
    require(len(original)==2 and all(c['passed'] is False for c in original)
        and candidate['initial_120_240_480_comparisons_preserved']==original,'Initial failed comparisons must remain unchanged')
    comparisons=candidate['extension_comparisons']
    require(len(comparisons)==2 and {c['response'] for c in comparisons}=={'zero_external_force','diagonal_pulse_handle0'}
        and all(c['actual_rates_hz']==[120,240,480,960] and c['finest_adjacent_comparison_passed'] is True for c in comparisons),
        'Both complete-response960 candidate comparisons must pass')
    require(candidate['temporal_manifest_sha256']==declaration['completed_proofs']['candidate_manifest']['sha256']
        and candidate['baseline_manifest_sha256']==declaration['completed_proofs']['baseline_manifest']['sha256'],
        'Candidate proof manifest lineage mismatch')
    require(len(floor['cells'])==50 and all(c['passed'] is True for c in floor['cells'])
        and floor['coverage']['missing_slots']==[] and floor['coverage']['unexpected_slots']==[],'All50 prior baseline/floor cells required')
    floor_study=[c for c in floor['comparisons'] if c['kind']=='absolute_floor']
    require(len(floor_study)==1 and floor_study[0]['passed'] is True,'Completed480 effective-floor study required')
    cells=declaration['new_cells'];matches=[row for row in cells if row['id']==study_cell_id]
    require(isinstance(study_cell_id,str) and len(cells)==8 and len(matches)==1,'Exactly one of eight declared study cell IDs required')
    cell=matches[0];shift=np.asarray(translation,dtype=float)
    require(episode==declaration['episode_id'] and type(hz) is int and hz==declaration['physics_hz']==960
        and response==cell['response'] and relative_tolerance==cell['relative_tolerance']
        and absolute_tolerance_N==cell['absolute_tolerance_N'] and type(maximum_iterations) is int
        and maximum_iterations==declaration['nonlinear_iteration_cap']==250
        and shift.shape==(3,) and np.array_equal(shift,cell['translation_m']),
        'Request differs from exact declared matched960 study cell')
    gate=read(gate_path)
    require(gate['seconds']==declaration['seconds']==3. and gate['output_hz']==declaration['output_hz']==10,'Frozen physical clocks required')
    return {'protocol_path':str(protocol_path),'protocol_sha256':PROTOCOL_SHA256,
        'parent_gate_sha256':declaration['parent_gate_sha256'],'study_cell_id':study_cell_id,'kind':cell['kind'],
        'episode_id':episode,'physics_hz':hz,'response':response,'relative_tolerance':relative_tolerance,
        'absolute_residual_floor_N':absolute_tolerance_N,'translation_m':shift.tolist(),
        'completed_proof_sha256':{key:item['sha256'] for key,item in declaration['completed_proofs'].items()},
        'candidate_temporal_pass_verified':True,'initial_temporal_failure_preserved':True,
        'physical_model_unchanged':True,'reference_qualification_implied':False,
        'provenance_dependency_paths':list(dict.fromkeys(str(path) for path in dependencies))}
