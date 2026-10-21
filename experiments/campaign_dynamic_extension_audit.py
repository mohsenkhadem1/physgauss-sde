"""Read-only audit of the two conditional moving-case 960 Hz responses.

Preserves the published failed three-rate comparisons. A passing extension is
only a temporal candidate, never full reference or operating-point adoption.
No model, native simulation, training or reserved observation is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import campaign_independent_dynamic_audit as base
except ImportError:
    from experiments import campaign_independent_dynamic_audit as base


BASE_AUDIT_SHA = '310d4e22acdd14e57ebbcb302e7d66eeb09960343f5594c1e87bd6ed6b06f6a9'
BASE_MANIFEST_SHA = '688f89e1de05e98ce91783e1d53c1ad775fa5630c607c288c1ecd444f93c56db'
TEMPORAL_PATH = Path('configs/gaussian_campaign_dynamic_temporal_extension_v01.json')
TEMPORAL_SHA = 'e6c9efb31c34ea1b6ffe476da192edecad3da5697c0cebaf185ddc5d8813df6f'
TRIGGER_SHA = '3733850713231a5d750a74e8d454475fe30f579f6b4e638699a53e6268872f43'
RUNNER_SHA = 'a0fefae860eef4f7442c019045a72b2544d7708bd3fc47e737a7b3a649b0d87f'
HELPER_SHA = 'dd9e43aa0bc8e570133fc21b6d54e1a85e678fb395bba660e6ab544ee43ea64a'
BASE_RUNNER_SHA = 'f062f0db2c2ac7db00a5897c846b2f588f7394cf92895b28a808c7aa4223b65d'
CORE_SHA = 'da0db7018b4913020bb228c2fc4df3c240e7dd872f17db4868de147eaaf78fa2'


def extension_slot(response):
    return ('t0003',response,960,1e-7,(0.,0.,0.),'temporal_extension',1e-10)


def validate_manifest(manifest,baseline_manifest):
    """Reject implicit runner revisions, unapproved extra cells and replays."""
    base.require(manifest['parent_gate_sha256']==base.GATE_SHA,'Extension manifest parent gate mismatch')
    base.require(manifest['temporal_protocol_sha256']==TEMPORAL_SHA,'Extension manifest protocol mismatch')
    base.require(manifest['baseline_manifest_sha256']==BASE_MANIFEST_SHA,'Extension manifest baseline mismatch')
    base.require(manifest['trigger_audit_sha256']==TRIGGER_SHA,'Extension manifest failed trigger mismatch')
    base.require(manifest['approved_runner_source_sha256']==[RUNNER_SHA],'Unapproved extension runner revision')
    base.require(manifest['temporal_dependency_sha256']=={'campaign_dynamic_temporal_protocol.py':HELPER_SHA},'Unapproved temporal helper revision')
    cells=manifest['cells'];actual=[base.slot(cell) for cell in cells]
    expected=[extension_slot(response) for response in base.RESPONSES]
    base.require(len(actual)==2 and set(actual)==set(expected),'Exactly two distinct declared 960 Hz slots required')
    paths=[Path(cell['output']).resolve() for cell in cells]
    old_paths={Path(cell['output']).resolve() for cell in baseline_manifest['cells']}
    base.require(len(set(paths))==2 and not any(path in old_paths for path in paths),'Extension outputs must be distinct fresh paths')
    return cells


def verify_conditional_provenance(cell):
    """Base audit verifies all saved bytes; here enforce the new protocol role."""
    protocol=base.read(Path(cell['output'])/'protocol.json')
    names={Path(path).name:digest for path,digest in protocol['source_input_dependency_and_gate_sha256'].items()}
    required={'campaign_nonlinear_dynamics_case.py':RUNNER_SHA,
        'campaign_dynamic_temporal_protocol.py':HELPER_SHA,TEMPORAL_PATH.name:TEMPORAL_SHA,
        'independent-dynamic-t0003-refinement-trigger-v01.json':TRIGGER_SHA}
    base.require(all(names.get(name)==digest for name,digest in required.items()),'Conditional protocol/helper/trigger provenance mismatch')
    metadata=protocol['conditional_temporal_protocol']
    base.require(metadata['protocol_sha256']==TEMPORAL_SHA and metadata['parent_gate_sha256']==base.GATE_SHA
        and metadata['trigger_audit_sha256']==TRIGGER_SHA and metadata['episode_id']=='t0003'
        and metadata['physics_hz']==960 and metadata['response']==cell['response']
        and metadata['initial_temporal_failure_preserved'] is True
        and metadata['physical_model_unchanged'] is True and metadata['reference_qualification_implied'] is False,
        'Conditional request metadata mismatch')
    base.require(protocol.get('supplemental_accuracy_protocol') is None,'Combined floor/temporal request is not authorized')


def extension_metrics(trajectories,zero_finest,diagonal_finest):
    """Four full trajectories; subtract zero only for the signal denominator."""
    base.require(len(trajectories)==4,'Exactly four actual temporal levels required')
    differences=[base.coordinate_rmse(trajectories[i+1],trajectories[i]) for i in range(3)]
    signal=base.coordinate_rmse(diagonal_finest,zero_finest);tolerance=max(2e-6,.05*signal)
    contraction=[bool(differences[i+1]<differences[i]) for i in range(2)]
    floor_exception=bool(differences[-2]<2e-6 and differences[-1]<2e-6)
    passed=bool(differences[-1]<tolerance and (contraction[-1] or floor_exception))
    return {'actual_rates_hz':[120,240,480,960],
        'adjacent_rate_pairs_hz':[[120,240],[240,480],[480,960]],
        'full_coordinate_rmse_m':differences,'measured_contraction':contraction,
        'adjacent_error_ratios':[None if differences[i]==0 else differences[i+1]/differences[i] for i in range(2)],
        'finest_force_induced_coordinate_RMS_m':signal,'tolerance_m':tolerance,
        'final_two_differences_below_2micrometre_floor':floor_exception,
        'finest_adjacent_comparison_passed':passed,
        'initial_480_failure_is_not_replaced':True}


def preserve_initial_comparisons(recomputed,declared):
    base.require(len(recomputed)==len(declared)==2,'Both initial response comparisons required')
    for current,original in zip(recomputed,declared):
        base.require(all(current[key]==original[key] for key in ['case','response','kind','passed','both_differences_below_2micrometre_floor']),'Initial temporal decision differs from immutable trigger')
        for key in ['coordinate_rmse_m','finest_force_induced_coordinate_RMS_m','tolerance_m']:
            base.close(current[key],original[key],1e-15,'Initial temporal metric differs from immutable trigger')
    return declared


def run(baseline_manifest_path,output,*,temporal_manifest_path=None):
    base.require(base.sha(base.__file__)==BASE_AUDIT_SHA,'Published base audit revision changed')
    base.require(base.sha(baseline_manifest_path)==BASE_MANIFEST_SHA,'Frozen baseline manifest changed')
    base.require(base.sha(TEMPORAL_PATH)==TEMPORAL_SHA,'Published temporal declaration changed')
    base.require(base.sha(base.GATE_PATH)==base.GATE_SHA and base.sha(base.INPUT_PATH)==base.INPUT_SHA,'Published gate or static fixture changed')
    declaration=base.read(TEMPORAL_PATH);trigger_path=Path(declaration['trigger_audit_path'])
    base.require(base.sha(trigger_path)==TRIGGER_SHA,'Immutable failed trigger audit changed')
    trigger=base.read(trigger_path)
    declared=declaration['trigger_comparisons']
    base.require([c for c in trigger['comparisons'] if c['case']=='t0003' and c['kind']=='temporal']==declared,'Declaration/trigger comparisons differ')
    base.require(all(c['passed'] is False for c in declared),'Initial failed temporal decisions must remain failed')
    manifest=base.read(baseline_manifest_path);gate=base.read(base.GATE_PATH);canonical=base.read(base.INPUT_PATH)
    base.require(manifest['approved_runner_source_sha256']==[BASE_RUNNER_SHA],'Unexpected baseline runner revision')
    dependencies={Path(name).name:digest for name,digest in manifest['frozen_source_input_and_gate_sha256'].items()
        if Path(name).name in base.CORE+['isaac_dataset.py','boundary_block.py',base.INPUT_PATH.name]}
    base.require(dependencies.get('campaign_nonlinear_dynamics.py')==CORE_SHA,'Frozen physical core mismatch')
    results=[];loaded={};initial=[];extension_manifest_sha=None;manifest_state='missing';initial_recomputed=False
    selected=[cell for cell in manifest['cells'] if cell['case']=='t0003' and cell['kind']=='temporal']
    base.require(len(selected)==6,'Exactly six original moving-case temporal cells required')
    for cell in selected:
        report,data=base.audit_cell(cell,gate,canonical,{BASE_RUNNER_SHA},dependencies)
        report['evidence_role']='original_temporal';results.append(report)
        if report['passed']:loaded[(cell['response'],cell['hz'])]=data
    if all((response,hz) in loaded for response in base.RESPONSES for hz in [120,240,480]):
        zero=loaded[(base.RESPONSES[0],480)]['positions_m'];diag=loaded[(base.RESPONSES[1],480)]['positions_m']
        recomputed=[{'case':'t0003','response':response,'kind':'temporal',**base.temporal_metrics(
            [loaded[(response,hz)]['positions_m'] for hz in [120,240,480]],zero,diag)} for response in base.RESPONSES]
        initial=preserve_initial_comparisons(recomputed,declared)
        initial_recomputed=True
    else:initial=declared
    if temporal_manifest_path is not None and Path(temporal_manifest_path).exists():
        extension_manifest=base.read(temporal_manifest_path)
        extension_cells=validate_manifest(extension_manifest,manifest);extension_manifest_sha=base.sha(temporal_manifest_path);manifest_state='verified'
        for cell in extension_cells:
            report,data=base.audit_cell(cell,gate,canonical,{RUNNER_SHA},dependencies)
            report['evidence_role']='conditional_temporal'
            if report['passed']:
                try:
                    verify_conditional_provenance(cell)
                    base.require(report['external_handle_work_independently_reconstructed_from_saved_step_positions'] is True,
                        'New runner requires complete independent per-step handle work evidence')
                except (base.AuditError,KeyError,TypeError,ValueError,OSError) as error:
                    report.update(passed=False,state='invalid',error=str(error));data=None
            results.append(report)
            if report['passed']:loaded[(cell['response'],960)]=data
    comparisons=[]
    if all((response,hz) in loaded for response in base.RESPONSES for hz in [120,240,480,960]):
        zero=loaded[(base.RESPONSES[0],960)]['positions_m'];diag=loaded[(base.RESPONSES[1],960)]['positions_m']
        for response in base.RESPONSES:
            comparisons.append({'case':'t0003','response':response,'kind':'conditional_temporal_extension',
                **extension_metrics([loaded[(response,hz)]['positions_m'] for hz in [120,240,480,960]],zero,diag)})
    mechanics=base.mechanics_audit('runs/operator-dev/nonlinear-dynamics-mechanics-v01',
        {name:digest for name,digest in dependencies.items() if name in base.CORE})
    candidate=bool(len(results)==8 and all(report['passed'] for report in results) and len(comparisons)==2
        and all(c['finest_adjacent_comparison_passed'] for c in comparisons) and mechanics['passed'])
    result={'audit_source_sha256':base.sha(__file__),'published_base_audit_sha256':BASE_AUDIT_SHA,
        'baseline_manifest_sha256':BASE_MANIFEST_SHA,'temporal_manifest_sha256':extension_manifest_sha,
        'temporal_manifest_state':manifest_state,'temporal_protocol_sha256':TEMPORAL_SHA,'failed_trigger_sha256':TRIGGER_SHA,
        'cells':results,'required_temporal_cells':8,'indexed_temporal_cells':len(results),'mechanics':mechanics,
        'initial_120_240_480_comparisons_preserved':initial,'initial_comparisons_independently_recomputed':initial_recomputed,
        'initial_moving_temporal_gate_passed':False,'extension_comparisons':comparisons,
        'conditional_960_temporal_candidate_passed':candidate,
        'matched_960_operating_point_checks_completed':False,'moving_960_reference_qualified':False,
        'overall_independent_reference_qualified':False,'native_reference_qualified':False,
        'prior_480_absolute_floor_study_still_required':True,
        'interaction_limit':'Prior nonlinear accuracy, translation, replay and absolute-floor checks at 480 Hz remain 480 Hz evidence. They do not establish those interactions at 960 Hz. No 1920 Hz or force-envelope expansion is authorized.',
        'operating_envelope':'Declared TRAIN t0003 material, mesh, support law and initial rest; zero or [0.01,0.01,0.005] N midpoint-held 3 s pulse only. Original 0.12–0.24 N profiles, prediction, calibrated inference and policy are not qualified.',
        'qualification':'Temporal candidate only. Initial failed gate objects remain unchanged; matched 960 Hz nonlinear accuracy, absolute-floor moving-zero, translation and fresh reset controls require prospective declaration/publication before execution.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-manifest',required=True);parser.add_argument('--temporal-manifest')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.baseline_manifest,args.output,temporal_manifest_path=args.temporal_manifest)
    print(json.dumps({'verified_cells':sum(c['passed'] for c in result['cells']),
        'conditional_temporal_candidate':result['conditional_960_temporal_candidate_passed'],
        'overall_reference_qualified':False}))
