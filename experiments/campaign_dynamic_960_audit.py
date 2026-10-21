"""Read-only60-unique-cell audit of declared per-BC small-force operating points.

Reaudits all50 original/floor and both conditional responses, then eight matched
controls. Original480 failures remain false and visible. No physical jobs run.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
try:
    import campaign_independent_dynamic_audit as base
    import campaign_dynamic_extension_audit as temporal
    import campaign_dynamic_960_protocol as guard
except ImportError:
    from experiments import campaign_independent_dynamic_audit as base
    from experiments import campaign_dynamic_extension_audit as temporal
    from experiments import campaign_dynamic_960_protocol as guard

RUNNER_SHA='e7b6b785c262f43d4b1acc503b14feaf1cad659c171024346fc9e70a4dffc342'
HELPER_SHA='98576cc72c8172e6a0766d0fd4f2ddfb8f900d54ca51dd96e270faca8f85edcb'
TEMPORAL_AUDIT_SHA='bfeb11bad5aa243858c3b3fb2408a40b21898fa16cfe61cd68981f1e7a89fe73'


def validate_manifest(manifest,declaration,old_cells):
    base.require(manifest['matched_protocol_sha256']==guard.PROTOCOL_SHA256,'Matched protocol linkage mismatch')
    base.require(manifest['approved_runner_source_sha256']==[RUNNER_SHA],'Unapproved matched runner revision')
    base.require(manifest['matched_dependency_sha256']=={'campaign_dynamic_960_protocol.py':HELPER_SHA},'Unapproved matched helper revision')
    base.require(manifest['completed_proof_sha256']=={key:item['sha256'] for key,item in declaration['completed_proofs'].items()},'Completed proof linkage mismatch')
    cells=manifest['cells'];expected={cell['id']:cell for cell in declaration['new_cells']}
    base.require(len(cells)==8 and len({c['id'] for c in cells})==8 and set(c['id'] for c in cells)==set(expected),'Exactly eight declared matched cells required')
    old_paths={Path(c['output']).resolve() for c in old_cells};paths=[]
    for cell in cells:
        target=expected[cell['id']]
        base.require(cell['case']=='t0003' and type(cell['hz']) is int and cell['hz']==960,'Only declared TRAIN moving960 cells allowed')
        base.require(all(cell[key]==target[key] for key in ['kind','response','relative_tolerance','absolute_tolerance_N','translation_m']),'Matched cell tuple differs from declaration')
        path=Path(cell['output']).resolve();base.require(path not in old_paths,'Matched outputs must be fresh, not prior evidence');paths.append(path)
    base.require(len(set(paths))==8,'Matched outputs must be distinct')
    return cells


def verify_provenance(cell,declaration):
    protocol=base.read(Path(cell['output'])/'protocol.json')
    names={Path(p).name:h for p,h in protocol['source_input_dependency_and_gate_sha256'].items()}
    required={Path(guard.__file__).name:HELPER_SHA,Path(guard.PROTOCOL_PATH).name:guard.PROTOCOL_SHA256,
        'campaign_nonlinear_dynamics_case.py':RUNNER_SHA}
    required.update({Path(p).name:h for p,h in declaration['proof_source_sha256'].items()})
    required.update({Path(item['path']).name:item['sha256'] for item in declaration['completed_proofs'].values()})
    base.require(all(names.get(name)==wanted for name,wanted in required.items()),'Matched source/config/proof provenance mismatch')
    metadata=protocol['matched_960_protocol']
    base.require(metadata['protocol_sha256']==guard.PROTOCOL_SHA256 and metadata['study_cell_id']==cell['id']
        and metadata['kind']==cell['kind'] and metadata['physics_hz']==960 and metadata['episode_id']=='t0003'
        and metadata['response']==cell['response'] and metadata['relative_tolerance']==cell['relative_tolerance']
        and metadata['absolute_residual_floor_N']==cell['absolute_tolerance_N'] and metadata['translation_m']==cell['translation_m']
        and metadata['completed_proof_sha256']=={key:item['sha256'] for key,item in declaration['completed_proofs'].items()}
        and metadata['candidate_temporal_pass_verified'] is True and metadata['initial_temporal_failure_preserved'] is True
        and metadata['physical_model_unchanged'] is True and metadata['reference_qualification_implied'] is False,
        'Matched request metadata mismatch')
    base.require(protocol.get('supplemental_accuracy_protocol') is None and protocol.get('conditional_temporal_protocol') is None,'Mixed study roles rejected')


def compare_controls(loaded,reused):
    """Whole trajectories. Zero subtraction supplies only the common tolerance."""
    zero=reused[base.RESPONSES[0]];diagonal=reused[base.RESPONSES[1]]
    signal=base.coordinate_rmse(diagonal['positions_m'],zero['positions_m']);tolerance=max(2e-6,.05*signal)
    comparisons=[]
    for kind,ids,middle in [
        ('nonlinear_tolerance',['moving960-diagonal-rtol1e-5','moving960-diagonal-rtol1e-9'],diagonal),
        ('absolute_floor',['moving960-zero-floor1e-8','moving960-zero-floor1e-12'],zero)]:
        if all(key in loaded for key in ids):
            levels=[loaded[ids[0]],middle,loaded[ids[1]]]
            errors=[base.coordinate_rmse(levels[i+1]['positions_m'],levels[i]['positions_m']) for i in range(2)]
            comparisons.append({'kind':kind,'coordinate_rmse_m':errors,'tolerance_m':tolerance,'passed':bool(errors[-1]<tolerance)})
    for response,short in zip(base.RESPONSES,['zero','diagonal']):
        old=reused[response]
        for suffix,kind in [('translated','translated'),('fresh-reset','reset_replay')]:
            key=f'moving960-{short}-{suffix}'
            if key not in loaded:continue
            new=loaded[key];shift=np.array([0.,0.,-.25]) if kind=='translated' else np.zeros(3)
            error=base.coordinate_rmse(new['positions_m']-shift,old['positions_m'])
            record={'response':response,'kind':kind,'coordinate_rmse_m':error,'tolerance_m':tolerance,'passed':bool(error<tolerance)}
            if kind=='reset_replay':
                keys=['positions_m','velocities_point_chord_m_s','handle_centers_m','handle_rotations','force_step_commands_N']
                maximum=max(float(np.max(abs(new[k]-old[k]))) for k in keys)
                record.update(tolerance_m=base.REPLAY_ATOL,maximum_saved_array_absolute_error=maximum,
                    replay_arrays_bitwise={k:bool(np.array_equal(new[k],old[k])) for k in keys},passed=bool(maximum<=base.REPLAY_ATOL))
            comparisons.append(record)
    return comparisons,signal,tolerance


def run(baseline_manifest,supplemental_manifest,temporal_manifest,output,*,matched_manifest=None):
    output=Path(output);prior_output=output.with_name(output.stem+'-reaudited-prior50.json');temporal_output=output.with_name(output.stem+'-reaudited-temporal.json')
    base.require(not any(p.exists() for p in [output,prior_output,temporal_output]),'Fresh audit output paths required')
    base.require(base.sha(base.__file__)==temporal.BASE_AUDIT_SHA and base.sha(temporal.__file__)==TEMPORAL_AUDIT_SHA,'Published audit source changed')
    base.require(base.sha(guard.__file__)==HELPER_SHA and base.sha(guard.PROTOCOL_PATH)==guard.PROTOCOL_SHA256,'Matched helper/declaration revision differs')
    declaration=base.read(guard.PROTOCOL_PATH)
    for item in declaration['completed_proofs'].values():base.require(base.sha(item['path'])==item['sha256'],'Pinned completed proof changed')
    base.require(base.sha(baseline_manifest)==declaration['completed_proofs']['baseline_manifest']['sha256']
        and base.sha(temporal_manifest)==declaration['completed_proofs']['candidate_manifest']['sha256'],'Prior manifest linkage mismatch')
    prior=base.run(baseline_manifest,prior_output,supplemental_manifest=supplemental_manifest)
    candidate=temporal.run(baseline_manifest,temporal_output,temporal_manifest_path=temporal_manifest)
    baseline=base.read(baseline_manifest);floor=base.read(supplemental_manifest);temporal_cells=base.read(temporal_manifest)['cells']
    old_cells=baseline['cells']+floor['cells']+temporal_cells
    pinned_prior=base.read(declaration['completed_proofs']['floor_complete_audit']['path'])
    base.require({Path(c['output']).resolve() for c in baseline['cells']+floor['cells']}==
        {Path(c['output']).resolve() for c in pinned_prior['cells']},'Prior50 output lineage differs from pinned completed proof')
    base.require(len(old_cells)==52 and len({Path(c['output']).resolve() for c in old_cells})==52,'Exactly52 distinct prior evidence cells required')
    dependencies={Path(p).name:h for p,h in baseline['frozen_source_input_and_gate_sha256'].items()
        if Path(p).name in base.CORE+['isaac_dataset.py','boundary_block.py',base.INPUT_PATH.name]}
    gate=base.read(base.GATE_PATH);canonical=base.read(base.INPUT_PATH);results=[];loaded={};reused={}
    for cell in temporal_cells:
        report,data=base.audit_cell(cell,gate,canonical,{temporal.RUNNER_SHA},dependencies)
        base.require(report['passed'],'Previously verified temporal reuse is no longer valid')
        reused[cell['response']]=data
    if matched_manifest is not None:
        manifest=base.read(matched_manifest);cells=validate_manifest(manifest,declaration,old_cells)
        for cell in cells:
            # The old base verifier's absolute_floor role is specifically the
            # separate480 study. Validate this960 role below without pretending
            # its proof/config is that old study; all mechanics remain identical.
            mechanics_cell=dict(cell,kind='matched960_'+cell['kind'])
            report,data=base.audit_cell(mechanics_cell,gate,canonical,{RUNNER_SHA},dependencies)
            report['slot']=list(base.slot(cell));report['evidence_role']='matched960'
            if report['passed']:
                try:
                    verify_provenance(cell,declaration)
                    base.require(report['external_handle_work_independently_reconstructed_from_saved_step_positions'] is True,'Independent per-step handle work required')
                    steps=[json.loads(line) for line in (Path(cell['output'])/'steps.jsonl').read_text().splitlines()]
                    residuals=[row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'] for row in steps]
                    report['accepted_physical_residual_quantiles_N']=dict(zip(['minimum','p50','p95','maximum'],map(float,np.quantile(residuals,[0,.5,.95,1]))))
                except (base.AuditError,KeyError,TypeError,ValueError,OSError) as error:
                    report.update(passed=False,state='invalid',error=str(error));data=None
            results.append(report)
            if report['passed']:loaded[cell['id']]=data
    comparisons,signal,tolerance=compare_controls(loaded,reused)
    prior_except_old_failure=[c for c in prior['comparisons'] if not(c['case']=='t0003' and c['kind']=='temporal')]
    old_failure=[c for c in prior['comparisons'] if c['case']=='t0003' and c['kind']=='temporal']
    prior_ok=bool(len(prior['cells'])==50 and all(c['passed'] for c in prior['cells']) and len(prior_except_old_failure)==27
        and all(c['passed'] for c in prior_except_old_failure) and len(old_failure)==2 and all(c['passed'] is False for c in old_failure)
        and len(prior['analytic_static_zero_controls'])==3 and all(c['analytic_zero_rest_passed'] for c in prior['analytic_static_zero_controls']) and prior['mechanics']['passed'])
    qualified=bool(prior_ok and candidate['conditional_960_temporal_candidate_passed'] and len(results)==8 and all(c['passed'] for c in results)
        and len(comparisons)==6 and all(c['passed'] for c in comparisons))
    combined=prior['cells']+[c for c in candidate['cells'] if c['evidence_role']=='conditional_temporal']+results
    base.require(len({Path(c['output']).resolve() for c in combined})==len(combined),'Duplicate combined evidence paths')
    result={'audit_source_sha256':base.sha(__file__),'matched_protocol_sha256':guard.PROTOCOL_SHA256,
        'matched_manifest_sha256':base.sha(matched_manifest) if matched_manifest else None,
        'prior50_reaudit_path':str(prior_output),'prior50_reaudit_sha256':base.sha(prior_output),
        'temporal_reaudit_path':str(temporal_output),'temporal_reaudit_sha256':base.sha(temporal_output),
        'required_unique_cells':60,'indexed_unique_cells':len(combined),'verified_unique_cells':sum(c['passed'] for c in combined),
        'cells':combined,'matched960_comparisons':comparisons,'moving960_force_induced_coordinate_RMS_m':signal,'moving960_tolerance_m':tolerance,
        'original480_failed_temporal_comparisons_preserved':old_failure,'original_all480_gate_passed':False,
        'conditional960_temporal_candidate_passed':candidate['conditional_960_temporal_candidate_passed'],
        'matched960_operating_point_checks_passed':bool(len(results)==8 and all(c['passed'] for c in results) and len(comparisons)==6 and all(c['passed'] for c in comparisons)),
        'per_BC_small_force_operating_points_Hz':{'point_pin':480,'fixed_patch':480,'two_patches':480,'moving_patch':960},
        'independent_fixed_discretization_small_force_reference_qualified':qualified,
        'native_reference_qualified':False,'continuum_accuracy_established':False,'original_amplitude_dataset_qualified':False,
        'prediction_gate_passed':False,'uncertainty_calibrated':False,'active_policy_qualified':False,
        'qualification':'Only declared four TRAIN mechanics and zero/[.01,.01,.005]N pulse envelope, using stated per-BC rates. Original480 moving failure retained. Factorized checks do not establish exhaustive interactions, continuum/native equivalence, stronger force profiles or learned prediction.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline-manifest','supplemental-manifest','temporal-manifest','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--matched-manifest');a=p.parse_args()
    result=run(a.baseline_manifest,a.supplemental_manifest,a.temporal_manifest,a.output,matched_manifest=a.matched_manifest)
    print(json.dumps({key:result[key] for key in ['indexed_unique_cells','verified_unique_cells','independent_fixed_discretization_small_force_reference_qualified']}))
