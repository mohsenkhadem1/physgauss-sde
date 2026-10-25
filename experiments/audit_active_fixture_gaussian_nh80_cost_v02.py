"""Completed NH80 development audit from saved fields; no equilibrium replay."""
import json
import numpy as np
from experiments import run_active_fixture_gaussian_nh80_cost_v02 as stage


def classification(rows,spatial):
    physical=all(x['physical_passed'] for x in rows)
    eligible=all(x['load_eligible'] for x in rows)
    space=all(x['spatial_development_passed'] for x in spatial.values())
    return dict(status='COMPLETE_SIX_CONTROL_PASS_NO_BANK_ADMISSION' if physical and eligible and space else 'COMPLETE_NEGATIVE_NO_BANK_ADMISSION',physical_passed=physical,all_loads_eligible=eligible,spatial_development_passed=space,qualified_only_for_six_controls=bool(physical and eligible and space))


def audit():
    base=stage.LOCAL;digest=stage.sha(base/'protocol.json');p=stage.validate(digest)
    e=json.loads((base/'execution.json').read_text());r=json.loads((base/'result-LOCAL.json').read_text())
    if not e.get('terminal') or not e.get('completed') or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0 or e.get('failure') or not r.get('completed') or not r.get('all6_retained'):raise ValueError('Complete six-solve process barrier')
    if e['protocol_SHA256']!=digest or r['protocol_SHA256']!=digest or e['result']!=stage.record(base/'result-LOCAL.json') or r['release']!=stage.record(base/'release.json') or e['release']!=r['release']:raise ValueError('Protocol/release/result exact hashes')
    if e['external_cache']!=stage.record(stage.ROOT/e['external_cache']['path']) or (stage.ROOT/e['external_cache']['path']).read_bytes()!=(base/'result-LOCAL.json').read_bytes():raise ValueError('Immutable cache')
    stage.prior.shared.old.absent(e['controller_pid']);stage.prior.shared.old.absent(e['worker_pid'])
    expected={(n,label) for n in stage.LEVELS for label in stage.PILOT_LOADS}
    if len(r['rows'])!=6 or {(x['n'],x['label']) for x in r['rows']}!=expected:raise ValueError('All declared states retained')
    fields={};scalars=[]
    for row in r['rows']:
        if row['capture']!=stage.record(stage.ROOT/row['capture']['path']):raise ValueError('Capture hash')
        with np.load(stage.ROOT/row['capture']['path'],allow_pickle=False) as z:
            u=z['displacement_m'];rest=z['rest_m'];force=z['force_N'];fixed=z['exact_constraint_nodes'];query=z['query_displacement_m']
            if not all(np.isfinite(z[k]).all() for k in z.files) or np.any(u[fixed]!=0) or u.shape!=rest.shape or force.shape!=u.shape:raise ValueError('Capture finite/shapes/exact clamps')
            np.testing.assert_allclose(stage.bank.core.field_queries(row['n'],u,z['query_points_m']),query,rtol=0,atol=0)
        for k,v in stage.prior.force_wrench(rest,force,u).items():
            if v!=row['actual_deformed_wrench'][k]:raise ValueError('Actual known deadload wrench')
        status=stage.admissibility(row['deformation'],row['original_free_R_l2_N'],True,row['equilibrium'],row['stability'],row['label'])
        if any(row[k]!=v for k,v in status.items()):raise ValueError('Original physical/excitation classification')
        if row['layout']!=stage.PILOT_LAYOUT or (row['E_Pa'],row['K_N_m'])!=stage.PILOT_PARAMETERS:raise ValueError('Same support/material controls')
        fields[row['n'],row['label']]=u
        scalars.append(dict(n=row['n'],label=row['label'],**status,original_free_R_l2_N=row['original_free_R_l2_N'],minimum_sampled_J=row['deformation']['minimum_J'],maximum_symmetric_strain=row['deformation']['maximum_absolute_symmetric_strain_eigenvalue'],nonrigid_coordinate_RMS_m=row['deformation']['nonrigid_RMS_m'],nonrigid_fraction=row['deformation']['nonrigid_fraction'],force_balance_N=row['equilibrium']['force_balance_error_N'],moment_balance_Nm=row['equilibrium']['moment_balance_error_Nm'],equilibrium_passed=row['equilibrium']['passed'],stability_passed=row['stability']['passed'],Ritz_lower_endpoint=row['stability']['dimensionless_Ritz_lower_endpoint'],Newton_iterations=row['Newton']['iterations'],NH80_solve_s=row['NH80_solve_s'],checks_and_capture_s=row['independent_checks_and_capture_s'],capture=row['capture']))
    spatial=stage.spatial_results(fields)
    if spatial!=r['spatial']:raise ValueError('Saved-field spatial evidence')
    if r['passed']!=all(x['physical_passed'] for x in scalars) or e['physical_passed']!=r['passed'] or r['bank_admitted'] is not False:raise ValueError('Honest completed/physical-negative flags')
    compact=dict(schema='active-fixture-Gaussian-NH80-cost-compact-v02',**classification(scalars,spatial),completed=True,bank_admitted=False,protocol=stage.record(base/'protocol.json'),source_result=stage.record(base/'result-LOCAL.json'),execution=stage.record(base/'execution.json'),rows=scalars,spatial=spatial,wall_s=e['wall_s'],peak_RSS_bytes=r['peak_RSS_bytes'],scope='Six declared Gaussian-load DEVELOPMENT controls only. Original5%/20um thresholds retained; physical/excitation/spatial flags separate. Generalized stability is numerical Ritz evidence; J sampled only. Even a positive six-case pilot does not admit the full4layout/20parameterstate/11load bank. Old narrow-load negative remains unchanged. Raw fields remainLOCAL.')
    stage.write(base/'compact-summary.json',compact)
    review=dict(schema='active-fixture-Gaussian-NH80-cost-independent-review-v02',audit_passed=True,scientific_admission=False,source_SHA256=p['source_SHA256'],audit_source=stage.record(stage.ROOT/'experiments/audit_active_fixture_gaussian_nh80_cost_v02.py'),source_result=stage.record(base/'result-LOCAL.json'),compact_summary=stage.record(base/'compact-summary.json'),checks=dict(all6_capture_hashes=True,same_frozen_source_protocol_release=True,immutable_external_cache=True,PID_and_process_groups_absent=True,finite_exact_clamps=True,query_fields_and_deformed_wrenches_reproduced=True,original_physical_flags_reproduced=True,common_positive_volume_RMS_and_public_points_reproduced=True,all_failures_retained=True,no_model_solve_RNG_or_PDE_replay=True))
    stage.write(base/'independent-review.json',review)
    files=[stage.SOURCE,stage.TEST,'experiments/campaign_active_fixture_gaussian_load_v02.py','tests/test_active_fixture_gaussian_load_v02.py','experiments/audit_active_fixture_gaussian_nh80_cost_v02.py','tests/test_audit_active_fixture_gaussian_nh80_cost_v02.py']+[str((base/name).relative_to(stage.ROOT)) for name in ('protocol.json','release.json','execution.json','compact-summary.json','independent-review.json')]
    records=[dict(stage.record(stage.ROOT/f),bytes=(stage.ROOT/f).stat().st_size) for f in files]
    stage.write(base/'completed-publication-inventory.json',dict(schema='compact-publication-inventory-v01',files=records,count=len(records),total_bytes=sum(x['bytes'] for x in records),excluded=['result-LOCAL.json','raw-LOCAL/*.npz','external raw metadata cache','worker.log'],scope='Source/tests and compact scalar/hash evidence only; no field arrays.'))
    print(json.dumps(dict(inventory=stage.record(base/'completed-publication-inventory.json'),count=len(records),classification=classification(scalars,spatial),spatial=spatial)))

if __name__=='__main__':audit()
