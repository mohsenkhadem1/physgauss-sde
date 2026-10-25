"""Additional SOURCE-only checkpoint audit, outside frozen physics graph.

Root must externally reap the checkpoint phase and prepin this source/test
before invoking. No PDE, factorization, RNG, force array publication or old
source mutation. The final frozen264-field audit is still mandatory.
"""
import argparse,json,time
from experiments import audit_fixture_balanced_nh_catalog_v01 as frozen
stage=frozen.stage
SOURCE='experiments/audit_fixture_balanced_nh_checkpoint_v01.py'
TEST='tests/test_audit_fixture_balanced_nh_checkpoint_v01.py'


def audit(protocol_sha,source_sha,test_sha):
    start=time.monotonic();frozen.guard(start)
    if stage.sha(stage.ROOT/SOURCE)!=source_sha or stage.sha(stage.ROOT/TEST)!=test_sha:raise ValueError('Exact root-prepinned checkpoint auditprocedure bytes')
    output=stage.LOCAL/'checkpoint-independent-review.json'
    if output.exists():raise FileExistsError('No checkpoint independent-audit overwrite')
    p=stage.validate(protocol_sha)
    phase,groups=frozen.phase_metadata('checkpoint',p,protocol_sha)
    # phase_metadata verifies rootclosure and EVERY33 capture/context/hash
    # before this wrapper or its shared validators may open the first NPZ.
    if len(groups)!=1 or groups[0]['index']!=0 or groups[0]['value']['group']!=p['plan']['groups'][0]:raise ValueError('Only fixed originalgroup0 checkpoint')
    item=groups[0];group=item['value'];fields={};rows=[]
    for row in group['rows']:
        frozen.guard(start);u,d=frozen.field_audit.check_capture(row);frozen.guard(start)
        status=frozen.field_status(row,d);fields[row['n'],row['label']]=u
        rows.append(dict(n=row['n'],label=row['label'],completed=True,**status))
    spatial=frozen.spatial_results(fields);frozen.guard(start)
    if spatial!=group['spatial']:raise ValueError('Independent checkpoint all11 spatial evidence')
    flags=frozen.group_classification(rows,spatial,{n:v['surface']for n,v in group['meshes'].items()})
    if any(group[k]!=value for k,value in flags.items()):raise ValueError('Independent checkpoint physical/reference/excitation classification')
    result=dict(schema='fixture-balanced-nh-checkpoint-independent-v01',audit_passed=True,group_index=0,all33_retained=flags['all33_retained'],all11_reference_valid=flags['all11_reference_valid'],all_fields_excitation_goal_met=flags['all_fields_excitation_goal_met'],bank_admitted=False,source_SHA256={SOURCE:source_sha,TEST:test_sha},source_physical_SHA256=p['source_SHA256'],protocol=stage.record(stage.LOCAL/'protocol.json'),release=stage.record(stage.LOCAL/'release.json'),phase=phase,group_result=item['result'],captures=[dict(n=r['n'],label=r['label'],capture=r['capture'])for r in group['rows']],reference_flags=flags['references'],spatial=spatial,audit_resources=frozen.AUDIT_LIMITS,audit_wall_s=time.monotonic()-start,checks=dict(rootreap_and_all33_hashes_before_NPZ=True,geometry_knownload_query_wrench_sampled_deformation_reproduced=True,original_scalar_physical_and_Ritz_predicates=True,all11_common_volume_and_publicquery_spatial_reproduced=True,legacy_excitation_separate_from_referencevalid=True,no_PDE_factor_RNG_or_old_source_mutation=True),scope='Additional firstgroup33-field DEVELOPMENT integrity/reference diagnostic only. Does not replace final264-field audit, authorize remaining work, change frozen releasegate, or admit a general bank/policy/calibration claim. Only compact scalars and hash/path commitments; raw arrays LOCAL.')
    frozen.guard(start);stage.write(output,result)
    return stage.record(output)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol-sha',required=True);ap.add_argument('--source-sha',required=True);ap.add_argument('--test-sha',required=True);a=ap.parse_args()
    print(json.dumps(audit(a.protocol_sha,a.source_sha,a.test_sha)))
