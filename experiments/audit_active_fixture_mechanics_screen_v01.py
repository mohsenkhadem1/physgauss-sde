"""Read-only completed deterministic development audit; no mechanics replay."""
import json
from collections import Counter
import numpy as np
from experiments import run_active_fixture_mechanics_screen_v01 as pilot


def audit():
    gate=pilot.completed_screen_gate(pilot.sha(pilot.LOCAL/'protocol.json'))
    r=json.loads((pilot.LOCAL/'result-LOCAL.json').read_text())
    with np.load(pilot.ROOT/r['raw']['path'],allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    expected={'responses':(72,60),'jacobians':(72,60,2),'local_risk_reductions':(72,8),'field_queries':(72,11,16,3)}
    for name,shape in expected.items():
        if arrays[name].shape!=shape or not np.isfinite(arrays[name]).all():raise ValueError('Missing/nonfinite response '+name)
    if list(arrays['measurement_labels'])!=[x['id'] for x in pilot.bank.measurement_specs()] or list(arrays['load_labels'])!=[x['id'] for x in pilot.bank.load_specs()]:raise ValueError('Public label order')
    for i,row in enumerate(r['rows']):
        if (arrays['n'][i],arrays['layout'][i],arrays['E'][i],arrays['K'][i])!=(row['n'],row['layout'],row['E_Pa'],row['K_N_m']):raise ValueError('Raw row identity')
    delta=[]
    for i,row in enumerate(r['coarse_fine']):
        q=float(np.max(abs(arrays['field_queries'][i]-arrays['field_queries'][36+i])))
        m=float(np.max(abs(arrays['responses'][i]-arrays['responses'][36+i])))
        if q!=row['maximum_common_point_difference_m'] or m!=row['maximum_measurement_difference_m']:raise ValueError('Coarse/fine evidence mismatch')
        delta.append(q)
    separation=[]
    for n in (1,2):
        for E in pilot.E_VALUES:
            for K in pilot.K_VALUES:
                ids=np.flatnonzero((arrays['n']==n)&(arrays['E']==E)&(arrays['K']==K))
                for a,i in enumerate(ids):
                    for j in ids[a+1:]:
                        d=arrays['responses'][i]-arrays['responses'][j]
                        separation.append((n,float(np.linalg.norm(d[:4])/pilot.bank.SIGMA_M),float(np.sqrt(np.mean(d[4:12]**2))),float(np.sqrt(np.mean(d[12:]**2)))))
    for actual,stored in zip(separation,r['layout_separation']):
        if actual!=(stored['n'],stored['prefix_L2_over_sigma'],stored['candidate_RMS_difference_m'],stored['held_target_RMS_difference_m']):raise ValueError('Layout separation mismatch')
    n2=[v for v in r['rows'] if v['n']==2]
    nr=np.asarray([v['nonrigid_RMS_m'][1:9] for v in n2]);sp=[v for v in separation if v[0]==2]
    stats=dict(states_retained=72,loads_per_state=11,responses_retained=792,maximum_original_free_component_residual_N=max(v['free_residual_N'] for v in r['rows']),maximum_modal_parity_m=max(v['modal_parity_m'] for v in r['rows']),exact_clamps=True,maximum_sampled_linear_symmetric_strain=max(max(v['maximum_symmetric_strain']) for v in r['rows']),minimum_sampled_linear_J=min(min(v['minimum_sampled_J']) for v in r['rows']),n2_candidate_nonrigid_RMS_range_m=[float(nr.min()),float(nr.max())],n1_to_n2_maximum_common_coordinate_difference_m=max(delta),spatial_20um_gate_passed=bool(max(delta)<=20e-6),n2_prefix_layout_L2_difference_range_m=[min(v[1] for v in sp)*pilot.bank.SIGMA_M,max(v[1] for v in sp)*pilot.bank.SIGMA_M],n2_held_layout_RMS_difference_range_m=[min(v[3] for v in sp),max(v[3] for v in sp)],n2_local_surrogate_winner_counts=dict(Counter(v['development_local_risk_winner'] for v in n2)),maximum_support16_vs20_relative_Frobenius=max(v['support16_vs20_relative_Frobenius'] for m in r['meshes'].values() for v in m['layouts'].values()),all_support_positive_quadrature=True,new_physics_replayed=False)
    public=dict(schema='active-fixture-linear-development-compact-v01',status='NUMERICAL_IMPLEMENTATION_PASS_SPATIAL_UNQUALIFIED',completed=True,source_result=gate['result'],raw_hash_binding=gate['raw'],execution=gate['execution'],protocol=gate['protocol'],release=gate['release'],statistics=stats,scope='Development LINEAR discrete-model evidence only. The 799um n1-to-n2 discrepancy fails the unchanged20um displacement criterion. No NH, continuum, policy, posterior or calibration admission. Local surrogate winners are not categorical Bayes policy outcomes. Weak prefix is an intentional sensor control.',raw_arrays_published=False)
    pilot.write(pilot.LOCAL/'compact-summary.json',public)
    review=dict(schema='active-fixture-mechanical-pilot-independent-review-v01',passed=True,source_result=gate['result'],compact_summary=pilot.record(pilot.LOCAL/'compact-summary.json'),source_SHA256=gate['source_SHA256'],audit_source=pilot.record(pilot.ROOT/'experiments/audit_active_fixture_mechanics_screen_v01.py'),checks=dict(protocol_release_source_cache_raw_hashes=True,all72_Cartesian_states=True,all792_responses_finite=True,raw_label_identity=True,all36_spatial_differences_reproduced=True,all108_layout_separations_reproduced=True,original_free_residual_modal_clamps=True,controller_worker_PID_and_groups_absent=True,no_model_solve_RNG_or_observation=True),scope='Independent numerical/data-binding audit passes; spatial accuracy remains failed. No new numerical admission.')
    pilot.write(pilot.LOCAL/'independent-review.json',review)
    files=['experiments/campaign_active_fixture_bank_v01.py','tests/test_active_fixture_bank_v01.py',pilot.SOURCE,pilot.TEST,'experiments/audit_active_fixture_mechanics_screen_v01.py']+[str((pilot.LOCAL/name).relative_to(pilot.ROOT)) for name in ('protocol.json','release.json','execution.json','compact-summary.json','independent-review.json','initial-manufactured-roundoff-FAIL.txt')]
    records=[dict(pilot.record(pilot.ROOT/f),bytes=(pilot.ROOT/f).stat().st_size) for f in files]
    inventory=dict(schema='compact-publication-inventory-v01',files=records,count=len(records),total_bytes=sum(v['bytes'] for v in records),excluded=['result-LOCAL.json','responses-LOCAL.npz','worker.log','external raw result cache'],scope='Source/tests and compact scalar/provenance text only; no raw response arrays.')
    pilot.write(pilot.LOCAL/'completed-publication-inventory.json',inventory)
    print(json.dumps(dict(summary=stats,inventory=pilot.record(pilot.LOCAL/'completed-publication-inventory.json'),count=len(records),bytes=inventory['total_bytes'])))

if __name__=='__main__':audit()
