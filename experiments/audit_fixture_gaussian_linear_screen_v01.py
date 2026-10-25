"""Prospective completed72-row audit; no raw reads before terminal closure.

Reproduces measured separation/mesh diagnostics from saved responses, without
new FE/modal/observation/model calls. Physical full-field diagnostics are bound
producer witnesses, not independently reconstructed when fullu is not saved.
"""
import json
import numpy as np
from experiments import run_fixture_gaussian_linear_screen_v01 as stage


def metadata_barrier():
    base=stage.LOCAL;digest=stage.sha(base/'protocol.json');p=stage.validate(digest)
    e=json.loads((base/'execution.json').read_text())
    if not e.get('terminal') or not e.get('completed') or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0 or e.get('failure') or e['protocol_SHA256']!=digest:raise ValueError('Complete clean process barrier before result/NPZ access')
    stage.origin.prior.shared.old.absent(e['controller_pid']);stage.origin.prior.shared.old.absent(e['worker_pid'])
    path=base/'result-LOCAL.json'
    if e['result']!=stage.record(path):raise ValueError('Terminal result hash')
    r=json.loads(path.read_text())
    if not r.get('completed') or not r.get('passed') or not r.get('all72_retained') or not r.get('all792_load_responses_retained') or r['completed_states']!=72 or r['completed_load_responses']!=792 or r['protocol_SHA256']!=digest or r['release']!=stage.record(base/'release.json') or e['release']!=r['release']:raise ValueError('Every declared state/response before arrays')
    if len(r['queue'])!=72 or any(x['status']!='completed' for x in r['queue']) or [{k:x[k] for k in ('index','layout','E_Pa','K_N_m','n')} for x in r['queue']]!=p['plan']['worklist']:raise ValueError('Immutable complete72worklist, no replacement/drop')
    if e['external_cache']!=stage.record(stage.ROOT/e['external_cache']['path']) or (stage.ROOT/e['external_cache']['path']).read_bytes()!=path.read_bytes() or r['raw']!=stage.record(stage.ROOT/r['raw']['path']):raise ValueError('Exact cache/raw byte barriers')
    if len(r['rows'])!=72 or len(r['models'])!=8 or {(x['n'],x['layout']) for x in r['models']}!={(n,s['id']) for n in (3,6) for s in stage.mechanics.bank.LAYOUTS}:raise ValueError('Exact eight fullmode models')
    for model in r['models']:
        d=model['all_modes']
        if d['all_boundary_modes_retained'] is not True or d['interior_basis_truncated'] is not False or d['foundation_threshold_applied'] is not False or d['exact_particular_1_over_E_term'] is not True:raise ValueError('All modes AND exact1/E particular retained')
    return p,e,r


def summarize(raw_rows,producer_rows):
    separation=stage.mechanics.separation(raw_rows);pairs=[]
    for a,b in zip(raw_rows[::2],raw_rows[1::2]):
        pairs.append(dict(layout=a['layout'],E_Pa=a['E_Pa'],K_N_m=a['K_N_m'],maximum_common_point_coordinate_difference_m=float(np.max(abs(a['field_queries']-b['field_queries']))),maximum_measurement_difference_m=float(np.max(abs(a['response']-b['response'])))))
    bymesh={}
    for n in (3,6):
        p=[r for r in producer_rows if r['n']==n];s=[r for r in separation if r['n']==n]
        prefix=[r['prefix4_RMS_m'] for r in s];target=[r['held48_RMS_m'] for r in s];candidate=[r['candidate8_RMS_m'] for r in s]
        nr=np.array([r['nonrigid_coordinate_RMS_m'] for r in p]);strain=np.array([r['maximum_linear_symmetric_strain'] for r in p]);J=np.array([r['minimum_sampled_linear_J'] for r in p])
        bymesh[str(n)]=dict(states=36,loads=396,prefix4_layout_RMS_range_m=[min(prefix),max(prefix)],candidate8_layout_RMS_range_m=[min(candidate),max(candidate)],held48_layout_RMS_range_m=[min(target),max(target)],layout_pairs=54,prefix_pairs_above20um=sum(x>20e-6 for x in prefix),held_pairs_above20um=sum(x>20e-6 for x in target),candidate_coordinate_nonrigid_RMS_range_m=[float(nr[:,1:9].min()),float(nr[:,1:9].max())],prefix_coordinate_nonrigid_RMS_range_m=[float(nr[:,0].min()),float(nr[:,0].max())],maximum_sampled_linear_symmetric_strain=float(strain.max()),minimum_sampled_linear_J=float(J.min()))
    return separation,pairs,bymesh


def audit():
    p,e,r=metadata_barrier()
    with np.load(stage.ROOT/r['raw']['path'],allow_pickle=False) as z:a={name:z[name] for name in z.files}
    for name,shape in dict(responses=(72,60),jacobians=(72,60,2),field_queries=(72,11,16,3)).items():
        if a[name].shape!=shape or not np.isfinite(a[name]).all():raise ValueError('All finite raw responses/jacobians/queryfields')
    if list(a['measurement_labels'])!=[x['id'] for x in stage.mechanics.bank.measurement_specs()] or list(a['load_labels'])!=[x['id'] for x in stage.mechanics.bank.load_specs()] or not np.array_equal(a['query_points'],stage.mechanics.bank.measurement_points()):raise ValueError('Exact public observation/load/query ABI')
    raw=[]
    for i,(row,planned) in enumerate(zip(r['rows'],p['plan']['worklist'])):
        if {k:row[k] for k in ('index','layout','E_Pa','K_N_m','n')}!=planned or (a['layout'][i],a['E'][i],a['K'][i],a['n'][i])!=(row['layout'],row['E_Pa'],row['K_N_m'],row['n']):raise ValueError('Exact row identities')
        if row['free_residual_l2_N']>1e-10 or row['modal_parity_m']>2e-12 or row['exact_clamp_zeros'] is not True or len(row['equilibrium'])!=11 or not all(x['passed'] for x in row['equilibrium']):raise ValueError('Original numerical/equilibrium witnesses')
        np.testing.assert_allclose(np.array(row['nonrigid_Euclidean_RMS_m'])/np.sqrt(3),row['nonrigid_coordinate_RMS_m'],rtol=0,atol=0)
        raw.append(dict(planned,response=a['responses'][i],field_queries=a['field_queries'][i]))
    sep,pairs,by_mesh=summarize(raw,r['rows'])
    if sep!=r['layout_separation'] or pairs!=r['paired_mesh_differences']:raise ValueError('Independent measured separation/mesh reproduction')
    c=r['first_fixed_paired_case_cost_checkpoint']
    if c['indices']!=[0,1] or c['remaining_states']!=70 or c['queue_unchanged'] is not True:raise ValueError('Cost checkpoint retained same all72queue')
    compact=dict(schema='fixture-Gaussian-linear-development-compact-v01',completed=True,numerical_audit_passed=True,bank_admitted=False,states_retained=72,load_responses_retained=792,source_result=stage.record(stage.LOCAL/'result-LOCAL.json'),protocol=stage.record(stage.LOCAL/'protocol.json'),execution=stage.record(stage.LOCAL/'execution.json'),raw_HASH_ONLY=r['raw'],by_mesh=by_mesh,maximum_common_point_n3_n6_coordinate_difference_m=max(x['maximum_common_point_coordinate_difference_m'] for x in pairs),maximum_modal_direct_parity_m=max(x['modal_parity_m'] for x in r['rows']),maximum_original_free_R_l2_N=max(x['free_residual_l2_N'] for x in r['rows']),first_fixed_paired_timing=c,wall_s=e['wall_s'],peak_RSS_bytes=r['peak_RSS_bytes'],scope='All72 fixed LINEAR development states/792responses only. Separation and mesh differences reproduced; producerNR/strain/equilibrium witnesses hash-bound. No NH/continuum/inverse/policy/calibration admission. No sampledcasefilter or replacements. Rawresponses/jacobians/queryfields remainLOCAL.')
    stage.write(stage.LOCAL/'compact-summary.json',compact)
    review=dict(schema='fixture-Gaussian-linear-independent-audit-v01',audit_passed=True,scientific_admission=False,source_result=stage.record(stage.LOCAL/'result-LOCAL.json'),compact_summary=stage.record(stage.LOCAL/'compact-summary.json'),source_SHA256=p['source_SHA256'],audit_source=stage.record(stage.ROOT/'experiments/audit_fixture_gaussian_linear_screen_v01.py'),checks=dict(source_release_raw_cache_exact=True,owned_PID_and_groups_absent=True,all72_cartesian_states_and792_finite_responses=True,all8_exactmode_models_and1overE=True,all108_support_separations_reproduced=True,all36_mesh_differences_reproduced=True,Euclidean_vs_coordinate_NR_conversion_reproduced=True,no_FE_modal_PDE_RNG_or_observation_replay=True),full_nodal_fields_not_saved=True,scope='Numerical/data-binding independent audit, not independent full-fieldstrain reconstruction or physicaladmission.')
    stage.write(stage.LOCAL/'independent-review.json',review)
    files=[stage.SOURCE,stage.TEST,'experiments/campaign_fixture_gaussian_linear_screen_v01.py','tests/test_fixture_gaussian_linear_screen_v01.py','experiments/audit_fixture_gaussian_linear_screen_v01.py','tests/test_audit_fixture_gaussian_linear_screen_v01.py']+[str((stage.LOCAL/name).relative_to(stage.ROOT)) for name in ('protocol.json','release.json','execution.json','compact-summary.json','independent-review.json')]
    records=[dict(stage.record(stage.ROOT/f),bytes=(stage.ROOT/f).stat().st_size) for f in files]
    stage.write(stage.LOCAL/'completed-publication-inventory.json',dict(schema='compact-development-publication-inventory-v01',files=records,count=len(records),total_bytes=sum(x['bytes'] for x in records),excluded=['result-LOCAL.json','responses-LOCAL.npz','externalrawcache','worker.log'],scope='Source/test+compactscalar/hash evidence, no rawphysicalresponse arrays.'))
    print(json.dumps(dict(inventory=stage.record(stage.LOCAL/'completed-publication-inventory.json'),summary=compact)))

if __name__=='__main__':audit()
