"""Reconstructible SOURCE/DESIGN-only full-bank cost proposal; NEVER a launcher.

Reads only the sealed six-control NH result/audit for timing and hash evidence.
Execution remains HELD pending new36-case policy-development adequacy and root
source/protocol review. Existing six fields can be reused, never rerun quietly.
"""
import json
import numpy as np
from experiments import run_active_fixture_gaussian_refinement_v03 as known
ROOT=known.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-Gaussian-fullNH-cost-proposal-v01'


def catalog():
    return [dict(id=f'{group["id"]}__n{n}__{label}',group=group['id'],layout=group['layout'],E_Pa=group['E_Pa'],K_N_m=group['K_N_m'],n=n,label=label) for group in known.catalog() for n in (3,6,12) for label in [s['id'] for s in known.bank.load_specs()]]


def costs(result):
    means={};maximum={};initialization={}
    for n in (3,6,12):
        rows=[r for r in result['rows'] if r['n']==n]
        if len(rows)!=2:raise ValueError('Both fixed cost controls per mesh required')
        times=[sum(r[k] for k in ('NH80_solve_s','linear_warm_s','independent_checks_and_capture_s')) for r in rows]
        if not np.isfinite(times).all() or min(times)<0:raise ValueError('Finite nonnegative measured costs')
        means[str(n)]=float(np.mean(times));maximum[str(n)]=max(times);initialization[str(n)]=result['meshes'][str(n)]['initialization_s']
    # Sixty factors, NOT660: each parameter/layout/mesh factor serves all11loads.
    base=20*sum(initialization.values());payload=20*11*sum(means.values());upper=20*11*sum(maximum.values())
    reuse=sum(sum(r[k] for k in ('NH80_solve_s','linear_warm_s','independent_checks_and_capture_s')) for r in result['rows'])
    return dict(measured_perload_mean_s=means,measured_perload_slower_control_s=maximum,measured_initialization_s=initialization,parameter_layout_groups=20,meshes_per_group=3,loads_per_group_mesh=11,new_rest_factors=60,full_fields=660,reused_fields=6,new_fields=654,estimated_initialization_s=base,estimated_full_payload_s=payload,measured_reused_payload_credit_s=reuse,estimated_new_serial_s=base+payload-reuse,estimated_new_serial_hours=(base+payload-reuse)/3600,slower_control_serial_s=base+upper-reuse,medium_only_all220_n3_n6_serial_s=20*sum(initialization[str(n)]+11*means[str(n)] for n in (3,6)),fourworker_ideal_mean_wall_s=(base+payload-reuse)/4,scope='Conditional timing extrapolation from two loads/one parameter/layout. Additional Newton/stability/fill/commonRMS/scheduler cost may differ. No factor-reuse credit for dead childLU objects; no assumedGPU PDEspeedup.')


def propose():
    base=known.LOCAL;known.validate(known.sha(base/'protocol.json'));e=json.loads((base/'execution.json').read_text())
    if not e.get('terminal') or not e.get('completed') or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0:raise ValueError('Closed measured origin required')
    known.prior.shared.old.absent(e['controller_pid']);known.prior.shared.old.absent(e['worker_pid'])
    if e['result']!=known.record(base/'result-LOCAL.json') or e['result']['sha256']!='dfddef7f1e86b259b7c15ed535b09e549b7ae0dd0e3ce621b0e3684bc3329894':raise ValueError('Exact positive six-control result')
    r=json.loads((base/'result-LOCAL.json').read_text());a=json.loads((base/'independent-review.json').read_text())
    if a.get('audit_passed') is not True or a.get('scientific_admission') is not False or a['source_result']!=e['result']:raise ValueError('Independent six-only audit')
    cases=catalog();raw_sizes=[]
    for row in r['rows']:
        if row['capture']!=known.record(ROOT/row['capture']['path']):raise ValueError('Exact reused-field hash')
        raw_sizes.append((row['n'],(ROOT/row['capture']['path']).stat().st_size))
    estimated_full_raw=sum(np.mean([b for n,b in raw_sizes if n==level]) for level in (3,6,12))*20*11
    canonical=json.dumps(cases,sort_keys=True).encode();import hashlib
    proposal=dict(schema='fixture-Gaussian-fullNH-cost-proposal-v01',status='DESIGN_ONLY_HELD_NO_EXECUTION_AUTHORITY',execution_authorized=False,prerequisites=['NEW36case Gaussian linearpolicy development must independently demonstrate useful nonuniversal choices; mechanicalseparation alone is insufficient','Root-reviewed frozen fullcatalog source/protocol/resources','No unresolved originalphysical/mesh/force/gate changes'],origin=dict(protocol=known.record(base/'protocol.json'),execution=known.record(base/'execution.json'),result=e['result'],audit=known.record(base/'independent-review.json')),catalog_recipe=dict(function='propose_fixture_gaussian_fullNH_cost_v01.catalog',sha256=hashlib.sha256(canonical).hexdigest(),cases=660,groups=20,layouts=[s['id'] for s in known.bank.LAYOUTS],parameters=[list(x) for x in known.PARAMETERS],levels=[3,6,12],loads=[s['id'] for s in known.bank.load_specs()],all_cases_retained=True,no_symmetry_or_E_scaling_substitution=True),cost_projection=costs(r),prospective_resources=dict(workers=4,worker_RSS_cap_bytes=18*1024**3,controller_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=76*1024**3,host_free_reserve_bytes=30*1024**3,minimum_initial_MemAvailable_bytes=106*1024**3,measured_one_worker_peak_bytes=r['peak_RSS_bytes'],worker_margin_ratio=18*1024**3/r['peak_RSS_bytes'],GPU_PDE_acceleration_assumed=False,per_parameter_layout_group_wall_proposal_s=5400,global_wall_proposal_s=36000,CPU_assignments='Root must freeze disjoint allowed cores after currentjobs close; one-threadworkers',resources_not_yet_frozen_for_execution=True),storage=dict(estimated_full_raw_bytes=estimated_full_raw,projected_write_reserve_bytes=8*1024**3,filesystem_free_reserve_bytes=30*1024**3,raw_LOCAL_only=True,publish_source_protocol_compact_scalars_only=True),reuse=dict(fields=6,records=[x['capture'] for x in r['rows']],same_parameters_layout_loads_geometry_rule=True,old_fields_never_overwritten=True,factors_not_persisted=True),gates=dict(known.metrics.GATES),scope='All660 deterministic DEVELOPMENTrun plan only. Six-control spatial pass doesNOT admit otherlayouts/EK/loads. SampledJ and numericalRitz stability retain existing scope; conditionalp>=1 spatial indicators are not rigorous continuum bounds. Everynegative retained; no adaptiveforce tuning, casefilter, posterior, NN or calibration claim.')
    known.write(LOCAL/'cost-proposal.json',proposal)
    print(json.dumps(dict(proposal=known.record(LOCAL/'cost-proposal.json'),serial_hours=proposal['cost_projection']['estimated_new_serial_hours'],four_worker_ideal_s=proposal['cost_projection']['fourworker_ideal_mean_wall_s'],new_fields=654,execution_authorized=False)))
    return proposal

if __name__=='__main__':propose()
