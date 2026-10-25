"""Closed LINEAR72 excitation-planning diagnostic; no solves or gate changes.

Coordinate RMS is compared with the unchanged100um goal. Nonrigid fraction
was NOT retained by this producer, so it is neither inferred nor imputed.
"""
import json
from collections import Counter
import numpy as np
from experiments import audit_fixture_gaussian_linear_screen_v01 as closed
m=closed.stage
GOAL_M=100e-6
FUTURE_PARAMETERS={(2000.,5.),(2000.,30.),(8000.,5.),(8000.,30.),(4000.,float(np.sqrt(150.)))}


def role(j):return 'prefix' if j==0 else 'candidate' if j<9 else 'held'


def summary(rows):
    labels=[s['id'] for s in m.mechanics.bank.load_specs()];answer={}
    for n in (3,6):
        selected=[r for r in rows if r['n']==n]
        if len(selected)!=36:raise ValueError('All36 fixed parameter/layout states required permesh')
        low=[]
        for r in selected:
            values=np.asarray(r['nonrigid_coordinate_RMS_m'])
            if values.shape!=(11,) or not np.isfinite(values).all() or np.any(values<0):raise ValueError('All11 finite coordinateNR values required')
            np.testing.assert_allclose(np.asarray(r['nonrigid_Euclidean_RMS_m'])/np.sqrt(3),values,rtol=0,atol=0)
            low.extend((r,j) for j,v in enumerate(values) if v<GOAL_M)
        candidates=[(r,j) for r,j in low if role(j)=='candidate'];future=[r for r in selected if (r['E_Pa'],r['K_N_m']) in FUTURE_PARAMETERS]
        def counts(items):return dict(Counter(role(j) for r,j in items))
        f_low=[(r,j) for r,j in low if (r['E_Pa'],r['K_N_m']) in FUTURE_PARAMETERS]
        answer[str(n)]=dict(fixed_states=36,total_load_responses=396,total_candidate_responses=288,below100um_total=len(low),below100um_by_role=counts(low),candidate_below100um=len(candidates),states_with_low_candidate=len({r['index'] for r,j in candidates}),candidate_low_by_E_Pa={str(E):sum(r['E_Pa']==E for r,j in candidates) for E in (2000.,4000.,8000.)},candidate_low_by_K_N_m={str(K):sum(r['K_N_m']==K for r,j in candidates) for K in (5.,float(np.sqrt(150.)),30.)},candidate_low_per_public_load={label:sum(labels[j]==label for r,j in candidates) for label in labels[1:9]},candidate_low_coordinate_RMS_range_m=[min(r['nonrigid_coordinate_RMS_m'][j] for r,j in candidates),max(r['nonrigid_coordinate_RMS_m'][j] for r,j in candidates)],candidate_low_witnesses=dict(minimum_sampled_linear_J=min(r['minimum_sampled_linear_J'][j] for r,j in candidates),maximum_linear_symmetric_strain=max(r['maximum_linear_symmetric_strain'][j] for r,j in candidates),all_original_linear_equilibrium_passed=all(r['equilibrium'][j]['passed'] for r,j in candidates),all_exact_clamps=all(r['exact_clamp_zeros'] for r,j in candidates),maximum_free_R_l2_N=max(r['free_residual_l2_N'] for r,j in candidates),physical_NH_admission=False),mapped_five_parameter_catalog=dict(parameter_layout_states=len(future),total_candidate_responses=8*len(future),below100um_by_role=counts(f_low),scope='Only matching LINEAR3/6rows of future20statecatalog, not a prediction/proof of unrunNH12fields.'),fraction_status='NOT_RETAINED',fraction_goal_evaluated=False,fraction_goal_unchanged=.3,nonrigid_fraction_imputed=False)
    return answer


def run():
    p,e,r=closed.metadata_barrier()
    review=json.loads((m.LOCAL/'independent-review.json').read_text())
    if review.get('audit_passed') is not True or review.get('scientific_admission') is not False or review['source_result']!=e['result']:raise ValueError('Completed independent LINEAR72 audit required')
    data=summary(r['rows'])
    keys=lambda n:{(x['layout'],x['E_Pa'],x['K_N_m'],j) for x in r['rows'] if x['n']==n for j in range(11) if x['nonrigid_coordinate_RMS_m'][j]<GOAL_M}
    out=dict(schema='fixture-Gaussian-excitation-design-issue-v01',source_result=e['result'],source_execution=m.record(m.LOCAL/'execution.json'),independent_audit=m.record(m.LOCAL/'independent-review.json'),source=m.record(m.ROOT/'experiments/diagnose_fixture_gaussian_excitation_v01.py'),absolute_coordinate_nonrigid_goal_m=GOAL_M,unchanged_threshold=True,by_mesh=data,below_threshold_membership_identical_n3_n6=keys(3)==keys(6),all72_original_states_retained=True,no_case_filter_or_gate_change=True,new_solves=0,planning_issue='AtE8000all8fixedcandidates atall4layouts/all3K fall below100um coordinateNR onbothmeshes, despite smallstrain/positiveJ/validlinearequilibrium. Requiringlegacyload_eligibleTrue forEVERYfutureNHcatalogentry is therefore a foreseeable excitation-admission problem, not evidence of invalidlinearphysics. ActualNHcases remainunrun. Futureprotocol must separately reportphysicalvalidity andexperimentalexcitation, with thresholds/roles prospectivelyfrozen. Never silentlyrelax100um/.3/5%/20um orchangeforcesconditionedonunknownE.',fraction_limit='Producer savedEuclideanNR andcoordinateNR only, nottotalvolumeRMS ornonrigidfraction/fullu. No legitimate fraction>=.3 check or comparisoncanbe reconstructed fromthese archives; doNOT substitute16markerfit orassumePASS.',scope='Closeddevelopmentdata planningdiagnostic only. No additionalPDE/model/observations/noise/draws, no policy/calibration/NH/bankadmission.')
    path=m.LOCAL/'excitation-design-issue-compact.json';m.write(path,out)
    files=['experiments/diagnose_fixture_gaussian_excitation_v01.py','tests/test_fixture_gaussian_excitation_v01.py',str(path.relative_to(m.ROOT))]
    records=[dict(m.record(m.ROOT/f),bytes=(m.ROOT/f).stat().st_size) for f in files]
    m.write(m.LOCAL/'excitation-design-issue-publication-inventory.json',dict(schema='compact-planning-diagnostic-inventory-v01',files=records,count=len(records),total_bytes=sum(x['bytes'] for x in records),scope='Source/test/derivedcounts+ranges only; no rawresponse/coordinates/fields.'))
    print(json.dumps(dict(report=m.record(path),inventory=m.record(m.LOCAL/'excitation-design-issue-publication-inventory.json'),n6=data['6'])))

if __name__=='__main__':run()
