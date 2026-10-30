"""Independent scope/adoption audit. Never imports the runtime scope guard."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]

def checked_json(root,ref):
    rel=Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Evidence path must remain inside repository')
    path=root/rel
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Evidence hash mismatch: '+str(rel))
    return json.loads(path.read_text())

def validate_scope(p,q,frozen,root=ROOT):
    primary=p['primary_forecast_scenarios'];stress=p['stress_forecast_scenarios'];all_scenes=p['forecast_scenarios']
    if len(set(primary))!=len(primary) or len(set(stress))!=len(stress) or set(primary)&set(stress) or set(primary+stress)!=set(all_scenes):raise ValueError('Invalid fixed scenario partition')
    if set(all_scenes) not in ({'slide','curved','moving_sphere','edge_drop'},{'slide','curved','moving_sphere','rounded_support_drop'}) or len(all_scenes)!=4:raise ValueError('Unrecognized scene matrix')
    mode=p['adopted_scope']
    if mode=='all_forecast_primary':
        if stress or set(primary)!=set(all_scenes) or q['all_gates_pass'] is not True:raise ValueError('Full primary scope lacks full qualification')
    elif mode=='primary_with_stress':
        if set(primary)!={'slide','curved','moving_sphere'} or stress!=['edge_drop'] or q['all_gates_pass'] is not False:raise ValueError('Stress scope must preserve overall failed qualification')
    else:raise ValueError('Scope has not been explicitly adopted')
    if q['adopted_scope']!=mode or q['primary_gates_pass'] is not True:raise ValueError('Primary development qualification missing')
    for key in ('primary_forecast_scenarios','stress_forecast_scenarios'):
        if q[key]!=p[key]:raise ValueError('Qualification scope differs from protocol')
    qualpath=root/'reports/coupled-surface-v13/physics-development-qualification.json'
    if hashlib.sha256(qualpath.read_bytes()).hexdigest()!=frozen['physics_qualification_sha256']:raise ValueError('Frozen qualification hash mismatch')
    if json.loads(qualpath.read_text())!=q:raise ValueError('Qualification argument differs from frozen artifact')
    decision=checked_json(root,q['decision'])
    if decision.get('status')!='adopted':raise ValueError('Decision is not adoption')
    for key in ('adopted_scope','primary_forecast_scenarios','stress_forecast_scenarios'):
        if decision.get(key)!=p[key]:raise ValueError('Adoption decision scope mismatch')
    evidence=q['primary_evidence']
    if set(evidence)!=set(['squeeze']+primary):raise ValueError('Missing or extra primary evidence scenes')
    for scene,refs in evidence.items():
        bound=False
        if not refs:raise ValueError('Empty primary evidence: '+scene)
        for ref in refs:
            value=checked_json(root,ref)
            if ref.get('passed_key')==['scenarios',scene,'qualified']:
                bound=True;entry=value['scenarios'][scene];settings=p['reference_settings_by_scenario'][scene]
                def cells(v):return [v]*3 if isinstance(v,int) else v
                if cells(entry['primary_cells'])!=settings['cells'] or cells(entry['companion_cells'])!=settings['companion_cells'] or entry['dt']!=settings['dt'] or entry['half_dt']!=settings['dt']/2:raise ValueError('Scene evidence discretization mismatch: '+scene)
                if cells(entry['predictor_cells'])!=p['coarse_cells'] or entry['predictor_dt']!=p['dt_coarse'] or entry['surface_spacing_m']!=p['surface_spacing_m'] or entry['friction_level']!=p['friction_quadrature_level']:raise ValueError('Scene evidence predictor/contact mismatch: '+scene)
                if entry.get('source_hashes_verified') is not True or entry.get('pending'):raise ValueError('Scene evidence sources/pending invalid: '+scene)
                for gate in ('spatial','primary_time','companion_time','contact','endpoints'):
                    evidence_gate=entry['gates'][gate]
                    if any(evidence_gate.get(k) is not True for k in ('passed','expected_coverage_pass','source_hashes_verified')):raise ValueError('Scene evidence gate incomplete: '+scene+'/'+gate)
            if not isinstance(ref.get('passed_key'),list) or not ref['passed_key']:raise ValueError('Explicit nonempty pass path required')
            for key in ref['passed_key']:value=value[key]
            if value is not True:raise ValueError('Evidence does not pass: '+scene)
        if not bound:raise ValueError('Missing scene-bound aggregate evidence: '+scene)
    return dict(primary=primary,stress=stress,primary_reserved_expected=8*len(primary),full_reserved_expected=32)


def supplement_cohorts(rows,p):
    """Independent recomputation from fixed membership; failures remain denominators."""
    result={}
    for split in ('development','reserved','all_objects'):
        objects=[o['id'] for o in p['objects'] if split=='all_objects' or o['split']==split]
        result[split]={}
        for cohort,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios']),('full',p['forecast_scenarios'])]:
            expected={(o,s) for o in objects for s in scenes};selected=[r for r in rows if (r['object'],r['scenario']) in expected]
            if len({(r['object'],r['scenario']) for r in selected})!=len(selected):raise ValueError('Duplicate supplement cohort cases')
            group=dict(scenarios=scenes,expected=len(expected),present=len(selected),reference_qualified=sum(bool(r.get('joint_reference_qualified')) for r in selected),methods={})
            for method in ('vertex','nominal'):
                complete=[r for r in selected if r.get('results',{}).get(method,{}).get('complete')]
                gains=[r['surface_fem_gain_mm'][method] for r in selected if method in r.get('surface_fem_gain_mm',{})]
                group['methods'][method]=dict(expected=len(expected),complete=len(complete),failed_or_missing=len(expected)-len(complete),means_conditional_on_success=len(complete)!=len(expected),mean_coordinate_mm=sum(r['results'][method]['coordinate_rmse_m']*1000 for r in complete)/len(complete) if complete else None,paired_cases=len(gains),surface_fem_gain_mm=sum(gains)/len(gains) if gains else None)
            result[split][cohort]=group
    return result

def verify_main_cohorts(a,p,report):
    stored=[]
    for split in ('development','reserved'):
        path=report/(split+'-results.json')
        if path.exists():stored+=json.loads(path.read_text())['cases']
    index={(r['object'],r['scenario']):r for r in stored}
    a.check(len(index)==len(stored),'Unique main cohort records')
    full_summary=json.loads((report/'summary.json').read_text());summary=full_summary['forecast_cohorts']
    if (report/'ablation-results.json').exists():
        supplementary=json.loads((report/'ablation-results.json').read_text())
        a.check(full_summary.get('matched_ablation',{}).get('forecast_cohorts')==supplementary.get('forecast_cohorts'),'Summary supplementary cohorts exactly match independently audited artifact')
    methods=['fem','native','rigid','position_ridge','position_rigid','persistence','oracle']
    for split in ('development','reserved','all_objects'):
        ids=[o['id'] for o in p['objects'] if split=='all_objects' or o['split']==split]
        for name,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios']),('full',p['forecast_scenarios'])]:
            keys=[(o,s) for o in ids for s in scenes];rows=[index.get(k,dict(object=k[0],scenario=k[1],results={})) for k in keys];got=summary[split][name];label=split+'/'+name
            a.check(got['scenarios']==scenes and got['expected']==len(keys) and got['present']==sum(k in index for k in keys),label+' fixed cohort accounting')
            a.check(got['reference_qualified']==sum(bool(r.get('joint_reference_qualified')) for r in rows),label+' qualification does not filter membership')
            for method in methods:
                results=[r['results'][method] for r in rows if r.get('results',{}).get(method,{}).get('complete')];qualified=[r['results'][method] for r in rows if r.get('joint_reference_qualified') and r.get('results',{}).get(method,{}).get('complete')];m=got['methods'][method]
                failed={(r['object'],r['scenario']) for r in rows if not r.get('results',{}).get(method,{}).get('complete')}
                a.check(m['expected']==len(keys) and m['complete']==len(results) and m['mean_conditional_on_complete_cases']==(len(results)!=len(keys)),label+'/'+method+' complete denominator')
                a.check(len(m['failures'])==len(failed) and {(r['object'],r['scenario']) for r in m['failures']}==failed,label+'/'+method+' explicit failures')
                for field,coordinate in [('coordinate_mean_mm','coordinate_rmse_m'),('shape_mean_mm','shape_rmse_m')]:
                    value=sum(r[coordinate] for r in results)*1000/len(results) if results else None
                    if value is None:a.check(m[field] is None,label+'/'+method+' empty mean')
                    else:a.close(m[field],value,label+'/'+method+'/'+field)
                a.check(m['qualified_calibration_and_forecast_count']==len(qualified),label+'/'+method+' qualified subset count')
                value=sum(r['coordinate_rmse_m'] for r in qualified)*1000/len(qualified) if qualified else None
                if value is None:a.check(m['qualified_reference_coordinate_mean_mm'] is None,label+'/'+method+' empty qualified mean')
                else:a.close(m['qualified_reference_coordinate_mean_mm'],value,label+'/'+method+' secondary qualified mean')
                if method=='fem':continue
                gains=[r['results'][method]['coordinate_rmse_m']-r['results']['fem']['coordinate_rmse_m'] for r in rows if r.get('results',{}).get(method,{}).get('complete') and r.get('results',{}).get('fem',{}).get('complete')]
                pair=got['fem_comparisons'].get(method)
                if not gains:a.check(pair is None,label+'/'+method+' no invented pairs');continue
                a.check(pair is not None and pair['pairs']==len(gains),label+'/'+method+' paired coverage')
                if pair:
                    a.close(pair['mean_gain_mm'],sum(gains)*1000/len(gains),label+'/'+method+' paired gain')
                    a.check(pair['wins']==sum(x>1e-12 for x in gains) and pair['losses']==sum(x< -1e-12 for x in gains) and pair['ties']==sum(abs(x)<=1e-12 for x in gains),label+'/'+method+' paired outcomes')
