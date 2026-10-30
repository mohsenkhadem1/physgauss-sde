"""Report the separately frozen surface-contact extension, preserving v13."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/coupled-surface-v13'
BASE=ROOT/'runs/coupled-surface-v13'


def implementation():
    name='scripts.coupled_contact_v13._surface_isolated_report'
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/report.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.REPORT=REPORT
    return module


def supplementary_comparisons():
    path=REPORT/'ablation-results.json'
    if not path.exists():return dict(complete=False,reason='Matched supplements not available yet'),['','MATCHED SUPPLEMENTS: pending']
    data=json.loads(path.read_text());out={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'splits':{}}
    lines=['','MATCHED SCENE/MESH VERTEX-CONTACT FORMULATION AND FIXED-NOMINAL COMPARISONS',
           'All use the new public scene geometry, contact stiffness and reference. Vertex parameters are refit from the same calibration observations. Nominal surface parameters are fixed before forecasts.',
           'The vertex comparator differs in contact sampling and effective damping mass (nodal versus area-distributed mass); gains do not isolate quadrature alone.',
           'Paired gain is comparator error minus fitted surface FEM error; positive favors fitted surface FEM. Conditional means exclude explicitly failed forecasts.']
    for split in ('development','reserved'):
        mainpath=REPORT/f'{split}-results.json'
        main=json.loads(mainpath.read_text()) if mainpath.exists() else {'cases':[]}
        lookup={(r['object'],r['scenario']):r for r in main['cases']}
        rows=[r for r in data['cases'] if r['split']==split];out['splits'][split]={}
        lines+=['',split.upper()+' / FULL COHORT','Method | complete/expected | world mm | aligned shape mm | matched pairs | gain vs fitted surface FEM mm | FEM wins/losses/ties']
        for family in ('vertex','nominal'):
            valid=[r['results'][family] for r in rows if r.get('results',{}).get(family,{}).get('complete')]
            pairs=[]
            for row in rows:
                comparator=row.get('results',{}).get(family,{})
                fitted=lookup.get((row['object'],row['scenario']),{}).get('results',{}).get('fem',{})
                if comparator.get('complete') and fitted.get('complete'):pairs.append([fitted['coordinate_rmse_m'],comparator['coordinate_rmse_m']])
            paired=np.asarray(pairs).reshape(-1,2)*1000;gain=paired[:,1]-paired[:,0]
            record=dict(expected=len(rows),complete=len(valid),mean_coordinate_mm=float(np.mean([r['coordinate_rmse_m'] for r in valid])*1000) if valid else None,
                        mean_shape_mm=float(np.mean([r['shape_rmse_m'] for r in valid])*1000) if valid else None,
                        means_conditional_on_success=len(valid)!=len(rows),paired_cases=len(pairs),
                        paired_fem_mean_mm=float(paired[:,0].mean()) if len(pairs) else None,
                        paired_comparator_mean_mm=float(paired[:,1].mean()) if len(pairs) else None,
                        mean_fem_gain_mm=float(gain.mean()) if len(gain) else None,
                        fem_wins=int((gain>1e-9).sum()),fem_losses=int((gain< -1e-9).sum()),ties=int((abs(gain)<=1e-9).sum()),
                        failed_cases=[dict(object=r['object'],scenario=r['scenario'],reason=r.get('results',{}).get(family,{}).get('failure',r.get('unavailable_reason','unavailable'))) for r in rows if not r.get('results',{}).get(family,{}).get('complete')])
            out['splits'][split][family]=record
            def fmt(value):return 'unavailable' if value is None else f'{value:.4f}'
            lines.append(f"{family} | {len(valid)}/{len(rows)} | {fmt(record['mean_coordinate_mm'])} | {fmt(record['mean_shape_mm'])} | {len(pairs)} | {fmt(record['mean_fem_gain_mm'])} | {record['fem_wins']}/{record['fem_losses']}/{record['ties']}")
    out['forecast_cohorts']=data.get('forecast_cohorts',{})
    lines+=['','FIXED SUPPLEMENT PRIMARY / STRESS / FULL COHORTS','Membership is prospectively fixed, never selected by case quality; stress remains development-unqualified.']
    if not out['forecast_cohorts']:lines.append('Fixed supplement cohort summaries pending')
    for split,cohorts in out['forecast_cohorts'].items():
        for label,record in cohorts.items():
            lines.append(f"{split.upper()} / {label.upper()}: expected {record['expected']}; present {record['present']}; scenarios={record['scenarios']}")
            lines.append('Method | complete/expected | world mm | failed/missing | paired cases | fitted FEM gain mm')
            for method,m in record['methods'].items():
                lines.append(f"{method} | {m['complete']}/{m['expected']} | {m['mean_coordinate_mm']} | {m['failed_or_missing']} | {m['paired_cases']} | {m['surface_fem_gain_mm']}")
    out['accounting_complete']=len(data['cases'])==data['expected_cases']
    return out,lines


def reference_accounting(protocol):
    scenarios=['squeeze']+protocol['forecast_scenarios'];objects=protocol['objects']
    levels={}
    for directory in ('reference','reference-refinement','reference-spatial'):
        rows=[]
        for obj in objects:
            for scene in scenarios:
                path=BASE/directory/obj['id']/scene/'complete.json'
                status=json.loads(path.read_text()) if path.exists() else {}
                rows.append(dict(object=obj['id'],scenario=scene,present=bool(status),success=status.get('success',False),
                                 simulation_settings=status.get('simulation_settings'),error=status.get('error')))
        levels[directory]=dict(expected=len(rows),present=sum(r['present'] for r in rows),complete=sum(r['success'] for r in rows),cases=rows)
    spatial=[]
    for obj in objects:
        for scene in scenarios:
            path=BASE/'reference'/obj['id']/scene/'spatial-quality.json'
            q=json.loads(path.read_text()) if path.exists() else None
            spatial.append(dict(object=obj['id'],scenario=scene,checked=q is not None,qualified=bool(q and q['qualified']),quality=q))
    return dict(expected_total=len(objects)*len(scenarios)*3,levels=levels,
                reference_settings_by_scenario=protocol['reference_settings_by_scenario'],
                spatial=dict(expected=len(spatial),checked=sum(r['checked'] for r in spatial),qualified=sum(r['qualified'] for r in spatial),cases=spatial),
                interpretation='Primary target is fixed prospectively. Half-dt checks temporal consistency; companion mesh at the same base dt checks spatial consistency. Missing or failed checks remain unqualified; no target substitution.')


def forecast_cohorts(protocol,module):
    from scripts.coupled_contact_v13.surface_scope import partition
    primary,stress=partition(protocol);out={};stored=[]
    for split in ('development','reserved'):
        path=REPORT/f'{split}-results.json'
        if path.exists():stored.extend(json.loads(path.read_text())['cases'])
    lookup={(r['object'],r['scenario']):r for r in stored}
    if len(lookup)!=len(stored):raise ValueError('Duplicate forecast records')
    expected={(o['id'],s) for o in protocol['objects'] for s in protocol['forecast_scenarios']}
    if set(lookup)-expected:raise ValueError('Undeclared forecast records')
    for split in ('development','reserved','all_objects'):
        objects=[o for o in protocol['objects'] if split=='all_objects' or o['split']==split];out[split]={}
        for name,scenarios in [('primary',primary),('stress',stress),('full',protocol['forecast_scenarios'])]:
            keys=[(o['id'],s) for o in objects for s in scenarios]
            rows=[lookup.get(k,dict(object=k[0],scenario=k[1],results={},unavailable_reason='Fixed forecast record unavailable')) for k in keys]
            summary=module.summarize(rows)
            summary.update(scenarios=scenarios,expected=len(keys),present=sum(k in lookup for k in keys),reference_qualified=sum(bool(r.get('joint_reference_qualified')) for r in rows),membership='Fixed protocol scenarios and objects; never filtered by per-case reference quality')
            out[split][name]=summary
    return out


def main():
    module=implementation();module.main()
    path=REPORT/'summary.json';result=json.loads(path.read_text())
    protocol=json.loads((REPORT/'protocol.json').read_text())
    result['forecast_cohorts']=forecast_cohorts(protocol,module)
    result['adopted_scope']=protocol.get('adopted_scope','NOT ADOPTED')
    result['stress_scope']='Exploratory stress scenarios remain development-unqualified even when individual fresh references pass per-case checks; no case-level pass promotes them into the fixed primary cohort.'
    result['scope']='Separately frozen material-surface-contact study: two development and eight new reserved objects; controlled synthetic cross-interaction transfer with same-constitutive-family fine/coarse FEM and known scene geometry. Native spring-force core adaptation is not full PhysTwin; empirical baselines lack collision geometry.'
    result['contact_representation']='Exact piecewise-linear normal-force integration over clipped rest-surface triangles for planes and boxes, refined friction quadrature, and conservative sparse material-surface quadrature for spheres; shared configuration between reference and all physical predictors. Exact normal integration does not imply exact friction, continuous collision detection or converged trajectories.'
    result['inertia_representation']='Consistent tetrahedral P1 mass for reference, FEM, native spring adaptation and matched vertex comparator; rigid comparator uses exact uniform-volume inertia. Contact damping law remains formulation-specific.'
    result['reference_accounting']=reference_accounting(json.loads((REPORT/'protocol.json').read_text()))
    result['reference_qualification']='Per-case temporal/body checks plus distinct-mesh spatial companions at 27 exact shared points and 2678 P1-interpolated surface points; world RMS<=1mm, aligned shape RMS<=0.5mm, per-collider center RMS<=1mm. Both spatial resolutions require prior development timestep qualification. These checks are discretization consistency, not measured physical ground truth.'
    result['public_benchmark_scope']='This controlled synthetic extension does not establish improvement on both archived public benchmarking datasets. The separate development-only rigid-pose transfer pilot is not promoted: SoftVT mean gain 0.0214mm with10 wins/10 losses; PhysTwin is unchanged.'
    result['study_relationship']='Post-audit extension; historical vertex-contact experiment is preserved. Matched scene/mesh vertex-contact formulation comparison uses the new public geometry/stiffness and separately refitted calibration. Its contact damping mass differs, so it is not a pure quadrature ablation.'
    supplementary,table=supplementary_comparisons()
    result['matched_ablation']=supplementary
    module.write(path,result)
    textpath=REPORT/'RESULTS.txt';text=textpath.read_text()
    text=text.replace('COUPLED CONTACT v13 RESULTS','COUPLED SURFACE CONTACT v13 RESULTS',1)
    cohort_lines=['FIXED PRIMARY / STRESS / FULL FORECAST COHORTS',result['stress_scope'],'Membership is fixed before testing; all declared members remain, including failed or unqualified cases. Numerical means with failures are explicitly conditional on completion.']
    for split,cohorts in result['forecast_cohorts'].items():
        for label,summary in cohorts.items():
            cohort_lines.append(f"{split}/{label}: expected {summary['expected']}, present {summary['present']}, reference-qualified {summary['reference_qualified']}; scenarios={summary['scenarios']}")
            for method,record in summary['methods'].items():
                cohort_lines.append(f"  {method}: {record['complete']}/{record['expected']} complete; mean mm={record['coordinate_mean_mm']}; failures={len(record['failures'])}")
    text='\n'.join(cohort_lines)+'\n\n'+text
    account=result['reference_accounting']
    reference_lines=['REFERENCE ACCOUNTING',f"Expected {account['expected_total']} simulations: primary, half-dt and spatial companion for every object/scenario."]
    reference_lines += [f"{name}: {record['complete']}/{record['expected']} completed, {record['present']} recorded" for name,record in account['levels'].items()]
    reference_lines += [f"Per-case spatial checks: {account['spatial']['qualified']}/{account['spatial']['expected']} qualified; {account['spatial']['checked']} checked.",json.dumps(account['reference_settings_by_scenario'],sort_keys=True),account['interpretation']]
    text += '\n'+result['inertia_representation']+'\n'+result['reference_qualification']+'\n'+result['public_benchmark_scope']+'\n'+'\n'.join(reference_lines)+'\n'
    textpath.write_text(text+'\n'+result['scope']+'\n'+result['contact_representation']+'\n'+result['study_relationship']+'\n'+'\n'.join(table)+'\n')


if __name__=='__main__':main()
