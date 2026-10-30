"""Report the separately frozen surface-contact extension, preserving v13."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/coupled-surface-v13'


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
        lines+=['',split.upper(),'Method | complete/expected | world mm | aligned shape mm | matched pairs | gain vs fitted surface FEM mm | FEM wins/losses/ties']
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
    out['accounting_complete']=len(data['cases'])==data['expected_cases']
    return out,lines


def main():
    module=implementation();module.main()
    path=REPORT/'summary.json';result=json.loads(path.read_text())
    result['scope']='Separately frozen material-surface-contact study: two development and eight new reserved objects; controlled synthetic cross-interaction transfer with same-constitutive-family fine/coarse FEM and known scene geometry. Native spring-force core adaptation is not full PhysTwin; empirical baselines lack collision geometry.'
    result['contact_representation']='Exact piecewise-linear normal-force integration over clipped rest-surface triangles for planes and boxes, refined friction quadrature, and conservative sparse material-surface quadrature for spheres; shared configuration between reference and all physical predictors. Exact normal integration does not imply exact friction, continuous collision detection or converged trajectories.'
    result['study_relationship']='Post-audit extension; historical vertex-contact experiment is preserved. Matched scene/mesh vertex-contact formulation comparison uses the new public geometry/stiffness and separately refitted calibration. Its contact damping mass differs, so it is not a pure quadrature ablation.'
    supplementary,table=supplementary_comparisons()
    result['matched_ablation']=supplementary
    module.write(path,result)
    textpath=REPORT/'RESULTS.txt';text=textpath.read_text()
    text=text.replace('COUPLED CONTACT v13 RESULTS','COUPLED SURFACE CONTACT v13 RESULTS',1)
    textpath.write_text(text+'\n'+result['scope']+'\n'+result['contact_representation']+'\n'+result['study_relationship']+'\n'+'\n'.join(table)+'\n')


if __name__=='__main__':main()
