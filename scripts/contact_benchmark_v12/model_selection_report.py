"""Complete-cohort exploratory report; never pools incompatible dataset metrics."""
import collections,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,write_json,sha256,datetime,timezone
from scripts.contact_benchmark_v12.feasible_experiment import COHORTS,specifications
from scripts.contact_benchmark_v12.report import finite,mean,median,fmt

METRIC={'SoftVTBench':'coordinate_rmse_mm','PhysTwin':'manual_track_mean_mm'}
def build():
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='ALL cohorts post-audit exploratory; no new independent generalization claim',
        freeze_sha256=sha256(REPORT/'model-selection-freeze.json'),cohorts={},sequences={},metadata={},complete=True)
    lines=['MODEL-CLASS SELECTION — POST-AUDIT EXPLORATORY',output['created_utc'],output['scope'],
        'Original frozen studies remain unchanged. Candidates selected only with observed-prefix validation.',
        'Dataset metrics have different definitions and are not pooled. Finite means are accompanied by explicit failure counts.','']
    for cohort in COHORTS:
        specs=specifications(cohort);rows=[];metadata={};missing=[]
        for spec in specs:
            name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
            path=BASE/'model-selection-evaluation'/cohort/name/'metrics.json'
            if not path.exists():missing.append(name);continue
            row=json.loads(path.read_text());assert row['freeze_sha256']==output['freeze_sha256']
            assert sha256(path.with_name('prediction-metadata.json'))==row['metadata_sha256']
            rows.append(row);metadata[name]=json.loads(path.with_name('prediction-metadata.json').read_text())
        choices=collections.Counter(m['selected_configuration']['family'] for m in metadata.values())
        switches=sum(m['switched_from_previous'] for m in metadata.values())
        all_failed=sum(not any(r['validation_finite'] for r in m['validation']) for m in metadata.values())
        output['sequences'][cohort]=rows;output['metadata'][cohort]=metadata;output['cohorts'][cohort]={'missing':missing,'datasets':{},'family_counts':dict(choices),'switched':switches,'all_validation_failed':all_failed}
        if missing:output['complete']=False
        lines+=['COHORT '+cohort+' (exploratory)',f'Available {len(rows)}/{len(specs)}; missing: {missing}',f'Chosen families {dict(choices)}; changed {switches}; all-validation-failed fallback {all_failed}']
        for dataset,metric in METRIC.items():
            selected=[r for r in rows if r['dataset']==dataset];table={}
            expected=sum(('case' in s)==(dataset=='PhysTwin') for s in specs)
            methods=sorted(set().union(*(r['results'].keys() for r in selected)))
            lines += [dataset+' — '+metric,'method | available/expected | complete | mean mm | median mm | shape mm | failed frames | wins/losses vs previous guarded']
            prior='feasible_contact_scene_state' if dataset=='SoftVTBench' else 'feasible_object_frame'
            for method in methods:
                summaries=[r['results'][method]['summary'] for r in selected if method in r['results']]
                complete=[s for s in summaries if s.get('failed_frames',0)==0 and s.get('valid_frames')==s.get('total_frames') and finite(s.get(metric))]
                pairs=[]
                for r in selected:
                    a=r['results'].get(method,{}).get('summary',{});b=r['results'][prior]['summary']
                    if finite(a.get(metric)) and finite(b.get(metric)) and not a.get('failed_frames',0) and not b.get('failed_frames',0):pairs.append(b[metric]-a[metric])
                record=dict(present=len(summaries),expected=expected,complete=len(complete),mean_mm=mean([s[metric] for s in summaries if finite(s.get(metric))]),
                    median_mm=median([s[metric] for s in summaries if finite(s.get(metric))]),shape_mean_mm=mean([s['aligned_shape_rmse_mm'] for s in summaries if finite(s.get('aligned_shape_rmse_mm'))]),
                    worst_mm=max([s[metric] for s in summaries if finite(s.get(metric))],default=None),
                    failed_frames=sum(s.get('failed_frames',0) for s in summaries),paired_count=len(pairs),wins=sum(v>1e-9 for v in pairs),losses=sum(v< -1e-9 for v in pairs),mean_gain_mm=mean(pairs))
                table[method]=record
                lines.append(f"{method} | {record['present']}/{expected} | {record['complete']} | {fmt(record['mean_mm'])} | {fmt(record['median_mm'])} | {fmt(record['shape_mean_mm'])} | {record['failed_frames']} | {record['wins']}/{record['losses']}")
            output['cohorts'][cohort]['datasets'][dataset]=table
            lines+=['Per case: name | selected | previous guarded | v11 | field-free | rigid (mm)']
            for row in selected:
                keys=['model_selection_selected',prior,'v11_fem','model_selection_field_free','model_selection_rigid']
                lines.append(row['name']+' | '+' | '.join(fmt(row['results'].get(k,{}).get('summary',{}).get(metric)) for k in keys))
        lines.append('')
    jobs=[json.loads(p.read_text()) for p in sorted((BASE/'model-selection-jobs').glob('*.json'))]
    output['jobs']=dict(states=dict(collections.Counter(j['state'] for j in jobs)),records=jobs)
    output['complete'] &= len(jobs)==296 and all(j['state']=='complete' for j in jobs)
    lines += ['Execution '+json.dumps(output['jobs']['states']),'Complete declared coverage: '+str(output['complete'])]
    REPORT.mkdir(exist_ok=True,parents=True)
    with gzip.open(REPORT/'model-selection-results.json.gz','wt') as f:json.dump(output,f,allow_nan=False)
    (REPORT/'MODEL-SELECTION-RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return output

if __name__=='__main__':
    out=build();print('Model-selection report complete coverage:',out['complete'])
