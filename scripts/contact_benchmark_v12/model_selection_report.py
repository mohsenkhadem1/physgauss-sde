"""Complete-cohort exploratory report; never pools incompatible dataset metrics."""
import collections,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,datetime,timezone
from scripts.contact_benchmark_v12.feasible_experiment import COHORTS,specifications
from scripts.contact_benchmark_v12.report import finite,mean,median,fmt
METRIC={'SoftVTBench':'coordinate_rmse_mm','PhysTwin':'manual_track_mean_mm'}
SELECTED='model_selection_selected'
NATIVE=['adapted_phystwin','adapted_phystwin_scene_state']

def complete(s,metric):
    return finite(s.get(metric)) and s.get('failed_frames',0)==0 and s.get('valid_frames',0)==s.get('total_frames',0) and s.get('total_frames',0)>0

def paired(rows,dataset,method,reference):
    key=METRIC[dataset];pairs=[]
    for row in rows:
        a=row['results'].get(method,{}).get('summary',{});b=row['results'].get(reference,{}).get('summary',{})
        if complete(a,key) and complete(b,key):pairs.append(dict(name=row['name'],cohort=row['cohort'],selected_mm=a[key],reference_mm=b[key],gain_mm=b[key]-a[key]))
    return dict(paired_count=len(pairs),wins=sum(r['gain_mm']>1e-9 for r in pairs),losses=sum(r['gain_mm']< -1e-9 for r in pairs),ties=sum(abs(r['gain_mm'])<=1e-9 for r in pairs),mean_gain_mm=mean([r['gain_mm'] for r in pairs]),pairs=pairs)

def summarize(rows,dataset,expected,include_native=False,native_expected=None):
    metric=METRIC[dataset];prior='feasible_contact_scene_state' if dataset=='SoftVTBench' else 'feasible_object_frame'
    required=['v11_fem',prior,SELECTED,'model_selection_current','model_selection_field_free','model_selection_rigid','prefix_calibrated_rigid']
    required+=['rigid_scene_state'] if dataset=='SoftVTBench' else ['official_phystwin']
    if dataset=='SoftVTBench' and include_native:required+=NATIVE
    methods=sorted(set(required).union(*(r['results'].keys() for r in rows)));table={}
    for method in methods:
        eligible=[r for r in rows if method not in NATIVE or r['cohort']!='development']
        wanted=(native_expected if native_expected is not None else expected) if method in NATIVE else expected
        items=[(r,r['results'][method]['summary']) for r in eligible if method in r['results']]
        values=[s[metric] for _,s in items if finite(s.get(metric))];full=[s[metric] for _,s in items if complete(s,metric)]
        failures=[dict(name=r['name'],cohort=r['cohort'],failed_frames=s.get('failed_frames',0),valid_frames=s.get('valid_frames'),total_frames=s.get('total_frames'),finite_metric=finite(s.get(metric))) for r,s in items if not complete(s,metric)]
        worst=sorted([dict(name=r['name'],cohort=r['cohort'],error_mm=s[metric],failed_frames=s.get('failed_frames',0)) for r,s in items if finite(s.get(metric))],key=lambda r:r['error_mm'],reverse=True)[:5]
        table[method]=dict(present=len(items),expected=wanted,complete=len(full),finite_metric_count=len(values),mean_mm=mean(values),complete_only_mean_mm=mean(full),median_mm=median(values),
            mean_is_conditional_on_valid_frames=any(s.get('failed_frames',0)>0 for _,s in items),shape_mean_mm=mean([s['aligned_shape_rmse_mm'] for _,s in items if finite(s.get('aligned_shape_rmse_mm'))]),
            worst_mm=max(values,default=None),worst_cases=worst,failed_frames=sum(s.get('failed_frames',0) for _,s in items),failed_cases=failures,missing_count=max(0,wanted-len(items)),missing_names=[r['name'] for r in eligible if method not in r['results']],**paired(eligible,dataset,method,prior))
    refs=[prior,'v11_fem','prefix_calibrated_rigid','model_selection_rigid']+(['rigid_scene_state']+(NATIVE if include_native else []) if dataset=='SoftVTBench' else ['official_phystwin'])
    return dict(methods=table,expected=expected,present=len(rows),required_methods=required,selected_complete=table[SELECTED]['complete']==expected,
                all_comparisons_complete=all(table[m]['complete']==table[m]['expected'] for m in required),
                selected_comparisons={ref:paired(rows,dataset,SELECTED,ref) for ref in refs})

def append_table(lines,dataset,s):
    lines += [dataset+' — '+METRIC[dataset], 'method | available/expected | complete | finite metrics | finite mean mm | complete-only mean mm | median mm | shape mm | failed frames | wins/losses/pairs vs previous guarded']
    for method,r in s['methods'].items():
        lines.append(f"{method} | {r['present']}/{r['expected']} | {r['complete']} | {r['finite_metric_count']} | {fmt(r['mean_mm'])} | {fmt(r['complete_only_mean_mm'])} | {fmt(r['median_mm'])} | {fmt(r['shape_mean_mm'])} | {r['failed_frames']} | {r['wins']}/{r['losses']}/{r['paired_count']}")
        if r['mean_is_conditional_on_valid_frames']:lines.append('  CAUTION finite mean includes partial forecasts; valid-frame conditional, not full-coverage performance.')
        if r['missing_count']:lines.append('  Missing count '+str(r['missing_count'])+'; present-case missing names '+str(r['missing_names']))
        if r['failed_cases']:lines.append('  Failed/incomplete cases: '+json.dumps(r['failed_cases']))
    lines += ['Selected against strong baselines: reference | wins/losses/ties | complete pairs | mean gain mm (positive better)']
    for reference,p in s['selected_comparisons'].items():
        lines.append(f"{reference} | {p['wins']}/{p['losses']}/{p['ties']} | {p['paired_count']} | {fmt(p['mean_gain_mm'])}")
        if finite(p['mean_gain_mm']) and p['mean_gain_mm']<0:lines.append('  Selected is WORSE on paired mean than '+reference)
    lines.append('Worst selected finite cases (failed-frame flags retained): '+json.dumps(s['methods'][SELECTED]['worst_cases']))

def build():
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='ALL cohorts post-audit exploratory; no new independent generalization claim',freeze_sha256=sha256(REPORT/'model-selection-freeze.json'),protocol_sha256=sha256(REPORT/'model-selection-protocol.json'),cohorts={},sequences={},metadata={},sources={},overall_datasets={})
    lines=['MODEL-CLASS SELECTION — POST-AUDIT EXPLORATORY',output['created_utc'],output['scope'],'Original frozen studies remain unchanged. Candidates selected only with observed-prefix validation.','Dataset metrics differ and are not pooled. Finite means, complete-only means and all missing/failure counts are separate.','']
    allrows=[];expected_counts=collections.Counter();native_expected=0
    for cohort in COHORTS:
        specs=specifications(cohort);rows=[];metadata={};missing=[]
        for spec in specs:
            name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}";path=BASE/'model-selection-evaluation'/cohort/name/'metrics.json'
            if not path.exists():missing.append(name);continue
            row=json.loads(path.read_text());assert row['freeze_sha256']==output['freeze_sha256'] and row['name']==name and row['cohort']==cohort
            mp=path.with_name('prediction-metadata.json');assert sha256(mp)==row['metadata_sha256'];assert row['protocol_sha256']==output['protocol_sha256']
            rows.append(row);metadata[name]=json.loads(mp.read_text());output['sources'][str(path.relative_to(ROOT))]=sha256(path);output['sources'][str(mp.relative_to(ROOT))]=sha256(mp)
        allrows.extend(rows);choices=collections.Counter(m['selected_configuration']['family'] for m in metadata.values());switches=sum(m['switched_from_previous'] for m in metadata.values());all_failed=sum(not any(r['validation_finite'] for r in m['validation']) for m in metadata.values())
        output['sequences'][cohort]=rows;output['metadata'][cohort]=metadata;output['cohorts'][cohort]=dict(missing=missing,expected=len(specs),present=len(rows),datasets={},dataset_diagnostics={},family_counts=dict(choices),switched=switches,all_validation_failed=all_failed)
        lines+=['COHORT '+cohort+' (exploratory)',f'Available {len(rows)}/{len(specs)}; missing: {missing}',f'Chosen families {dict(choices)}; changed {switches}; all-validation-failed fallback {all_failed}']
        for dataset in METRIC:
            subset=[r for r in rows if r['dataset']==dataset];expected=sum(('case' in s)==(dataset=='PhysTwin') for s in specs);expected_counts[dataset]+=expected
            if dataset=='SoftVTBench' and cohort!='development':native_expected+=expected
            summary=summarize(subset,dataset,expected,include_native=cohort!='development');output['cohorts'][cohort]['datasets'][dataset]=summary['methods'];output['cohorts'][cohort]['dataset_diagnostics'][dataset]={k:v for k,v in summary.items() if k!='methods'};append_table(lines,dataset,summary)
            prior='feasible_contact_scene_state' if dataset=='SoftVTBench' else 'feasible_object_frame';metric=METRIC[dataset]
            lines+=['Per case: name | selected | previous guarded | v11 | field-free | rigid (mm) | selected failed/total frames | fit terminal flags']
            for row in subset:
                keys=[SELECTED,prior,'v11_fem','model_selection_field_free','model_selection_rigid'];s=row['results'].get(SELECTED,{}).get('summary',{});fit=row.get('field_free_fit_diagnostics',{})
                lines.append(row['name']+' | '+' | '.join(fmt(row['results'].get(k,{}).get('summary',{}).get(metric)) for k in keys)+f" | {s.get('failed_frames','NA')}/{s.get('total_frames','NA')} | "+json.dumps({tag:r.get('completion',{}).get('terminal_failure',False) for tag,r in fit.items()}))
        lines.append('')
    jobs=[json.loads(p.read_text()) for p in sorted((BASE/'model-selection-jobs').glob('*.json'))];output['jobs']=dict(states=dict(collections.Counter(j['state'] for j in jobs)),records=jobs)
    output['accounting_complete']=len(allrows)==148 and len(jobs)==296 and all(j['state'] in ['complete','failed'] for j in jobs)
    output['selected_complete']=all(d['selected_complete'] for c in output['cohorts'].values() for d in c['dataset_diagnostics'].values())
    output['all_comparisons_complete']=all(d['all_comparisons_complete'] for c in output['cohorts'].values() for d in c['dataset_diagnostics'].values())
    output['complete']=output['accounting_complete'] and output['selected_complete']
    if output['accounting_complete']:
        lines+=['OVERALL WITHIN EACH DATASET — POST-AUDIT EXPLORATORY, DEVELOPMENT INCLUDED','Equal sequence weights;140SoftVT episodes share assets;8PhysTwin cases. These are descriptive pooled WITHIN-dataset summaries, not independent generalization estimates.']
        for dataset in METRIC:
            summary=summarize([r for r in allrows if r['dataset']==dataset],dataset,expected_counts[dataset],include_native=True,native_expected=native_expected if dataset=='SoftVTBench' else None);output['overall_datasets'][dataset]=summary;append_table(lines,dataset,summary)
    lines+=['Execution '+json.dumps(output['jobs']['states']),f"Accounting complete: {output['accounting_complete']}; selected full forecasts complete: {output['selected_complete']}; all required standalone/baseline comparisons complete: {output['all_comparisons_complete']}",'Terminal failure markers mean accounted for, not a successful fit. Standalone unavailable families remain explicit failed outputs.']
    REPORT.mkdir(exist_ok=True,parents=True)
    with gzip.open(REPORT/'model-selection-results.json.gz','wt') as f:json.dump(output,f,allow_nan=False)
    (REPORT/'MODEL-SELECTION-RESULTS.txt').write_text('\n'.join(lines)+'\n')
    tex=[r'\begin{table}[ht]\centering\small',r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrrrr}',r'\toprule',r'Cohort & Dataset & Complete/$N$ & Failed frames & v11 & Guarded & Selected & Rigid+scene & PhysTwin \\',r'\midrule']
    labels={'development':'Development','regression':'Regression','untouched_confirmation':'Earlier cohort','fresh_confirmation':'Later cohort'}
    for cohort,record in output['cohorts'].items():
        for dataset,table in record['datasets'].items():
            selected=table[SELECTED];prior='feasible_contact_scene_state' if dataset=='SoftVTBench' else 'feasible_object_frame';rigid='rigid_scene_state' if dataset=='SoftVTBench' else 'model_selection_rigid';reference='adapted_phystwin_scene_state' if dataset=='SoftVTBench' else 'official_phystwin'
            values=[fmt(table.get(k,{}).get('mean_mm')) for k in ['v11_fem',prior,SELECTED,rigid,reference]]
            tex.append(labels[cohort]+' & '+dataset+f" & {selected['complete']}/{selected['expected']} & {selected['failed_frames']} & "+' & '.join(values)+r' \\')
    tex += [r'\bottomrule\end{tabular}}',r'\caption{All rows are post-audit exploratory, including earlier confirmation cohorts. Means in millimeters; dataset metrics differ. Finite means with failed frames are conditional, not full-coverage performance. PhysTwin column is the common-scene adapted baseline on SoftVTBench and official reference on PhysTwin. Rigid uses scene contact on SoftVTBench; prefix nodal rigid on PhysTwin. NA denotes unavailable. Full-case/failure and complete-only means accompany the text report.}',r'\end{table}']
    (ROOT/'papers/contact-benchmark-v12/generated-model-selection-results.tex').write_text('\n'.join(tex)+'\n');return output

if __name__=='__main__':
    out=build();print('Model-selection report coverage:',out['accounting_complete'],out['selected_complete'],out['all_comparisons_complete'])
