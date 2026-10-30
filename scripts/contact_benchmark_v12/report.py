"""Frozen-cohort reporting; nondevelopment results require method-freeze.json."""
from pathlib import Path
import argparse, collections, datetime, gzip, json, math, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT, BASE, REPORT, sha256
PRIMARY={'SoftVTBench':'contact_scene_state','PhysTwin':'object_frame'}
METRIC={'SoftVTBench':'coordinate_rmse_mm','PhysTwin':'manual_track_mean_mm'}
COHORTS=['development','regression','untouched_confirmation']
DISPLAY={'v11_fem':'Frozen v11 FEM','contact_scene_state':'Ours + scene contact','object_frame':'Ours object frame','pose_reset':'FEM + pose reset','prefix_calibrated_rigid':'Prefix rigid','rigid_scene_state':'Rigid + scene contact','adapted_phystwin':'Adapted PhysTwin','adapted_phystwin_scene_state':'Adapted PhysTwin + scene','official_phystwin':'Official PhysTwin'}
COMMON=['v11_fem','pose_reset','object_frame','persistence','control_rigid','position_ridge','prefix_calibrated_rigid']

def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def mean(xs):return sum(xs)/len(xs) if xs else None
def median(xs):
    xs=sorted(xs);n=len(xs)
    return (xs[(n-1)//2]+xs[n//2])/2 if n else None
def fmt(x):return f'{x:.3f}' if finite(x) else 'NA'
def names(spec):return [f"{s['subset']}-task{s['task']}-demo{s['episode']}" for s in spec['SoftVTBench']]+spec['PhysTwin']
def group(name):return name.rsplit('-demo',1)[0] if '-demo' in name else name

def summarize(rows,expected,dataset,cohort):
    key=METRIC[dataset]
    required=COMMON+(['contact_floor_state','contact_scene','contact_scene_state','rigid_scene_state'] if dataset=='SoftVTBench' else ['official_phystwin'])
    if dataset=='SoftVTBench' and cohort!='development':
        required+=['adapted_phystwin']
        if (REPORT/'supplemental-baseline-freeze.json').exists():required+=['adapted_phystwin_scene_state']
    methods=sorted(set(required).union(*(r['results'].keys() for r in rows)))
    result={}
    for method in methods:
        found=[(r,r['results'][method]['summary']) for r in rows if method in r['results']]
        values=[s[key] for _,s in found if finite(s.get(key))]
        shape=[s['aligned_shape_rmse_mm'] for _,s in found if finite(s.get('aligned_shape_rmse_mm'))]
        failed=sum(s.get('failed_frames',0) for _,s in found)
        complete=[s[key] for _,s in found if finite(s.get(key)) and s.get('failed_frames',0)==0 and s.get('valid_frames',0)==s.get('total_frames',0)]
        paired=[]
        for r,s in found:
            b=r['results'].get('v11_fem',{}).get('summary',{})
            if finite(s.get(key)) and finite(b.get(key)) and not s.get('failed_frames',0) and not b.get('failed_frames',0):
                paired.append(dict(name=r['name'],baseline_mm=b[key],candidate_mm=s[key],improvement_mm=b[key]-s[key],improvement_percent=100*(b[key]-s[key])/b[key] if b[key]>0 else None))
        result[method]=dict(expected_sequences=len(expected),present_sequences=len(found),finite_metric_sequences=len(values),complete_sequences=len(complete),missing_sequences=[n for n in expected if not any(r['name']==n for r,_ in found)],failed_frames=failed,
            failed_sequences=[r['name'] for r,s in found if s.get('failed_frames',0)>0],mean_mm=mean(values),median_mm=median(values),worst_mm=max(values) if values else None,complete_only_mean_mm=mean(complete),shape_mean_mm=mean(shape),shape_sequences=len(shape),mean_is_conditional_on_valid_frames=failed>0,
            paired_with_v11=paired,paired_count=len(paired),wins=sum(p['improvement_mm']>1e-9 for p in paired),ties=sum(abs(p['improvement_mm'])<=1e-9 for p in paired),losses=sum(p['improvement_mm']< -1e-9 for p in paired),paired_mean_gain_mm=mean([p['improvement_mm'] for p in paired]))
    return dict(primary_method=PRIMARY[dataset],primary_metric=key,expected_sequences=len(expected),present_sequences=len(rows),missing_sequences=[n for n in expected if n not in {r['name'] for r in rows}],required_methods=required,comparison_complete=all(result[m]['complete_sequences']==len(expected) for m in required),methods=result)

def paired_primary(rows,dataset,reference):
    primary=PRIMARY[dataset];key=METRIC[dataset];pairs=[]
    for row in rows:
        a=row['results'].get(primary,{}).get('summary',{});b=row['results'].get(reference,{}).get('summary',{})
        if finite(a.get(key)) and finite(b.get(key)) and not a.get('failed_frames',0) and not b.get('failed_frames',0):
            pairs.append(dict(name=row['name'],primary_mm=a[key],reference_mm=b[key],gain_mm=b[key]-a[key]))
    return dict(reference=reference,paired_count=len(pairs),wins=sum(p['gain_mm']>1e-9 for p in pairs),losses=sum(p['gain_mm']< -1e-9 for p in pairs),ties=sum(abs(p['gain_mm'])<=1e-9 for p in pairs),mean_gain_mm=mean([p['gain_mm'] for p in pairs]),primary_mean_mm=mean([p['primary_mm'] for p in pairs]),reference_mean_mm=mean([p['reference_mm'] for p in pairs]),pairs=pairs)

def contact_diagnostics(metadata):
    methods={}
    for method in ['contact_floor_state','contact_scene','contact_scene_state','rigid_scene_state','adapted_phystwin_scene_state']:
        per=[]
        for name,meta in metadata.items():
            if method not in meta:continue
            m=meta[method];diag=m.get('diagnostics',[])
            values=[d['max_preprojection_penetration_m']*1000 for d in diag if finite(d.get('max_preprojection_penetration_m'))]
            per.append(dict(name=name,attachment_state=m.get('attachment_state',{}).get('state','unavailable'),forecast_free_frames=len(diag),contact_frames=sum(d.get('contact_substeps',0)>0 for d in diag),contact_substeps=sum(d.get('contact_substeps',0) for d in diag),max_preprojection_penetration_mm=max(values) if values else None,median_frame_max_preprojection_penetration_mm=median(values)))
        methods[method]=dict(sequences=len(per),attachment_state_counts=dict(collections.Counter(p['attachment_state'] for p in per)),sequences_with_contact=sum(p['contact_substeps']>0 for p in per),sequence_max_penetration_median_mm=median([p['max_preprojection_penetration_mm'] for p in per if finite(p['max_preprojection_penetration_mm'])]),sequence_max_penetration_worst_mm=max([p['max_preprojection_penetration_mm'] for p in per if finite(p['max_preprojection_penetration_mm'])],default=None),per_sequence=per)
    return methods

def collect(cohorts):
    freeze=REPORT/'method-freeze.json'
    if any(c!='development' for c in cohorts) and not freeze.exists():raise RuntimeError('Nondevelopment metric access requires method-freeze.json; no reserved result files read')
    protocol_path=REPORT/'benchmark-protocol.json';protocol=json.loads(protocol_path.read_text())
    output=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohorts=cohorts,protocol_sha256=sha256(protocol_path),method_freeze_sha256=sha256(freeze) if freeze.exists() else None,primary_methods=PRIMARY,metrics=METRIC,summary={},groups={},sequences={},jobs={},contact_diagnostics={},source_files={})
    for cohort in cohorts:
        expected=names(protocol[cohort]);rows=[];metadata={}
        for name in expected:
            p=BASE/'evaluation'/cohort/name/'metrics.json'
            if p.exists():
                row=json.loads(p.read_text());assert row['name']==name and row['cohort']==cohort
                rows.append(row);output['source_files'][str(p.relative_to(ROOT))]=sha256(p)
                mp=p.with_name('prediction-metadata.json')
                if mp.exists():
                    metadata[name]=json.loads(mp.read_text());output['source_files'][str(mp.relative_to(ROOT))]=sha256(mp)
                sp=p.with_name('native-scene-supplement.json')
                if sp.exists():
                    extra=json.loads(sp.read_text());assert extra['name']==name and extra['cohort']==cohort
                    assert extra['decision_sha256']==sha256(REPORT/'supplemental-baseline-freeze.json')
                    row['results'][extra['method']]=extra['result']
                    for phase,value in extra['segments'].items():row.setdefault('segments',{}).setdefault(phase,{})[extra['method']]=value
                    metadata.setdefault(name,{})[extra['method']]=extra['metadata']
                    row['supplemental_provenance']={k:v for k,v in extra.items() if k not in ['result','segments','metadata']}
                    output['source_files'][str(sp.relative_to(ROOT))]=sha256(sp)

        output['contact_diagnostics'][cohort]=contact_diagnostics(metadata)
        output['sequences'][cohort]=rows;output['summary'][cohort]={};output['groups'][cohort]={}
        for dataset in PRIMARY:
            wanted=names({dataset:protocol[cohort][dataset],('PhysTwin' if dataset=='SoftVTBench' else 'SoftVTBench'):[]})
            subset=[r for r in rows if r['dataset']==dataset]
            output['summary'][cohort][dataset]=summarize(subset,wanted,dataset,cohort)
            output['summary'][cohort][dataset]['primary_comparisons']={ref:paired_primary(subset,dataset,ref) for ref in ['v11_fem','prefix_calibrated_rigid']+(['rigid_scene_state','adapted_phystwin','adapted_phystwin_scene_state'] if dataset=='SoftVTBench' else ['official_phystwin'])}
            output['groups'][cohort][dataset]={g:summarize([r for r in subset if group(r['name'])==g],[n for n in wanted if group(n)==g],dataset,cohort) for g in sorted({group(n) for n in wanted})}
        jobs=[json.loads(p.read_text()) for p in sorted((BASE/'jobs').glob('*.json'))]
        jobs=[j for j in jobs if j.get('cohort')==cohort]
        output['jobs'][cohort]=dict(counts=dict(collections.Counter(j['state'] for j in jobs)),jobs=jobs)
    output['comparison_complete']=all(d['comparison_complete'] for c in output['summary'].values() for d in c.values())
    return output

def render_text(result):
    lines=['CONTACT BENCHMARK V12 — '+('COMPLETE COMPARISONS' if result['comparison_complete'] else 'PARTIAL / INCOMPLETE COMPARISONS'),f"Generated {result['created_utc']}",'',
    'Primary: SoftVTBench contact_scene_state; PhysTwin object_frame. Fixed before reserved evaluation.',
    'Full forecast suffix is primary; equal sequence weighting. Different dataset metrics are not pooled.',
    'Means with failed frames are conditional and do not establish full-sequence accuracy.',
    'Paired wins/gains use only pairs without failed frames; coverage is explicit. No frame-level confidence intervals.','']
    for cohort,ds in result['summary'].items():
        lines+=['COHORT '+cohort.upper(), 'Job states: '+json.dumps(result['jobs'][cohort]['counts'])]
        for dataset,summary in ds.items():
            lines += ['',f"{dataset}: {summary['present_sequences']}/{summary['expected_sequences']} sequences; metric {summary['primary_metric']}", 'method | n/expected | complete | mean mm | shape mm | failed frames | paired wins/count | paired gain mm']
            for name,s in summary['methods'].items():
                lines.append(f"{name} | {s['present_sequences']}/{s['expected_sequences']} | {s['complete_sequences']} | {fmt(s['mean_mm'])} | {fmt(s['shape_mean_mm'])} | {s['failed_frames']} | {s['wins']}/{s['paired_count']} | {fmt(s['paired_mean_gain_mm'])}")
                if s['missing_sequences']:lines.append('  Missing: '+', '.join(s['missing_sequences']))
                if s['failed_sequences']:lines.append('  Failed-frame sequences: '+', '.join(s['failed_sequences']))
            lines += ['','PRIMARY against strong baselines (complete pairs only; positive gain is improvement):','reference | pairs | primary mm | reference mm | gain mm | wins/losses/ties']
            for ref,c in summary['primary_comparisons'].items():
                lines.append(f"{ref} | {c['paired_count']} | {fmt(c['primary_mean_mm'])} | {fmt(c['reference_mean_mm'])} | {fmt(c['mean_gain_mm'])} | {c['wins']}/{c['losses']}/{c['ties']}")
                if finite(c['mean_gain_mm']) and c['mean_gain_mm']<0:lines.append('  Primary is WORSE on the paired mean than '+ref+'.')
            lines+=['','Per-group PRIMARY versus frozen v11 (positive gain is improvement):','group | n/expected | primary mm | v11 mm | paired gain mm | wins/count']
            for g,gs in result['groups'][cohort][dataset].items():
                p=gs['methods'][PRIMARY[dataset]];b=gs['methods']['v11_fem']
                lines.append(f"{g} | {p['present_sequences']}/{p['expected_sequences']} | {fmt(p['mean_mm'])} | {fmt(b['mean_mm'])} | {fmt(p['paired_mean_gain_mm'])} | {p['wins']}/{p['paired_count']}")
            lines+=['','Per-sequence PRIMARY versus frozen v11:','sequence | primary mm | v11 mm | gain mm | primary failed frames']
            for row in result['sequences'][cohort]:
                if row['dataset']!=dataset:continue
                p=row['results'].get(PRIMARY[dataset],{}).get('summary',{});b=row['results'].get('v11_fem',{}).get('summary',{})
                a=p.get(METRIC[dataset]);v=b.get(METRIC[dataset]);gain=v-a if finite(a) and finite(v) else None
                lines.append(f"{row['name']} | {fmt(a)} | {fmt(v)} | {fmt(gain)} | {p.get('failed_frames','NA')}")
    lines += ['','CONTACT MECHANISM DIAGNOSTICS','Penetration is measured before projection, not a final nonpenetration or CCD certificate.','Median per sequence means median of framewise maximum preprojection gaps, including zero-contact free frames.','A sequence with no free-motion diagnostics has NA penetration, not a measured zero.']
    for cohort,methods in result['contact_diagnostics'].items():
        lines.append(cohort.upper())
        for method,d in methods.items():
            lines.append(f"{method}: states={d['attachment_state_counts']}; contact sequences={d['sequences_with_contact']}/{d['sequences']}; median sequence-max penetration={fmt(d['sequence_max_penetration_median_mm'])} mm; worst={fmt(d['sequence_max_penetration_worst_mm'])} mm")
        lines += ['Primary per sequence: name | state | contact frames/free frames | contact substeps | max penetration mm | median frame-max penetration mm']
        for d in methods['contact_scene_state']['per_sequence']:
            lines.append(f"{d['name']} | {d['attachment_state']} | {d['contact_frames']}/{d['forecast_free_frames']} | {d['contact_substeps']} | {fmt(d['max_preprojection_penetration_mm'])} | {fmt(d['median_frame_max_preprojection_penetration_mm'])}")
    lines += ['','INTERPRETATION AND LIMITS',
    'V11 code/results are preserved; regression cases were already examined. Untouched confirmation uses predeclared new sequences.',
    'SoftVTBench has two primary pastry assets across four task/suite groups, not 40 independent object categories.',
    'PhysTwin confirmation adds trajectories of known toy categories. Tiny case counts do not establish broad category generalization.',
    'Inputs include privileged initial geometry, sparse observed-prefix positions and prescribed future tool positions, not true forces/materials.',
    'The object-frame adapter is prefix-calibrated rigid correction plus optional bounded empirical deformation residual; it is not new material identification.',
    'A post-freeze topology audit found inverted FEM tetrahedra in residual-corrected outputs (see shape-jacobian-audit.json). These forecasts are point-geometry predictions, not physically admissible FEM states. A separately declared feasibility extension preserves this frozen comparison.',
    'Contact primary splits FEM shape and rigid contact pose, with shape frozen at separation. Static mesh heightfields do not resolve vertical walls, undercuts, deformable contact forces or two-way moving-object coupling.',
    'Vertical projection removes residual represented-support penetration but is not continuous collision detection or a general nonpenetration guarantee.',
    'Public scene meshes are extra input versus v11; rigid_scene_state supplies a geometry-matched baseline, and contact_floor_state is the same-input geometry ablation.',
    'Independent shorter-prefix fits select the adapter; no future object truth selects candidates. Contact policy is fixed on development.',
    'All original per-frame metrics, phase diagnostics, selection metadata and source hashes are retained in results.json.gz.',
    'Supplement adapted_phystwin_scene_state was declared after primary freeze and partial scores: identical frozen scene/state pose postprocessing, no native refit or future-score selection, all80SoftVT episodes. It is not unchanged author PhysTwin.',
    'No claim of calibrated physical uncertainty, active probing superiority, real-time speed or release convergence follows from these results.','']
    return '\n'.join(lines)

def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[(c,d,s) for c,ds in result['summary'].items() for d,s in ds.items()]
    fig,axes=plt.subplots(len(result['cohorts']),2,figsize=(13,3.7*len(result['cohorts'])),squeeze=False)
    for ax,(cohort,dataset,summary) in zip(axes.flat,panels):
        choices=['v11_fem',PRIMARY[dataset],'prefix_calibrated_rigid','rigid_scene_state' if dataset=='SoftVTBench' else 'official_phystwin']
        if dataset=='SoftVTBench' and cohort!='development':choices+=['adapted_phystwin','adapted_phystwin_scene_state']
        present=[(m,summary['methods'][m]) for m in choices if m in summary['methods'] and finite(summary['methods'][m]['mean_mm'])]
        if present:
            ax.barh(range(len(present)),[s['mean_mm'] for _,s in present],color=['#25796d' if m==PRIMARY[dataset] else '#738397' for m,_ in present]);ax.set_yticks(range(len(present)),[DISPLAY.get(m,m) for m,_ in present]);ax.invert_yaxis()
            for i,(m,s) in enumerate(present):ax.text(s['mean_mm'],i,f" {s['mean_mm']:.1f} ({s['present_sequences']}/{s['expected_sequences']})",va='center',fontsize=8)
            ax.set_xlim(0,max(s['mean_mm'] for _,s in present)*1.35)
        else:ax.text(.5,.5,'No results yet',ha='center',transform=ax.transAxes)
        ax.set_title(f'{cohort} / {dataset}',fontsize=10);ax.set_xlabel(summary['primary_metric']+' (lower better)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('V12 fixed primary comparison — counts shown; incomplete methods are not full benchmarks',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(path,dpi=180);plt.close(fig)

def paper_tables(result,path):
    """Produce numbers only from this collected snapshot, retaining coverage."""
    if not all(c in result['cohorts'] for c in ['regression','untouched_confirmation']):
        raise ValueError('Paper tables require both regression and confirmation cohorts')
    def escape(text):return str(text).replace('_',r'\_').replace('%',r'\%')
    lines=['% Generated by scripts/contact_benchmark_v12/report.py; do not edit numbers.',
           r'\noindent The additional adapted\_phystwin\_scene\_state comparison applies the same frozen contact postprocessor to native outputs. It was declared after partial primary scores, without refitting or future-score selection; the primary method is unchanged. ',
           r'\noindent Snapshot: '+escape(result['created_utc'])+'. '+
           ('All required method/sequence comparisons are complete.' if result['comparison_complete'] else r'\textbf{Incomplete comparisons:} missing predictions or failed frames remain; see coverage below.')]
    for cohort in ['regression','untouched_confirmation']:
        lines += [r'\begin{table}[ht]',r'\centering\small',r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrr}',r'\toprule',
                  r'Dataset & Method & $n/N$ & Error & Shape & Failures \\',r'\midrule']
        for dataset,s in result['summary'][cohort].items():
            methods=['v11_fem',PRIMARY[dataset],'pose_reset','prefix_calibrated_rigid']
            methods += ['rigid_scene_state','adapted_phystwin'] if dataset=='SoftVTBench' else ['official_phystwin']
            if 'adapted_phystwin_scene_state' in s['methods']:methods+=['adapted_phystwin_scene_state']
            for method in methods:
                m=s['methods'][method]
                lines.append(f"{escape(dataset)} & {escape(DISPLAY.get(method,method))} & {m['present_sequences']}/{m['expected_sequences']} & {fmt(m['mean_mm'])} & {fmt(m['shape_mean_mm'])} & {m['failed_frames']} "+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}}',r'\caption{'+escape(cohort)+r'. Errors in millimeters: SoftVTBench coordinate RMSE, PhysTwin manual-marker mean; shape is rigid-aligned RMSE. $n/N$ is available versus planned sequences; failures count invalid frames. Means with failures are conditional on valid frames. Different dataset metrics are not pooled.}',r'\end{table}']
        for dataset,s in result['summary'][cohort].items():
            m=s['methods'][PRIMARY[dataset]]
            lines.append(escape(dataset)+': the fixed primary improves over v11 on '+f"{m['wins']}/{m['paired_count']}"+' complete paired sequences; mean paired improvement '+fmt(m['paired_mean_gain_mm'])+r'\,mm (negative means regression). '+f"{m['complete_sequences']}/{m['expected_sequences']}"+' primary forecasts are complete. '+r'\par')
            for ref,c in s.get('primary_comparisons',{}).items():
                if ref=='v11_fem' or not c['paired_count']:continue
                relation='better' if c['mean_gain_mm']>0 else 'worse' if c['mean_gain_mm']<0 else 'equal'
                lines.append('Against '+escape(DISPLAY.get(ref,ref))+', primary paired mean is '+relation+' by '+fmt(abs(c['mean_gain_mm']))+r'\,mm across '+str(c['paired_count'])+' complete pairs. '+r'\par')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohorts',nargs='+',choices=COHORTS,default=['development']);parser.add_argument('--paper',action='store_true');a=parser.parse_args()
    result=collect(a.cohorts)
    # Development-only progress never overwrites the final campaign artifact.
    prefix='development-' if a.cohorts==['development'] else ''
    raw=(json.dumps(result,indent=2,allow_nan=True)+'\n').encode()
    (REPORT/(prefix+'results.json')).write_bytes(raw)
    (REPORT/(prefix+'results.json.gz')).write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    (REPORT/(prefix+'RESULTS.txt')).write_text(render_text(result))
    plot(result,REPORT/(prefix+'comparison.png'))
    if a.paper:paper_tables(result,ROOT/'papers/contact-benchmark-v12/generated-results.tex')
    print(json.dumps({'cohorts':a.cohorts,'complete':result['comparison_complete'],'report':str(REPORT/(prefix+'RESULTS.txt'))}))
if __name__=='__main__':main()
