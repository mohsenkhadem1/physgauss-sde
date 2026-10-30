"""Separate post-audit feasibility evidence; never overwrites frozen v12 reports."""
import argparse,collections,datetime,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256
from scripts.contact_benchmark_v12.report import finite,mean,median,fmt,group,DISPLAY
COHORTS=['development','regression','untouched_confirmation','fresh_confirmation']
PRIMARY={'SoftVTBench':'feasible_contact_scene_state','PhysTwin':'feasible_object_frame'}
UNCORRECTED={'SoftVTBench':'contact_scene_state','PhysTwin':'object_frame'}
METRIC={'SoftVTBench':'coordinate_rmse_mm','PhysTwin':'manual_track_mean_mm'}
LABELS={**DISPLAY,'feasible_contact_scene_state':'Feasible + scene contact','feasible_object_frame':'Feasible object frame'}

def compare(rows,dataset,reference):
    key=METRIC[dataset];pairs=[]
    for r in rows:
        a=r['results'].get(PRIMARY[dataset],{}).get('summary',{});b=r['results'].get(reference,{}).get('summary',{})
        if finite(a.get(key)) and finite(b.get(key)) and not a.get('failed_frames',0) and not b.get('failed_frames',0):
            pairs.append(dict(name=r['name'],feasible_mm=a[key],reference_mm=b[key],gain_mm=b[key]-a[key]))
    return dict(pairs=pairs,count=len(pairs),wins=sum(p['gain_mm']>1e-9 for p in pairs),losses=sum(p['gain_mm']< -1e-9 for p in pairs),mean_gain_mm=mean([p['gain_mm'] for p in pairs]))

def summarize(rows,expected,dataset,cohort):
    methods=['v11_fem',UNCORRECTED[dataset],PRIMARY[dataset],'pose_reset','prefix_calibrated_rigid']
    methods+=['rigid_scene_state'] if dataset=='SoftVTBench' else ['official_phystwin']
    if dataset=='SoftVTBench' and cohort!='development':methods+=['adapted_phystwin','adapted_phystwin_scene_state']
    required=methods.copy();methods=sorted(set(methods).union(*(r['results'].keys() for r in rows)))
    tables={}
    for m in methods:
        found=[(r,r['results'][m]['summary']) for r in rows if m in r['results']]
        values=[s[METRIC[dataset]] for _,s in found if finite(s.get(METRIC[dataset]))]
        shapes=[s['aligned_shape_rmse_mm'] for _,s in found if finite(s.get('aligned_shape_rmse_mm'))]
        complete=[r['name'] for r,s in found if finite(s.get(METRIC[dataset])) and s.get('failed_frames',0)==0 and s.get('valid_frames')==s.get('total_frames')]
        tables[m]=dict(present=len(found),expected=len(expected),complete=len(complete),mean_mm=mean(values),median_mm=median(values),worst_mm=max(values,default=None),shape_mean_mm=mean(shapes),failed_frames=sum(s.get('failed_frames',0) for _,s in found),missing=[n for n in expected if n not in {r['name'] for r,_ in found}],paired=compare(rows,dataset,m))
    return dict(expected=len(expected),present=len(rows),missing=[n for n in expected if n not in {r['name'] for r in rows}],primary=PRIMARY[dataset],metric=METRIC[dataset],required=required,methods=tables,complete=all(tables[m]['complete']==len(expected) for m in required))

def guard_summary(records):
    rows=[]
    for name,meta in records.items():
        guard=meta['guard'];gframes=guard['frames'];primary=next((k for k in meta['topology'] if k=='feasible_contact_scene_state'),'feasible_object_frame');top=meta['topology'][primary]
        vals=lambda key:[r[key] for r in gframes if finite(r.get(key))]
        alpha=vals('alpha');d=vals('guard_coordinate_rms_m')
        rows.append(dict(name=name,frames=len(gframes),guarded_frames=len(guard['modified_frames']),guarded_fraction=guard['modified_fraction'],failed_guard_frames=len(guard['failed_frames']),cutoff_J=guard['cutoff_J'],numerical_margin=guard['numerical_margin'],min_anchor_J=min(vals('min_base_J'),default=None),min_proposal_J=min(vals('min_proposal_J'),default=None),min_final_J=top['min_J'],proposal_frames_with_inversions=sum(r.get('min_proposal_J',1)<=0 for r in gframes),proposal_frames_below_margin=sum(r.get('min_proposal_J',1)<=.03 for r in gframes),final_frames_with_inversions=top['frames_with_inversions'],final_frames_below_margin=top['frames_below_solver_margin'],final_failed_frames=top['failed_frames'],alpha_counts=dict(collections.Counter(str(a) for a in alpha)),max_guard_coordinate_rms_mm=max(d,default=0)*1000,median_guard_coordinate_rms_mm=median(d)*1000 if d else None,guard_temporal_change=guard.get('temporal_change'),final_temporal_change=meta.get('final_temporal_change_vs_uncorrected')))
    total=sum(r['frames'] for r in rows);changed=sum(r['guarded_frames'] for r in rows)
    return dict(sequences=len(rows),sequences_guarded=sum(r['guarded_frames']>0 for r in rows),frames=total,guarded_frames=changed,guarded_frame_fraction=changed/total if total else None,final_inverted_frames=sum(r['final_frames_with_inversions'] for r in rows),final_below_margin_frames=sum(r['final_frames_below_margin'] for r in rows),failed_frames=sum(r['final_failed_frames'] for r in rows),minimum_final_J=min([r['min_final_J'] for r in rows if finite(r['min_final_J'])],default=None),per_sequence=rows)

def collect(cohorts):
    freeze=REPORT/'feasibility-freeze.json'
    if not freeze.exists():raise RuntimeError('Feasibility freeze required before report access')
    protocols={c:json.loads((REPORT/('feasibility-protocol.json' if c=='fresh_confirmation' else 'benchmark-protocol.json')).read_text())[c] for c in cohorts}
    out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohorts=cohorts,scope='106 old cases post-audit secondary;42 fresh predeclared confirmation, known categories',freeze_sha256=sha256(freeze),protocol_sha256=sha256(REPORT/'feasibility-protocol.json'),summaries={},groups={},sequences={},geometry={},guard_records={},sources={},jobs={})
    for c,p in protocols.items():
        specs=p['SoftVTBench']+[dict(case=x) for x in p['PhysTwin']];rows=[];metadata={}
        for spec in specs:
            name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
            path=BASE/'feasibility-evaluation'/c/name/'metrics.json'
            if not path.exists():continue
            row=json.loads(path.read_text());assert row['name']==name and row['cohort']==c
            assert row['freeze_sha256']==sha256(freeze)
            mp=path.with_name('prediction-metadata.json');assert sha256(mp)==row['guard_metadata_sha256']
            rows.append(row);metadata[name]=json.loads(mp.read_text());out['sources'][str(path.relative_to(ROOT))]=sha256(path);out['sources'][str(mp.relative_to(ROOT))]=sha256(mp)
        out['sequences'][c]=rows;out['guard_records'][c]=metadata;out['geometry'][c]=guard_summary(metadata);out['summaries'][c]={};out['groups'][c]={}
        for dataset in PRIMARY:
            expected=[f"{s['subset']}-task{s['task']}-demo{s['episode']}" for s in p[dataset]] if dataset=='SoftVTBench' else p[dataset]
            subset=[r for r in rows if r['dataset']==dataset]
            out['summaries'][c][dataset]=summarize(subset,expected,dataset,c)
            out['groups'][c][dataset]={g:summarize([r for r in subset if group(r['name'])==g],[n for n in expected if group(n)==g],dataset,c) for g in sorted({group(n) for n in expected})}
    jobs=[json.loads(p.read_text()) for p in sorted((BASE/'feasibility-jobs').glob('*.json'))]
    out['jobs']=dict(counts=dict(collections.Counter(j['state'] for j in jobs)),records=jobs)
    out['complete']=all(s['complete'] for ds in out['summaries'].values() for s in ds.values())
    out['geometry_complete_and_feasible']=all(g['final_inverted_frames']==g['final_below_margin_frames']==g['failed_frames']==0 for g in out['geometry'].values()) and out['complete']
    return out

def text_report(out):
    lines=['POST-AUDIT FEASIBILITY EXTENSION — '+('COMPLETE' if out['complete'] else 'PARTIAL / INCOMPLETE'),out['created_utc'],out['scope'],'Original v12 evidence unchanged. Primary full-suffix metrics differ by dataset and are not pooled.','Failures remain explicit; means may be conditional on finite frames. Paired gains use complete pairs.','']
    for c,ds in out['summaries'].items():
        lines+=['COHORT '+c.upper()+(' — FRESH CONFIRMATION' if c=='fresh_confirmation' else ' — POST-AUDIT SECONDARY')]
        for dataset,s in ds.items():
            lines += [dataset+' '+s['metric'],'method | n/N | complete | mean mm | shape mm | failed frames | feasible wins/pairs | feasible gain mm']
            for m,v in s['methods'].items():
                p=v['paired'];lines.append(f"{m} | {v['present']}/{v['expected']} | {v['complete']} | {fmt(v['mean_mm'])} | {fmt(v['shape_mean_mm'])} | {v['failed_frames']} | {p['wins']}/{p['count']} | {fmt(p['mean_gain_mm'])}")
                if v['missing']:lines.append('Missing: '+', '.join(v['missing']))
                if finite(p['mean_gain_mm']) and p['mean_gain_mm']<0:lines.append('Feasible primary is WORSE than '+m+' on paired mean.')
            lines += ['Group | feasible | uncorrected | v11 (mm)']
            for groupname,g in out['groups'][c][dataset].items():lines.append(groupname+' | '+' | '.join(fmt(g['methods'][m]['mean_mm']) for m in [PRIMARY[dataset],UNCORRECTED[dataset],'v11_fem']))
            lines+=['Per-sequence: name | feasible | uncorrected | v11 (mm)']
            for r in out['sequences'][c]:
                if r['dataset']==dataset:lines.append(r['name']+' | '+' | '.join(fmt(r['results'].get(m,{}).get('summary',{}).get(METRIC[dataset])) for m in [PRIMARY[dataset],UNCORRECTED[dataset],'v11_fem']))
        g=out['geometry'][c];lines += ['GEOMETRY '+json.dumps({k:v for k,v in g.items() if k!='per_sequence'}),'name | guarded frames/fraction | proposal minJ | final minJ | final inversions/margin failures | max correction mm | alpha counts']
        for r in g['per_sequence']:
            lines.append(f"{r['name']} | {r['guarded_frames']}/{r['frames']} ({r['guarded_fraction']:.4f}) | {fmt(r['min_proposal_J'])} | {fmt(r['min_final_J'])} | {r['final_frames_with_inversions']}/{r['final_frames_below_margin']} | {fmt(r['max_guard_coordinate_rms_mm'])} | {r['alpha_counts']}")
            lines.append('Temporal changes: '+json.dumps(r['final_temporal_change']))
    lines += ['','LIMITATIONS','Local original-element J>0.03 plus numerical margin is checked at saved frames; it does not certify between-frame feasibility, global injectivity, collision freedom, equilibrium or material inference.','Deterministic backtracking is before contact and velocity initialization; no future-score choice. Original invalid v12 forecasts remain archived and are not called mechanically feasible.','Contact still uses static heightfields and freezes shape at separation; no coupled elastic impact or movable partner dynamics.','Fresh42 are new trajectories of known categories. Old106 repaired evaluations are post-audit secondary, never independent confirmation.','Fair native+scene postprocessing is separately labelled and original baselines remain visible. No calibrated uncertainty, active-probing or SOTA claim.']
    return '\n'.join(lines)+'\n'

def plot(out,path):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(out['cohorts']),2,figsize=(14,3.7*len(out['cohorts'])),squeeze=False)
    for ax,(c,d,s) in zip(axes.flat,[(c,d,s) for c,ds in out['summaries'].items() for d,s in ds.items()]):
        methods=['v11_fem',UNCORRECTED[d],PRIMARY[d],'rigid_scene_state' if d=='SoftVTBench' else 'official_phystwin']
        if d=='SoftVTBench' and c!='development':methods+=['adapted_phystwin_scene_state']
        rows=[(m,s['methods'][m]) for m in methods if m in s['methods'] and finite(s['methods'][m]['mean_mm'])]
        if rows:
            ax.barh(range(len(rows)),[v['mean_mm'] for _,v in rows],color=['#25796d' if m==PRIMARY[d] else '#738397' for m,_ in rows]);ax.set_yticks(range(len(rows)),[LABELS.get(m,m) for m,_ in rows]);ax.invert_yaxis()
            for i,(_,v) in enumerate(rows):ax.text(v['mean_mm'],i,f" {v['mean_mm']:.1f} ({v['present']}/{v['expected']})",va='center',fontsize=8)
            ax.set_xlim(0,max(v['mean_mm'] for _,v in rows)*1.35)
        else:ax.text(.5,.5,'No results yet',ha='center',transform=ax.transAxes)
        ax.set_title(c+' / '+d,fontsize=10);ax.set_xlabel(s['metric']);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Feasibility extension — fresh confirmation separate from post-audit secondary cohorts');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(path,dpi=160);plt.close(fig)

def paper(out,path):
    escape=lambda s:str(s).replace('_',r'\_')
    lines=[r'\noindent Post-audit secondary extension: a deterministic original-tetrahedron volume guard is applied before contact. Frozen v12 remains unchanged. Only the additional 42 sequences are fresh confirmation; they share known object categories.',r'\par '+('All required comparisons complete.' if out['complete'] else r'\textbf{Incomplete snapshot; coverage is explicit.}')]
    for c,ds in out['summaries'].items():
        lines += [r'\begin{table}[ht]\centering\small',r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrr}\toprule',r'Dataset & Method & $n/N$ & Error(mm) & Shape(mm) & Failed frames \\\midrule']
        for d,s in ds.items():
            for m in ['v11_fem',UNCORRECTED[d],PRIMARY[d],'rigid_scene_state' if d=='SoftVTBench' else 'official_phystwin']+(['adapted_phystwin_scene_state'] if d=='SoftVTBench' and c!='development' else []):
                v=s['methods'][m];lines.append(f"{d} & {escape(LABELS.get(m,m))} & {v['present']}/{v['expected']} & {fmt(v['mean_mm'])} & {fmt(v['shape_mean_mm'])} & {v['failed_frames']} "+r'\\')
        lines += [r'\bottomrule\end{tabular}}',r'\caption{'+escape(c)+r'. Full-suffix coordinate RMSE for SoftVTBench and manual-marker mean for PhysTwin; metrics not pooled.}',r'\end{table}']
        g=out['geometry'][c];lines.append(f"Guard changed {g['guarded_frames']}/{g['frames']} saved frames. Final inversion frames: {g['final_inverted_frames']}; margin-violation frames: {g['final_below_margin_frames']}; nonfinite frames: {g['failed_frames']}. "+r'\par')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',nargs='+',choices=COHORTS,default=COHORTS);a=p.parse_args();out=collect(a.cohorts)
    raw=(json.dumps(out,indent=2)+'\n').encode();(REPORT/'feasibility-results.json').write_bytes(raw);(REPORT/'feasibility-results.json.gz').write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    (REPORT/'FEASIBILITY-RESULTS.txt').write_text(text_report(out));plot(out,REPORT/'feasibility-comparison.png');paper(out,ROOT/'papers/contact-benchmark-v12/generated-feasibility-results.tex')
    print(json.dumps({'complete':out['complete'],'geometry_complete_and_feasible':out['geometry_complete_and_feasible']}))
if __name__=='__main__':main()
