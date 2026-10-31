"""Paired dataset-specific summaries; failures are explicit and never dropped silently."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import REPORT,read,write,sha


def compare(rows, baseline, metric):
    pairs=[];unavailable=[];coverage=[]
    for row in rows:
        methods=dict(row.get('archived_controls',{}),**row['metrics'])
        proposed=methods['benchmark_v14']['summary'];old=methods.get(baseline,{}).get('summary',{})
        for summary in [proposed,old]:
            if not summary:continue
            for key in ['total_frames','valid_frames','failed_frames']:
                value=summary.get(key)
                if not isinstance(value,int) or value<0:raise ValueError('Invalid frame count: '+key)
            if summary['valid_frames']+summary['failed_frames']!=summary['total_frames']:
                raise ValueError('Inconsistent coverage counts')
        coverage.append(dict(name=row['case']['name'],new_failed=proposed.get('failed_frames'),
                             old_failed=old.get('failed_frames'),new_valid=proposed.get('valid_frames'),
                             old_valid=old.get('valid_frames'),total=proposed.get('total_frames'),
                             old_total=old.get('total_frames')))
        a=proposed.get(metric);b=old.get(metric)
        if a is None or b is None or not np.isfinite([a,b]).all():
            unavailable.append(row['case']['name']);continue
        if a<0 or b<0:raise ValueError('Negative distance metric')
        pairs.append(dict(name=row['case']['name'],proposal_mm=a,baseline_mm=b,improvement_mm=b-a,
                          proposal_failed_frames=proposed.get('failed_frames',0),baseline_failed_frames=old.get('failed_frames',0),
                          complete_pair=proposed['total_frames']==old['total_frames'] and proposed['failed_frames']==old['failed_frames']==0))
    if not pairs:return dict(baseline=baseline,metric=metric,declared_cases=len(rows),available_pairs=0,
                             all_cases_and_frames_complete=False,unavailable=unavailable,coverage=coverage)
    a=np.array([p['proposal_mm'] for p in pairs]);b=np.array([p['baseline_mm'] for p in pairs]);delta=b-a
    rng=np.random.default_rng(140031);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    complete=not unavailable and all(c['new_valid']==c['old_valid']==c['total']==c['old_total'] and
                                    c['new_failed']==c['old_failed']==0 for c in coverage)
    complete_pairs=[p for p in pairs if p['complete_pair']]
    complete_summary=dict(count=len(complete_pairs),proposal_mean_mm=float(np.mean([p['proposal_mm'] for p in complete_pairs])) if complete_pairs else None,
                          baseline_mean_mm=float(np.mean([p['baseline_mm'] for p in complete_pairs])) if complete_pairs else None,
                          interpretation='Descriptive complete-pair subset; not a substitute for full-cohort coverage.')
    return dict(baseline=baseline,metric=metric,available_pairs=len(pairs),declared_cases=len(rows),
        proposal_mean_mm=float(a.mean()),baseline_mean_mm=float(b.mean()),mean_improvement_mm=float(delta.mean()),
        relative_improvement_percent=float(100*delta.mean()/b.mean()) if b.mean() else None,
        wins=int((delta>1e-8).sum()),losses=int((delta< -1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),
        paired_episode_bootstrap_95_interval_mm=np.quantile(boot,[.025,.975]).tolist(),
        uncertainty_note='Descriptive paired episode bootstrap; repeated objects, not independent-object generalization.',
        all_cases_and_frames_complete=complete,unavailable=unavailable,coverage=coverage,per_case=pairs,complete_pairs=complete_summary,
        interpretation='Means are over available scored frames/cases; consult coverage before any performance claim.' if not complete else 'All declared cases and frames included.')


def summarize(dataset):
    filename='softvt-scores.json' if dataset=='SoftVTBench' else 'phystwin-scores.json'
    source=REPORT/filename;record=read(source);rows=record['cases']
    protocol=read(REPORT/'protocol.json')
    expected={c['name']:c for c in protocol['historical_cases']+protocol['fresh_softvt_cases'] if c['dataset']==dataset}
    names=[r['case']['name'] for r in rows]
    if len(set(names))!=len(names) or {r['case']['name']:r['case'] for r in rows}!=expected:
        raise ValueError('Score cohort differs from exact protocol census')
    metric='coordinate_rmse_mm' if dataset=='SoftVTBench' else 'manual_track_mean_mm'
    baseline='contact_scene_state' if dataset=='SoftVTBench' else 'object_frame'
    methods=[baseline,'model_selection_selected','prefix_calibrated_rigid','rigid_scene_state'] if dataset=='SoftVTBench' else [baseline,'model_selection_selected','official_phystwin','position_ridge']
    groups={}
    for row in rows:groups.setdefault(row['case']['cohort'],[]).append(row)
    groups['all_historical_exposed']=[r for r in rows if r['case']['cohort']!='prospective_softvt_confirmation']
    summary=dict(dataset=dataset,primary_metric=metric,primary_baseline=baseline,source_sha256=sha(source),
                 freeze_sha256=record['freeze_sha256'],groups={})
    for name,group in groups.items():
        if not group:continue
        summary['groups'][name]={method:compare(group,method,metric) for method in methods}
        summary['groups'][name]['shape_vs_original']=compare(group,baseline,'aligned_shape_rmse_mm')
    write(REPORT/('softvt-summary.json' if dataset=='SoftVTBench' else 'phystwin-summary.json'),summary)
    for name,group in summary['groups'].items():
        p=group[baseline];print(dataset,name,p.get('baseline_mean_mm'),'->',p.get('proposal_mean_mm'),
                              'wins/losses',p.get('wins'),p.get('losses'),'complete',p.get('all_cases_and_frames_complete'))


if __name__=='__main__':
    summarize(sys.argv[1])
