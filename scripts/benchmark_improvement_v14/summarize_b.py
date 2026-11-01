"""PhaseB coverage, complete-pair and matched-frame comparisons, kept separate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import BASE,REPORT,read,write,sha
from scripts.benchmark_improvement_v14.summarize import compare


def main():
    source=REPORT/'softvt-scores-b.json';record=read(source);rows=record['cases'];protocol=read(REPORT/'protocol-b.json')
    expected={c['name']:c for c in protocol['regression_cases']+protocol['fresh_softvt_cases']}
    if len(rows)!=len(expected) or {r['case']['name']:r['case'] for r in rows}!=expected:raise ValueError('B cohort census mismatch')
    groups={}
    for row in rows:groups.setdefault(row['case']['cohort'],[]).append(row)
    groups['all_exposed_regression']=[r for r in rows if r['case']['cohort']!='prospective_softvt_phase_b']
    result=dict(dataset='SoftVTBench',phase='B: explicit numerical-failure fallback',freeze_sha256=record['freeze_sha256'],
                scores_sha256=sha(source),groups={},fallbacks=[],regression_identity=[])
    for row in rows:
        case=row['case'];folder=BASE/'robust/evaluation'/case['cohort']/case['name']
        meta=read(folder/'metadata.json')['benchmark_v14']
        if meta.get('method')=='validity_triggered_rigid_contact_fallback':
            result['fallbacks'].append(dict(case=case,applied=meta['fallback_applied'],original_nonfinite_frames=meta['original_nonfinite_frames'],failure_retained=meta.get('failure_retained')))
        if case['cohort']!='prospective_softvt_phase_b':
            with np.load(folder/'predictions.npz') as a:
                finite_original=bool(np.isfinite(a['contact_scene_state']).all())
                identical=bool(np.array_equal(a['benchmark_v14'],a['phase_a_prediction'],equal_nan=True))
            if finite_original and not identical:raise ValueError('Previously finite forecast changed: '+case['name'])
            result['regression_identity'].append(dict(name=case['name'],finite_original=finite_original,phase_a_identical=identical))
    for name,group in groups.items():
        values={method:compare(group,method,'coordinate_rmse_mm') for method in ['contact_scene_state','rigid_scene_state','prefix_calibrated_rigid','phase_a_prediction']}
        values['shape_vs_original']=compare(group,'contact_scene_state','aligned_shape_rmse_mm')
        values['matched_frame_comparisons']={}
        for method in ['contact_scene_state','rigid_scene_state','phase_a_prediction']:
            usable=[r['matched_frame_comparisons'][method] for r in group if method in r['matched_frame_comparisons'] and r['matched_frame_comparisons'][method]['count']]
            a=np.array([r['proposal']['coordinate_rmse_mm'] for r in usable]);b=np.array([r['baseline']['coordinate_rmse_mm'] for r in usable])
            values['matched_frame_comparisons'][method]=dict(cases=len(usable),declared_cases=len(group),
                proposed_mm=float(a.mean()) if len(a) else None,baseline_mm=float(b.mean()) if len(b) else None,
                scope='Same finite frames within each case; equal case weighting. Baseline missing frames remain failures, not imputed.')
        values['proposed_coverage']=dict(cases=len(group),complete_cases=sum(r['metrics']['benchmark_v14']['summary']['failed_frames']==0 for r in group),
            total_frames=sum(r['metrics']['benchmark_v14']['summary']['total_frames'] for r in group),failed_frames=sum(r['metrics']['benchmark_v14']['summary']['failed_frames'] for r in group))
        result['groups'][name]=values
    write(REPORT/'softvt-summary-b.json',result)
    for name,values in result['groups'].items():
        c=values['contact_scene_state'];r=values['rigid_scene_state']
        print(name,'original',c.get('baseline_mean_mm'),'new',c.get('proposal_mean_mm'),'rigid',r.get('baseline_mean_mm'),'coverage',values['proposed_coverage'])


if __name__=='__main__':main()
