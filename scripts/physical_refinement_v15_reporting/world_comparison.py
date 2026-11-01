"""Reporting-only all-world paired diagnostic; no prediction or selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
METRICS=('normalized_design_mse','aligned_shape_rmse_mm','coordinate_rmse_mm','reaction_force_rmse_N','reaction_torque_rmse_Nm')
BASELINES=('initial_only','fixed_all','visual_uncertainty')
TOLERANCE=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(report):
    final=[r for r in report['rows'] if r['is_final'] and r['policy'] in ('physical_risk',)+BASELINES]
    worlds=sorted({r['world'] for r in final})
    if len(worlds)!=12:raise ValueError('All twelve worlds required')
    rows=[]
    for track in ('V','VF'):
        for baseline in BASELINES:
            for world in worlds:
                active=[r for r in final if r['world']==world and r['track']==track and r['policy']=='physical_risk']
                other=[r for r in final if r['world']==world and r['track']==track and r['policy']==baseline]
                if len(active)!=4 or len(other)!=4 or {r['repeat'] for r in active}!={0,1,2,3} or {r['repeat'] for r in other}!={0,1,2,3}:raise ValueError('All four matched noise repeats required')
                for metric in METRICS:
                    a=float(np.mean([r['metrics'][metric] for r in active]));b=float(np.mean([r['metrics'][metric] for r in other]));delta=a-b
                    if not np.isfinite([a,b]).all():raise ValueError('No nonfinite result filtering')
                    rows.append(dict(track=track,baseline=baseline,world=world,initial_camera=active[0]['initial_camera'],metric=metric,
                                     physical_risk=a,baseline_value=b,active_minus_baseline=delta,
                                     outcome='near_tie' if abs(delta)<=TOLERANCE else 'win' if delta<0 else 'loss'))
    summary=[]
    for track in ('V','VF'):
        for baseline in BASELINES:
            for metric in METRICS:
                selected=[r for r in rows if r['track']==track and r['baseline']==baseline and r['metric']==metric]
                summary.append(dict(track=track,baseline=baseline,metric=metric,
                    wins=sum(r['outcome']=='win' for r in selected),losses=sum(r['outcome']=='loss' for r in selected),near_ties=sum(r['outcome']=='near_tie' for r in selected),
                    mean_active=float(np.mean([r['physical_risk'] for r in selected])),mean_baseline=float(np.mean([r['baseline_value'] for r in selected])),
                    mean_delta=float(np.mean([r['active_minus_baseline'] for r in selected])),
                    largest_improvement=min(selected,key=lambda r:r['active_minus_baseline'])['world'],largest_regression=max(selected,key=lambda r:r['active_minus_baseline'])['world']))
    return dict(scope='All twelve worlds, four matched noise repeats averaged before comparison; descriptive exploratory diagnostics, no per-world policy selection',
                absolute_near_tie_tolerance=TOLERANCE,rows=rows,summary=summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--scores-ready-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or sha(a.scores)!=a.scores_ready_sha256:raise ValueError('Fresh output and released input hash required')
    source=json.loads(a.scores.read_text());result=compare(source)
    result.update(input_scores_sha256=a.scores_ready_sha256,source_sha256=sha(__file__),protocol_sha256=source['protocol_sha256'],method_freeze_sha256=source['method_freeze_sha256'])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    lines=['Physical-risk comparison: equal weight per world, four repeats averaged.','Wins/losses/near-ties use absolute tolerance1e-12; all metrics lower is better.','']
    for row in result['summary']:
        lines.append(f"{row['track']:2s} versus {row['baseline']:18s} {row['metric']:26s}: {row['wins']} wins / {row['losses']} losses / {row['near_ties']} near-ties; mean {row['mean_active']:.8g} vs {row['mean_baseline']:.8g}")
    a.output.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
