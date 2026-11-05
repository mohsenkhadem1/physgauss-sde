"""Post hoc static-geometry control; prediction and target scoring are separate phases."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02 import isaac_recorded_evaluation as primary
from scripts.gaussian_boundary_flow_v02.isaac_prefix_cache import sha, write_json

DEFAULT = 'reports/gaussian-boundary-flow-v02/isaac-static-secondary'
PRIMARY = 'reports/gaussian-boundary-flow-v02/isaac-recorded-evaluation'


def static_prediction(rest):
    rest = np.asarray(rest)
    if rest.shape != (455, 3) or not np.isfinite(rest).all():
        raise ValueError('Expected finite public initial geometry with 455 ordered nodes')
    return np.repeat(rest[None], 61, axis=0)


def amplitude_ratio(shape, coordinate):
    if not np.isfinite([shape, coordinate]).all() or min(shape, coordinate) < 0:
        raise ValueError('Invalid amplitude')
    return None if coordinate == 0 else float(shape / coordinate)


def freeze(output, primary_output=PRIMARY):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists(): raise FileExistsError('Immutable secondary plan exists')
    plan, protocol, _ = primary.require_gate(primary_output)
    cohort = Path(plan['cohort']); summary = Path(primary_output) / 'summary.json'
    # Existing complete primary results are explicitly acknowledged before the post hoc rule.
    previous = json.loads(summary.read_text())
    if len(previous['score_sha256']) != 380: raise ValueError('Expected all 380 primary records')
    for rel, digest in previous['score_sha256'].items():
        if sha(Path(primary_output) / rel) != digest: raise ValueError('Changed primary record')
    record = dict(scope='RETROSPECTIVE post hoc static-initial-geometry baseline, not fresh confirmation',
                  rule='Repeat supplied public rest geometry of 455 ordered nodes at all 61 times; no fitting or target access for construction',
                  primary_output=str(Path(primary_output).resolve()), primary_summary_sha256=sha(summary),
                  cohort=str(cohort), cases=protocol['cases'], expected_cases=76, primary_indices=list(range(40,61)),
                  source_sha256={str(Path(__file__).resolve()):sha(__file__)},
                  public_sha256={r['case']:sha(cohort/r['case']/'public-prefix.npz') for r in protocol['cases']},
                  qualification='Coordinate RMS measures motion amplitude from initial geometry; rigid-aligned shape RMS measures residual deformation. Their ratio is a descriptive RMS ratio, not a variance decomposition. No material or force inference claims.')
    write_json(output/'plan.json', record)
    return sha(output/'plan.json')


def verify(output):
    output=Path(output); plan=json.loads((output/'plan.json').read_text())
    if any(sha(p)!=h for p,h in plan['source_sha256'].items()): raise ValueError('Secondary source changed')
    _,protocol,_=primary.require_gate(plan['primary_output'])
    if protocol['cases']!=plan['cases'] or len(plan['cases'])!=76: raise ValueError('Cohort identity changed')
    summary=Path(plan['primary_output'])/'summary.json'
    if sha(summary)!=plan['primary_summary_sha256']: raise ValueError('Primary summary changed')
    for rel,h in json.loads(summary.read_text())['score_sha256'].items():
        if sha(Path(plan['primary_output'])/rel)!=h: raise ValueError('Primary score changed')
    for case,h in plan['public_sha256'].items():
        if sha(Path(plan['cohort'])/case/'public-prefix.npz')!=h: raise ValueError('Public input changed')
    return plan,protocol


def predict(output):
    output=Path(output); plan,_=verify(output)
    for row in plan['cases']:
        folder=output/row['case']; folder.mkdir(exist_ok=True)
        if (folder/'seal.json').exists(): raise FileExistsError('No automatic replacement of static predictions')
        with np.load(Path(plan['cohort'])/row['case']/'public-prefix.npz',allow_pickle=False) as f:
            rest=f['rest']; frames=f['frame_ids']; times=f['times']
        mean=static_prediction(rest)
        np.savez_compressed(folder/'prediction.npz',mean_positions_m=mean,rest_points_m=rest,times_s=np.arange(61)/20,input_frame_ids=frames,input_times_s=times)
        write_json(folder/'seal.json',dict(case=row['case'],plan_sha256=sha(output/'plan.json'),public_sha256=plan['public_sha256'][row['case']],prediction_sha256=sha(folder/'prediction.npz')))
    return barrier(output,plan)


def barrier(output,plan):
    output=Path(output); seals={}
    # Verify the entire cohort before returning permission to read any targets.
    for row in plan['cases']:
        folder=output/row['case']; seal=json.loads((folder/'seal.json').read_text())
        if seal!={'case':row['case'],'plan_sha256':sha(output/'plan.json'),'public_sha256':plan['public_sha256'][row['case']],'prediction_sha256':sha(folder/'prediction.npz')}:
            raise ValueError('Changed static prediction seal')
        seals[row['case']]=sha(folder/'seal.json')
    return seals


def score(output):
    output=Path(output); plan,protocol=verify(output); seals=barrier(output,plan)
    if (output/'summary.json').exists(): raise FileExistsError('Secondary results immutable')
    write_json(output/'all-predictions-barrier.json',dict(plan_sha256=sha(output/'plan.json'),seals=seals,complete=76))
    rows=[]
    for row in plan['cases']:
        folder=output/row['case']; path=folder/'score.json'
        if path.exists(): raise FileExistsError('Partial secondary scoring preserved; no automatic retry')
        record={k:row[k] for k in ('case','dataset','group')}
        try:
            truth=primary.load_truth_after_gate(row,protocol)
            with np.load(folder/'prediction.npz',allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
            with np.load(Path(plan['cohort'])/row['case']/'public-prefix.npz',allow_pickle=False) as f: public={k:f[k] for k in f.files}
            arrays.update(positions_m=arrays['mean_positions_m'][None],physical=np.zeros((1,25)),effective_physical=np.zeros((1,25)))
            primary.validate_identity(arrays,public,truth['target'],truth['reference'],truth['times'],truth['simulated_times'],1)
            values=primary.trajectory_metrics(arrays['positions_m'][:,40:61],truth['target'][40:61],public['rest'])
            metrics={k:values[k] for k in ('coordinate_rmse_m','rigid_aligned_shape_rmse_m')}
            metrics['nonrigid_to_total_rms_ratio']=amplitude_ratio(metrics['rigid_aligned_shape_rmse_m'],metrics['coordinate_rmse_m'])
            record.update(state='complete',metrics=metrics,target_source_sha256=truth['source_sha256'])
            comparisons={}
            for kind in primary.prediction.KINDS:
                p=Path(plan['primary_output'])/row['case']/(kind+'.json'); old=json.loads(p.read_text())
                if old['case']!=row['case'] or old['dataset']!=row['dataset']: raise ValueError('Primary score case mismatch')
                if old['state']=='complete' and old['target_source_sha256']!=truth['source_sha256']: raise ValueError('Primary target identity mismatch')
                comparisons[kind]=dict(state=old['state'],primary_score_sha256=sha(p),difference={k:old['metrics'][k]-metrics[k] for k in ('coordinate_rmse_m','rigid_aligned_shape_rmse_m')} if old['state']=='complete' else None)
            record['primary_minus_static']=comparisons
        except Exception as exc: record.update(state='evaluation_failed',error_type=type(exc).__name__,error=str(exc))
        record['prediction_seal_sha256']=seals[row['case']];write_json(path,record);rows.append(record)
    summary=dict(scope=plan['scope'],qualification=plan['qualification'],plan_sha256=sha(output/'plan.json'),subsets={})
    for name,subset in [('all',rows)]+[(d,[r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})]:
        good=[r for r in subset if r['state']=='complete'];ratios=[r['metrics']['nonrigid_to_total_rms_ratio'] for r in good if r['metrics']['nonrigid_to_total_rms_ratio'] is not None]
        item=dict(expected=len(subset),complete=len(good),failed=len(subset)-len(good),zero_motion_ratio_undefined=sum(r['metrics']['nonrigid_to_total_rms_ratio'] is None for r in good),
                  mean_nonrigid_to_total_ratio=float(np.mean(ratios)) if ratios else None,
                  means={k:float(np.mean([r['metrics'][k] for r in good])) if good else None for k in ('coordinate_rmse_m','rigid_aligned_shape_rmse_m')},paired={})
        for kind in primary.prediction.KINDS:
            paired=[r['primary_minus_static'][kind]['difference'] for r in good if r['primary_minus_static'][kind]['state']=='complete']
            item['paired'][kind]=dict(expected=len(subset),complete_pairs=len(paired),means={k:float(np.mean([r[k] for r in paired])) if paired else None for k in ('coordinate_rmse_m','rigid_aligned_shape_rmse_m')},coordinate_wins=sum(r['coordinate_rmse_m']<0 for r in paired))
        summary['subsets'][name]=item
    summary['score_sha256']={r['case']:sha(output/r['case']/'score.json') for r in rows}
    write_json(output/'summary.json',summary); return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','predict','score']);parser.add_argument('--output',default=DEFAULT);args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        result=globals()[args.phase](args.output)
    print(json.dumps(result if args.phase=='freeze' else {'phase':args.phase,'complete':True}))
