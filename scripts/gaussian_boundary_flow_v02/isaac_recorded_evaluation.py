"""Independent recorded-trajectory scorer, blocked until all380 predictions seal.

This module is not imported by prediction. No interpolation, node matching,
posterior fitting, or simulator rerun. Metrics are against recorded Isaac motion,
not a claim of continuum convergence or real-world physical calibration.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from gaussian_operator import physical_vector,category_labels,PhysicalParameterTransform
from scripts.gaussian_boundary_flow_v02 import isaac_prediction_cohort as prediction
from scripts.gaussian_boundary_flow_v02.isaac_prefix_cache import sha,write_json


def freeze(cohort,output):
    cohort=Path(cohort).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'evaluation-plan.json'
    if path.exists():raise FileExistsError('Evaluation plan immutable')
    protocol=prediction.verify_protocol(cohort)
    plan=dict(cohort=str(cohort),protocol_sha256=sha(cohort/'protocol.json'),source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)},
              gate='Existing verified380terminal barrier required BEFORE any test plan, boundaries, trajectory or reference read',
              expected_cases=76,methods=list(prediction.KINDS),primary_time_indices=list(range(40,61)),times_s=(np.arange(61)/20).tolist(),nodes=455,
              tolerances=dict(prediction_clock_atol_s=2e-7,recorded_clock_atol_s=1e-10,measured_simulation_clock_atol_s=3e-7,initial_node_position_atol_m=1e-6,saved_mean_atol_m=2e-7,saved_mean_rtol=2e-6),
              metrics=['posterior_mean_coordinate_RMSE','per_frame_rigid_aligned_shape_RMSE','pointwise_coordinate90_coverage_width_Winkler','empirical_joint_trajectory_energy_score','active_physical_code_interval_coverage_width_error','four_material_physical_units','category_modal_accuracy_and_sample_frequency_Brier','raw_code_bound_saturation'],
              energy_score='All21 primary times x455nodes x3coordinates; Euclidean norm divided bysqrt(D); empirical128sample distribution V-statistic including diagonal, ridge1sample',
              parameters='Truth-family active scored coordinates, not intrinsic dimensionality; deterministic force1 duplicate excluded; canonical effectivecode used for parameter intervals; no inactivepadding success credit',
              aggregation='All76 denominators, separate64neural/12orbit; episode and layoutgroup macro means; paired shared-success differences and explicit exclusions;10000layoutgroup bootstrap seed61043',
              qualifications=['historically exposed test cohort','recorded simulator motion, not numerical or realphysical certification','shape metric removes optimal rigid pose perframe using truth for evaluation only','pointwise90percent coordinatecoverage is not joint90percent confidence','teacher-only validation selected models; measured accuracy evaluated here afterwards','archived chart/25D and ridge have different supervision; new flow/MDN are matched label-supervision comparison'],
              prediction_source_binding=protocol['source_sha256'])
    write_json(path,plan);return plan


def require_gate(output):
    output=Path(output);plan=json.loads((output/'evaluation-plan.json').read_text());cohort=Path(plan['cohort'])
    if plan['source_sha256']!={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)}:raise ValueError('Evaluation source changed after freeze')
    if sha(cohort/'protocol.json')!=plan['protocol_sha256']:raise ValueError('Prediction protocol changed')
    # Existence checked before barrier verification: evaluator cannot create a gate.
    if not(cohort/'barrier.json').exists():raise ValueError('All380 prediction barrier is absent; targets forbidden')
    barrier=prediction.barrier(cohort)
    if barrier['terminal']!=380 or barrier['expected_cases']!=76:raise ValueError('Incomplete prediction barrier')
    protocol=prediction.verify_protocol(cohort)
    return plan,protocol,barrier


def rigid_aligned_shape_rmse(mean,target):
    a=np.asarray(mean,dtype=np.float64);b=np.asarray(target,dtype=np.float64)
    a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
    u,_,vh=np.linalg.svd(np.swapaxes(a,1,2)@b)
    correction=np.broadcast_to(np.eye(3),(len(a),3,3)).copy();correction[:,-1,-1]=np.linalg.det(u@vh)
    aligned=a@(u@correction@vh)
    return float(np.sqrt(np.mean((aligned-b)**2)))


def trajectory_metrics(samples,target,rest):
    samples=np.asarray(samples,dtype=np.float64);target=np.asarray(target,dtype=np.float64);rest=np.asarray(rest,dtype=np.float64)
    if samples.ndim!=4 or samples.shape[1:]!=target.shape or rest.shape!=target.shape[1:] or not np.isfinite(samples).all() or not np.isfinite(target).all():raise ValueError('Invalid trajectory metric input')
    mean=samples.mean(0);lower,upper=np.quantile(samples,[.05,.95],axis=0)
    width=upper-lower;interval=width+20*np.maximum(lower-target,0)+20*np.maximum(target-upper,0)
    vectors=(samples-rest).reshape(len(samples),-1);truth=(target-rest).reshape(-1);d=vectors.shape[1]
    norms=np.square(vectors).sum(-1);squared=np.maximum((norms[:,None]+norms[None]-2*vectors@vectors.T)/d,0.)
    np.fill_diagonal(squared,0.);pair=np.sqrt(squared)
    energy=np.sqrt(np.square(vectors-truth).mean(-1)).mean()-.5*pair.mean()
    return dict(coordinate_rmse_m=float(np.sqrt(np.mean((mean-target)**2))),rigid_aligned_shape_rmse_m=rigid_aligned_shape_rmse(mean,target),
                coordinate90_coverage=float(((target>=lower)&(target<=upper)).mean()),coordinate90_width_m=float(width.mean()),coordinate90_interval_score_m=float(interval.mean()),
                trajectory_energy_score_m=float(energy),samples=int(len(samples)))


def active_coordinates(boundary,force):
    active=np.zeros(25,dtype=bool);active[:7]=True;active[10:13]=True;active[16:19]=True
    if boundary==2:active[7:10]=True
    if force==3:active[13:16]=True
    if boundary==3:active[22:25]=True
    return active


def parameter_metrics(arrays,physical,boundary,force):
    values=np.asarray(arrays['effective_physical'],dtype=np.float64);raw=np.asarray(arrays['physical'],dtype=np.float64)
    truth=np.asarray(physical,dtype=np.float64);active=active_coordinates(boundary,force)
    if values.ndim!=2 or values.shape[1]!=25 or raw.shape!=values.shape or not np.isfinite(values).all() or not np.isfinite(raw).all():raise ValueError('Invalid physical sample array')
    lower,upper=np.quantile(values,[.05,.95],axis=0);transform=PhysicalParameterTransform();span=(transform.upper-transform.lower).numpy()
    covered=(truth>=lower)&(truth<=upper);normerr=np.abs(values.mean(0)-truth)/span
    materials={}
    for i,(name,logarithmic) in enumerate([('youngs_modulus_Pa',True),('density_kg_m3',True),('poissons_ratio',False),('damping_s_inverse',True)]):
        samples=np.exp(values[:,i]) if logarithmic else values[:,i];actual=float(np.exp(truth[i]) if logarithmic else truth[i]);lo,hi=np.quantile(samples,[.05,.95])
        materials[name]=dict(truth=actual,posterior_mean=float(samples.mean()),absolute_error=float(abs(samples.mean()-actual)),interval90=[float(lo),float(hi)],covered=bool(lo<=actual<=hi),width=float(hi-lo))
    categories={}
    for name,actual in [('boundary_type',boundary),('force_type',force)]:
        labels=np.asarray(arrays[name])
        if labels.shape!=(len(values),) or labels.dtype.kind not in 'iu' or ((labels<0)|(labels>3)).any():raise ValueError('Invalid category samples')
        probs=np.bincount(labels,minlength=4)/len(labels);onehot=np.eye(4)[actual]
        categories[name]=dict(actual=int(actual),probabilities=probs.tolist(),correct=bool(probs.argmax()==actual),brier=float(np.square(probs-onehot).sum()))
    fraction=(raw-transform.lower.numpy())/span
    return dict(active_coordinate_ids=np.flatnonzero(active).tolist(),active_coordinate_count=int(active.sum()),active_parameter90_coverage=float(covered[active].mean()),
                active_parameter90_normalized_width=float(((upper-lower)/span)[active].mean()),active_parameter_normalized_absolute_error=float(normerr[active].mean()),
                active_parameter90_covered=covered[active].tolist(),materials=materials,categories=categories,raw_bound_saturation_fraction=float(((fraction<1e-4)|(fraction>1-1e-4)).mean()))


def validate_identity(arrays,public,truth,reference,times,simulated_times,expected_samples):
    expected=np.arange(61)/20;positions=arrays['positions_m'];target=truth
    if positions.shape!=(expected_samples,61,455,3) or target.shape!=(61,455,3):raise ValueError('Trajectory/node/sample shape mismatch; no interpolation')
    if not np.isfinite(positions).all() or not np.isfinite(target).all():raise ValueError('Nonfinite trajectory')
    if np.asarray(times).shape!=(61,) or not np.allclose(times,expected,atol=1e-10,rtol=0):raise ValueError('Recorded clock mismatch; no interpolation')
    if np.asarray(simulated_times).shape!=(61,) or not np.allclose(simulated_times,expected,atol=3e-7,rtol=0):raise ValueError('Measured simulation clock mismatch')
    if arrays['times_s'].shape!=(61,) or not np.allclose(arrays['times_s'],expected,atol=2e-7,rtol=0):raise ValueError('Forecast clock mismatch')
    rest=reference.astype(np.float32)
    if rest.shape!=(455,3) or not np.array_equal(rest,public['rest']) or not np.array_equal(rest,arrays['rest_points_m']):raise ValueError('Reference identity/node order mismatch')
    if not np.allclose(target[0],reference,atol=1e-6,rtol=0):raise ValueError('Recorded initial node identity mismatch')
    if not np.array_equal(arrays['input_frame_ids'],public['frame_ids']) or not np.array_equal(arrays['input_times_s'],public['times']) or not np.allclose(public['times'],[0,.25,.5,.75,1.],atol=1e-7,rtol=0):raise ValueError('Prefix budget identity mismatch')
    if arrays['mean_positions_m'].shape!=(61,455,3) or not np.allclose(arrays['mean_positions_m'],positions.mean(0),atol=2e-7,rtol=2e-6):raise ValueError('Saved posterior mean differs from samples')
    if arrays['physical'].shape!=(expected_samples,25) or arrays['effective_physical'].shape!=(expected_samples,25):raise ValueError('Physical sample budget differs')


def load_truth_after_gate(row,protocol):
    """Called ONLY by run after verified global barrier; no prediction imports it."""
    episode=Path(row['episode']);dataset_root=episode.parent.parent
    historical=json.loads(Path(protocol['exposed_seal']).read_text());declared=next(d for d in historical['datasets'] if Path(d['root']).resolve()==dataset_root.resolve())
    plan_path=dataset_root/'plan.json'
    if sha(plan_path)!=declared['plan_sha256']:raise ValueError('Recorded case recipe plan changed')
    plan=json.loads(plan_path.read_text());recipe=next(r for r in plan['episodes'] if r['episode_id']==row['episode_id'])
    frozen=next(r for r in declared['cases'] if r['episode_id']==row['episode_id'])
    if any(recipe[k]!=v for k,v in frozen.items()):raise ValueError('Recorded case identity changed')
    paths=[episode/'truth/trajectory.npz',episode/'truth/boundaries.json',episode/'reference.npz',plan_path]
    with np.load(paths[0],allow_pickle=False) as f:
        truth=f['positions_m'];times=f['times_s'];simulated=f['simulated_times_s']
    with np.load(paths[2],allow_pickle=False) as f:reference=f['rest_points_m']
    spec=json.loads(paths[1].read_text());physical=physical_vector(recipe,spec,reference);bc,force=category_labels(recipe)
    return dict(target=truth,times=times,simulated_times=simulated,reference=reference,physical=physical,boundary=bc,force=force,source_sha256={str(p):sha(p) for p in paths})


SUMMARY_METRICS=['coordinate_rmse_m','rigid_aligned_shape_rmse_m','coordinate90_coverage','coordinate90_width_m','coordinate90_interval_score_m','trajectory_energy_score_m','active_parameter90_coverage','active_parameter90_normalized_width','active_parameter_normalized_absolute_error','raw_bound_saturation_fraction']

def summarize(rows,protocol):
    result=dict(expected_cases=76,methods={},datasets={},paired={},qualification='Recorded Isaac forecast accuracy; not teacher-only consistency or real-world physical certification')
    for scope,cases in [('all',protocol['cases'])]+[(d,[r for r in protocol['cases'] if r['dataset']==d]) for d in sorted({r['dataset'] for r in protocol['cases']})]:
        ids={r['case'] for r in cases};table={}
        for method in prediction.KINDS:
            selected=[r for r in rows if r['case'] in ids and r['model']==method];good=[r for r in selected if r['state']=='complete']
            groups={}
            for r in good:groups.setdefault((r['dataset'],r['group']),[]).append(r)
            table[method]=dict(expected=len(cases),complete=len(good),failed=len(selected)-len(good),missing=len(cases)-len(selected),
                               excluded_cases=[r['case'] for r in selected if r['state']!='complete'],
                               episode_macro={k:float(np.mean([r['metrics'][k] for r in good])) if good else None for k in SUMMARY_METRICS},
                               layout_group_macro={k:float(np.mean([np.mean([r['metrics'][k] for r in g]) for g in groups.values()])) if groups else None for k in SUMMARY_METRICS},layout_groups=len(groups),
                               category_accuracy={name:float(np.mean([r['metrics']['categories'][name]['correct'] for r in good])) if good else None for name in ('boundary_type','force_type')},
                               category_brier={name:float(np.mean([r['metrics']['categories'][name]['brier'] for r in good])) if good else None for name in ('boundary_type','force_type')})
        if scope=='all':result['methods']=table
        else:result['datasets'][scope]=table
        pairs={};bykey={(r['case'],r['model']):r for r in rows}
        for baseline in ('mdn','chart','25d','ridge'):
            good=[r for r in cases if bykey.get((r['case'],'flow'),{}).get('state')=='complete' and bykey.get((r['case'],baseline),{}).get('state')=='complete']
            entry=dict(expected=len(cases),paired_cases=len(good),excluded_cases=sorted(ids-{r['case'] for r in good}),metrics={})
            for metric in ['coordinate_rmse_m','rigid_aligned_shape_rmse_m','trajectory_energy_score_m']:
                groups={};differences=[]
                for row in good:
                    difference=bykey[(row['case'],'flow')]['metrics'][metric]-bykey[(row['case'],baseline)]['metrics'][metric]
                    differences.append(difference);groups.setdefault((row['dataset'],row['group']),[]).append(difference)
                values=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(61043)
                ci=np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(1),[.025,.975]).tolist() if len(values)>=2 else None
                entry['metrics'][metric]=dict(episode_mean_flow_minus_baseline=float(np.mean(differences)) if differences else None,layout_group_mean_flow_minus_baseline=float(values.mean()) if len(values) else None,layout_group_bootstrap95=ci,layout_groups=len(values),negative_favors_flow=True)
            pairs[baseline]=entry
        result['paired'][scope]=pairs
    return result


def verify_existing_score(record,row,kind,case_dir,binding):
    if record['binding']!=binding or (record['case'],record['dataset'],record['group'],record['model'])!=(row['case'],row['dataset'],row['group'],kind):raise ValueError('Existing score identity/binding changed')
    status_path=case_dir/kind/'status.json'
    if record['prediction_status_sha256']!=sha(status_path):raise ValueError('Existing score prediction status changed')
    status=json.loads(status_path.read_text())
    if record['state']=='complete':
        if status['state']!='complete' or record['prediction_seal_sha256']!=sha(case_dir/kind/'prediction-seal.json') or any(sha(p)!=digest for p,digest in record['target_source_sha256'].items()):raise ValueError('Existing score prediction/target binding changed')
    elif record['state']=='prediction_failed':
        if status['state']!='failed':raise ValueError('Existing prediction failure state changed')
    elif record['state']=='evaluation_failed':
        if status['state']!='complete' or not record.get('error_type'):raise ValueError('Existing evaluation failure state changed')
    else:raise ValueError('Unknown evaluation status')


def run(output):
    output=Path(output);plan,protocol,barrier=require_gate(output);cohort=Path(plan['cohort']);binding=dict(evaluation_plan_sha256=sha(output/'evaluation-plan.json'),barrier_sha256=sha(cohort/'barrier.json'),protocol_sha256=plan['protocol_sha256'])
    summary_path=output/'summary.json'
    if summary_path.exists():
        previous=json.loads(summary_path.read_text())
        expected_paths={str(Path(row['case'])/f'{kind}.json') for row in protocol['cases'] for kind in prediction.KINDS}
        if previous['binding']!=binding or set(previous['score_sha256'])!=expected_paths or any(sha(output/path)!=digest for path,digest in previous['score_sha256'].items()):raise ValueError('Completed score summary changed; no overwrite')
        for row in protocol['cases']:
            for kind in prediction.KINDS:
                record=json.loads((output/row['case']/f'{kind}.json').read_text())
                verify_existing_score(record,row,kind,cohort/row['case'],binding)
        return previous
    rows=[]
    for row in protocol['cases']:
        case_dir=cohort/row['case'];destination=output/row['case'];destination.mkdir(exist_ok=True)
        paths=[destination/f'{kind}.json' for kind in prediction.KINDS]
        if all(p.exists() for p in paths):
            existing=[json.loads(p.read_text()) for p in paths]
            for kind,record in zip(prediction.KINDS,existing):verify_existing_score(record,row,kind,case_dir,binding)
            rows.extend(existing);continue
        if any(p.exists() for p in paths):raise ValueError('Partial scored case preserved; inspect without replacing scores')
        try:
            truth=load_truth_after_gate(row,protocol)
            with np.load(case_dir/'public-prefix.npz',allow_pickle=False) as f:public={k:f[k] for k in f.files}
            error=None
        except Exception as exc:error=(type(exc).__name__,str(exc))
        for kind,path in zip(prediction.KINDS,paths):
            score=dict(case=row['case'],dataset=row['dataset'],group=row['group'],model=kind,binding=binding)
            status=json.loads((case_dir/kind/'status.json').read_text());score['prediction_status_sha256']=sha(case_dir/kind/'status.json')
            if status['state']!='complete':score.update(state='prediction_failed',error=status.get('error'),prediction_status_sha256=sha(case_dir/kind/'status.json'))
            else:
                try:
                    if error:raise ValueError('Independent target/identity read failed: '+repr(error))
                    with np.load(case_dir/kind/'predictions.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
                    validate_identity(arrays,public,truth['target'],truth['reference'],truth['times'],truth['simulated_times'],protocol['models'][kind]['samples'])
                    metrics=trajectory_metrics(arrays['positions_m'][:,40:61],truth['target'][40:61],public['rest'])
                    metrics.update(parameter_metrics(arrays,truth['physical'],truth['boundary'],truth['force']))
                    score.update(state='complete',metrics=metrics,target_source_sha256=truth['source_sha256'],prediction_seal_sha256=sha(case_dir/kind/'prediction-seal.json'))
                except Exception as exc:score.update(state='evaluation_failed',error_type=type(exc).__name__,error=str(exc))
            write_json(path,score);rows.append(score)
        write_json(output/'progress.json',dict(cases=len(rows)//5,expected=76,binding=binding))
    summary=summarize(rows,protocol);summary['binding']=binding;summary['score_sha256']={str(p.relative_to(output)):sha(p) for row in protocol['cases'] for p in [output/row['case']/f'{k}.json' for k in prediction.KINDS]}
    write_json(output/'summary.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);p.add_argument('--cohort',default='runs/gaussian-boundary-isaac-exposed-v01');p.add_argument('--output',default='reports/gaussian-boundary-flow-v02/isaac-recorded-evaluation');a=p.parse_args()
    if a.command=='freeze':freeze(a.cohort,a.output)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):run(a.output)
