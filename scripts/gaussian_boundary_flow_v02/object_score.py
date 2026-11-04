"""Separate post-seal scorer for historically exposed PhysTwin object pilots.

This module is never imported by fitting/inference workers. It scores official
suffix frames and cannot turn nodewise marginal quantiles into point quantiles.
"""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import (
    PHYSTWIN, load_phystwin_predictor, load_phystwin_evaluation, predictor_digest)
from scripts.external_benchmark.data import phystwin_position_mask


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand_points(prediction,nodes,observation_matrix,observed_initial):
    """Linear map applies to complete trajectories/samples, never marginal bounds."""
    prediction=np.asarray(prediction);nodes=np.asarray(nodes);A=np.asarray(observation_matrix)
    initial=np.asarray(observed_initial)
    if prediction.shape[-2:]!=nodes.shape or A.shape!=(len(initial),len(nodes)):
        raise ValueError('Prediction/observation geometry shapes do not match')
    return initial+np.einsum('pn,...nc->...pc',A,prediction-nodes)


def aligned_shape_rmse(prediction,target):
    if len(prediction)<3:return None
    a=prediction-prediction.mean(0);b=target-target.mean(0)
    U,_,Vh=np.linalg.svd(a.T@b)
    d=np.eye(3);d[-1,-1]=np.linalg.det(U@Vh)
    aligned=a@(U@d@Vh)+target.mean(0)
    return float(np.sqrt(np.mean((aligned-target)**2))*1000)


def geometry_score(prediction,target,mask,start,end):
    prediction=np.asarray(prediction);target=np.asarray(target);mask=np.asarray(mask,bool)
    if prediction.shape!=target.shape or mask.shape!=target.shape[:2]:
        raise ValueError('Full-point prediction/target/mask shapes must match')
    if not 0<=start<end<=len(target):raise ValueError('Invalid scoring interval')
    rows=[]
    for t in range(start,end):
        valid=mask[t]&np.isfinite(target[t]).all(-1)
        if not np.isfinite(prediction[t]).all():
            rows.append(dict(frame=t,status='failed_nonfinite_prediction'));continue
        if not valid.any():
            rows.append(dict(frame=t,status='no_valid_reference'));continue
        p,y=prediction[t,valid],target[t,valid];error=p-y
        rows.append(dict(frame=t,status='scored',points=int(valid.sum()),
                         coordinate_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
                         mean_l2_mm=float(np.linalg.norm(error,axis=-1).mean()*1000),
                         vector_rms_mm=float(np.sqrt(np.mean(np.sum(error**2,axis=-1)))*1000),
                         aligned_shape_rmse_mm=aligned_shape_rmse(p,y)))
    good=[r for r in rows if r['status']=='scored']
    summary={key:float(np.mean([r[key]for r in good if r[key]is not None]))
             if any(r[key]is not None for r in good)else None
             for key in ('coordinate_rmse_mm','mean_l2_mm','vector_rms_mm','aligned_shape_rmse_mm')}
    summary.update(total_frames=end-start,valid_frames=len(good),
                   failed_frames=sum(r['status']=='failed_nonfinite_prediction'for r in rows),
                   no_reference_frames=sum(r['status']=='no_valid_reference'for r in rows),
                   aggregation='Equal frame means within this sequence; do not pool frame counts across sequences')
    return dict(summary=summary,frames=rows)


def manual_score(prediction,initial,manual,start,end):
    """Match initial manual points once; no future correspondence optimization."""
    manual=np.asarray(manual);valid0=np.isfinite(manual[0]).all(-1)
    if not valid0.any():return dict(mean_distance_mm=None,valid_frames=0)
    ids=cKDTree(initial).query(manual[0,valid0])[1];values=[]
    for t in range(start,min(end,len(manual))):
        y=manual[t,valid0];valid=np.isfinite(y).all(-1)
        if valid.any() and np.isfinite(prediction[t,ids[valid]]).all():
            values.append(float(np.linalg.norm(prediction[t,ids[valid]]-y[valid],axis=-1).mean()*1000))
    return dict(mean_distance_mm=float(np.mean(values))if values else None,valid_frames=len(values),
                correspondence='Nearest initial observed point; fixed across time, not future-fitted')


def prefix_baselines(payload,frame_ids=None,point_ids=None):
    """Last observation and last-two velocity; match inference's observation budget.

    Predictions at unobserved points use reference-geometry inverse-distance
    interpolation of displacement. No evaluation labels or suffix frames enter.
    """
    end=int(payload['train_end']);times=np.asarray(payload['times'])
    frames=np.arange(end)if frame_ids is None else np.asarray(frame_ids,int)
    ids=np.arange(len(payload['observed_initial']))if point_ids is None else np.asarray(point_ids,int)
    if not len(frames)or(frames<0).any()or(frames>=end).any():raise ValueError('Baseline frame beyond prefix')
    if not len(ids)or(ids<0).any()or(ids>=len(payload['observed_initial'])).any():raise ValueError('Invalid point IDs')
    frames=np.unique(frames);ids=np.unique(ids)
    initial=np.asarray(payload['observed_initial']);obs=np.asarray(payload['observed'])[frames][:,ids]
    valid=np.asarray(payload['observed_mask'])[frames][:,ids]&np.isfinite(obs).all(-1)
    anchor=initial[ids].copy();anchor_time=np.full(len(ids),times[0]);velocity=np.zeros_like(anchor)
    for j in range(len(ids)):
        seen=np.flatnonzero(valid[:,j])
        if len(seen):
            last=seen[-1];anchor[j]=obs[last,j];anchor_time[j]=times[frames[last]]
        if len(seen)>1:
            last,prev=seen[-1],seen[-2];dt=times[frames[last]]-times[frames[prev]]
            if dt<=0:raise ValueError('Observation times must increase')
            velocity[j]=(obs[last,j]-obs[prev,j])/dt
    dist,near=cKDTree(initial[ids]).query(initial,k=min(4,len(ids)))
    if dist.ndim==1:dist,near=dist[:,None],near[:,None]
    weights=1/np.maximum(dist,1e-10)**2;weights/=weights.sum(-1,keepdims=True)
    fixed=anchor-initial[ids]
    extrapolated=fixed[None]+(times[:,None]-anchor_time[None])[...,None]*velocity[None]
    result={'prefix_persistence':np.broadcast_to(initial+np.einsum('pk,pkc->pc',weights,fixed[near]),(len(times),*initial.shape)).copy(),
            'prefix_velocity':initial[None]+np.einsum('pk,tpkc->tpc',weights,extrapolated[:,near])}
    return result


def interval_score(joint_node_samples,nodes,A,initial,target,mask,start,end):
    if joint_node_samples is None:
        return dict(available=False,reason='Joint posterior rollouts were not stored; interpolated nodewise marginal quantiles are not valid pointwise posterior quantiles')
    samples=np.asarray(joint_node_samples)
    if samples.ndim!=4 or samples.shape[0]<2:raise ValueError('Need at least two complete joint trajectory samples')
    points=expand_points(samples,nodes,A,initial)
    if points.shape[1:]!=np.asarray(target).shape:raise ValueError('Joint posterior samples must span the complete target clock and geometry')
    if not np.isfinite(points).all():raise ValueError('Nonfinite posterior sample; do not silently drop it in scoring')
    lo,hi=np.quantile(points,[.05,.95],axis=0)
    valid=np.asarray(mask[start:end])&np.isfinite(target[start:end]).all(-1)
    covered=(target[start:end]>=lo[start:end])&(target[start:end]<=hi[start:end])
    if not valid.any():return dict(available=False,reason='No valid reference points')
    return dict(available=True,nominal_coverage=.9,
                coordinate_coverage=float(covered[valid].mean()),
                average_width_mm=float((hi[start:end]-lo[start:end])[valid].mean()*1000),
                samples=len(samples),transport='Joint node trajectories mapped before pointwise quantiles')


def load_sealed(run):
    run=Path(run)
    for name in ('complete.json','inference-sealed.json','plan.json','prediction.npz'):
        if not(run/name).exists():raise ValueError('Prediction is not complete/sealed: '+name)
    complete=json.loads((run/'complete.json').read_text());seal=json.loads((run/'inference-sealed.json').read_text());plan=json.loads((run/'plan.json').read_text())
    if not complete.get('completed')or complete.get('held_target_access')is not False or seal.get('forecast_target_access')is not False:
        raise ValueError('Completion/target-access contract failed')
    if sha(run/'prediction.npz')!=complete['prediction_sha256']or sha(run/'plan.json')!=seal['plan_sha256']:
        raise ValueError('Changed sealed prediction or plan')
    if (run/'bank.npz').exists()and sha(run/'bank.npz')!=seal['bank_sha256']:raise ValueError('Changed training bank')
    provenance={'inference_sha256','models_sha256','transform_sha256'}
    if provenance.intersection(seal):
        if not provenance.issubset(seal) or seal.get('schema')!=1:
            raise ValueError('Incomplete or unsupported checkpoint provenance seal')
        required={'inference-sealed.npz':seal['inference_sha256'],'transform.pt':seal['transform_sha256'],'bank.npz':seal['bank_sha256']}
        models=seal['models_sha256']
        if not isinstance(models,dict)or not models:raise ValueError('Invalid sealed model manifest')
        for name,digest in models.items():
            if not isinstance(name,str)or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',name):raise ValueError('Invalid sealed model name')
            required[name+'.pt']=digest
        for filename,digest in required.items():
            path=run/filename
            if not path.is_file() or sha(path)!=digest:raise ValueError('Changed sealed provenance: '+filename)
    if complete['fit_end']!=seal['train_end']:raise ValueError('Inconsistent observed cutoff')
    with np.load(run/'prediction.npz',allow_pickle=False)as f:pred={k:f[k].copy()for k in f.files}
    return plan,seal,complete,pred


def load_inference_payload(plan,seal,root=PHYSTWIN):
    """Prefer the frozen numerical package; old pilots retain re-export support."""
    package=plan.get('input_package')
    if package:
        from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
        directory=Path(package)
        if not directory.is_absolute():directory=Path(__file__).resolve().parents[2]/directory
        if plan.get('input_package_manifest_sha256') and sha(directory/'manifest.json')!=plan['input_package_manifest_sha256']:
            raise ValueError('Changed sealed input-package manifest')
        payload=load_package(directory)
        if payload['name']!=plan['case'] or payload['dataset']!='PhysTwin':
            raise ValueError('Input package is not the requested PhysTwin case')
        if int(payload['train_end'])!=int(seal['train_end']):raise ValueError('Input package cutoff differs from inference seal')
        if len(payload['nodes'])!=int(plan['args']['nodes']):raise ValueError('Input package node count differs from inference plan')
    else:
        payload=load_phystwin_predictor(plan['case'],max_nodes=int(plan['args']['nodes']),prefix_end=seal['train_end'],root=root)
    if predictor_digest(payload)!=plan['input_digest']:
        raise ValueError('Predictor inputs differ from sealed inference inputs')
    return payload


def prediction_methods(plan,pred,frames):
    """Validate declared trajectory channels before opening evaluation targets."""
    names=plan.get('prediction_methods',['flow','mixture'])
    if not isinstance(names,list)or not names or any(not isinstance(n,str)for n in names):
        raise ValueError('prediction_methods must be a nonempty list of names')
    if len(set(names))!=len(names):raise ValueError('Duplicate prediction method')
    reserved={'nodes','observation_matrix','observed_initial','times','frames','point_ids',
              'prefix_frames','prefix_persistence','prefix_velocity','lo','hi','samples',
              'controller_targets','attachment_weights','support_geometry','observed','observed_mask'}
    shape=(int(frames),len(pred['nodes']),3)
    for name in names:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',name)or name in reserved or name.endswith(('_samples','_lo','_hi')):
            raise ValueError('Reserved or invalid prediction method: '+name)
        if name not in pred:raise ValueError('Missing declared prediction method: '+name)
        array=np.asarray(pred[name])
        if array.shape!=shape or array.dtype.kind not in 'fiu':
            raise ValueError('Prediction method must be a numeric full node trajectory: '+name)
        samples=pred.get(name+'_samples')
        if samples is not None:
            samples=np.asarray(samples)
            if samples.ndim!=4 or samples.shape[0]<2 or samples.shape[1:]!=shape or samples.dtype.kind not in 'fiu':
                raise ValueError('Joint posterior samples must span the complete trajectory: '+name)
    return names


def support_screen(diagnostic):
    if diagnostic is None:return dict(out_of_bank_warning=None,support_screening='Unavailable; no support diagnostic stored')
    if 'out_of_bank_warning' in diagnostic:
        flag=diagnostic['out_of_bank_warning']
        if not isinstance(flag,bool):raise ValueError('Explicit support warning must be boolean')
        return dict(out_of_bank_warning=flag,support_screening='Stored explicit support decision; see feature_support and diagnostic provenance')
    keys=('real_standardized_context_norm','bank_context_norm_max')
    flag=bool(diagnostic[keys[0]]>diagnostic[keys[1]])if all(k in diagnostic for k in keys)else None
    return dict(out_of_bank_warning=flag,support_screening='Historical PCA-norm-only fallback: incomplete screening, missing discarded-space support check; not a positive adequacy certificate')


def score_case(run,root=PHYSTWIN):
    plan,seal,complete,pred=load_sealed(run);case=plan['case']
    # Load target data only after all seal checks and prefix identity verification.
    payload=load_inference_payload(plan,seal,root)
    for name in ('nodes','observation_matrix','observed_initial'):
        if not np.array_equal(pred[name],payload[name]):raise ValueError('Changed predictor geometry: '+name)
    split=json.loads((Path(root)/case/'split.json').read_text());start=int(split['train'][1]);end=int(split['test'][1])
    if end!=len(payload['times'])or complete['frames']!=end:raise ValueError('Incomplete or mismatched forecast clock')
    declared_methods=prediction_methods(plan,pred,end)
    evaluation=load_phystwin_evaluation(case,root=root)
    with (Path(root)/case/'final_data.pkl').open('rb')as f:raw=pickle.load(f)
    incoming=phystwin_position_mask(raw,len(evaluation['points']))
    target=evaluation['points'];official=np.asarray(evaluation['mask'],bool)
    methods={};intervals={}
    for name in declared_methods:
        methods[name]=expand_points(pred[name],pred['nodes'],pred['observation_matrix'],pred['observed_initial'])
        intervals[name]=interval_score(pred.get(name+'_samples'),pred['nodes'],pred['observation_matrix'],pred['observed_initial'],target,incoming,start,end)
    methods.update(prefix_baselines(payload,plan['prefix_frames'],plan['point_ids']))
    scores={}
    for name,positions in methods.items():
        scores[name]=dict(incoming_v22=geometry_score(positions,target,incoming,start,end),
                          official_visibility=geometry_score(positions,target,official,start,end),
                          manual=manual_score(positions,payload['observed_initial'],evaluation['manual'],start,end))
    diagnostic_path=Path(run)/'context-diagnostic.json'
    diagnostic=json.loads(diagnostic_path.read_text())if diagnostic_path.exists()else None
    return dict(prefix_support_diagnostic=diagnostic,**support_screen(diagnostic),
                parameter_interpretation='Neither point estimates nor stored samples establish calibrated physical parameter inference; identification and uncertainty require separate validation',
                case=case,scope='Historically exposed PhysTwin; sealed geometry-specific prediction methods, not zero-shot transfer',
                interval=[start,end],fit_end=seal['train_end'],nodes=len(payload['nodes']),
                scoring_mask='Primary: visibility[t] and incoming transition[t-1], with finite reference; official visibility-only secondary explicitly separate',
                baseline_budget=dict(frames=plan['prefix_frames'],point_ids=plan['point_ids'],scope='Same selected prefix observations as inference; last observation / last-two velocity with initial-geometry interpolation'),
                methods=scores,posterior_intervals=intervals,forecast_samples=complete.get('forecast_samples'),
                archived_comparison=dict(included=False,reason='Archived v22 uses 240 nodes, 18 fitting frames and different optimization/representation. Native published metrics also use different mapping/protocol; do not paste unmatched numbers into this table.',reference='reports/real-transfer-v22/protocol.json'),
                source_sha256={str(Path(run)/'prediction.npz'):sha(Path(run)/'prediction.npz'),str(Path(run)/'inference-sealed.json'):sha(Path(run)/'inference-sealed.json'),str(Path(root)/case/'final_data.pkl'):sha(Path(root)/case/'final_data.pkl')},
                evaluator_only=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/gaussian-boundary-flow-v02-objects/single_lift_sloth');parser.add_argument('--output');args=parser.parse_args()
    result=score_case(args.run);out=Path(args.output)if args.output else Path(args.run)/'scores.json'
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    print(json.dumps({k:v['incoming_v22']['summary']for k,v in result['methods'].items()},indent=2))

if __name__=='__main__':main()
