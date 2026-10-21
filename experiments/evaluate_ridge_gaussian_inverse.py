"""Evaluate sealed deterministic ridge models; no training or TEST selection.

A prefix physically loads only its observed RGB/mask/depth files. Features are
aligned into the checkpoint's training time grid; unavailable future rows remain
zero. Checkpoint identities are checked before any TEST target access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

EXPECTED_PROTOCOL='4d6149df81801fab4cddd752b43e0ff251cc66761f0c7e3ee7d2e485955d725d'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_prefix_features(feature_shape,training_times,cutoff,features,observed_times):
    features=np.asarray(features);observed_times=np.asarray(observed_times);training_times=np.asarray(training_times)
    keep=training_times<=cutoff+1e-8
    if features.shape!=(int(keep.sum()),feature_shape[-1]) or observed_times.shape!=(int(keep.sum()),):raise ValueError('Prefix observation count/feature shape differs from sealed training clock')
    if not np.isfinite(observed_times).all() or np.any(observed_times>cutoff+1e-8):raise ValueError('Future timestamp supplied to prefix predictor')
    np.testing.assert_allclose(observed_times,training_times[keep],atol=1e-7,rtol=0)
    result=np.zeros(feature_shape,dtype=features.dtype);result[keep]=features
    return result


def evaluate(roots,inverse_checkpoint,forward_checkpoint,output,test_protocol,threads=2):
    import torch
    from gaussian_final_protocol import load_protocol,require_model,collect_test_cases
    from ridge_gaussian_inverse import RidgeInverse,predict_forward
    from train_gaussian_inverse import load_frozen_operator
    from evaluate_gaussian_inverse import load_public_observations
    from gaussian_observed_features import extract_features
    from gaussian_operator import category_labels
    started=time.monotonic();torch.set_num_threads(threads)
    sealed=load_protocol(test_protocol)
    if sealed['protocol_sha256']!=EXPECTED_PROTOCOL:raise ValueError('Unexpected final TEST seal; do not silently switch protocols')
    require_model(sealed,inverse_checkpoint,forward_checkpoint)
    checkpoint_hashes={'inverse':sha(inverse_checkpoint),'forward':sha(forward_checkpoint)}
    model,metadata=RidgeInverse.load(inverse_checkpoint)
    if metadata['operator_checkpoint_sha256']!=checkpoint_hashes['forward']:raise ValueError('Ridge teacher differs from sealed teacher')
    for name in ['ridge_gaussian_inverse.py','gaussian_observed_features.py']:
        if sha(Path(__file__).with_name(name))!=metadata['source_sha256'][name]:raise ValueError('Ridge inference feature source differs from frozen training source')
    operator=load_frozen_operator(forward_checkpoint,'cpu')
    records=collect_test_cases(roots,sealed)
    if not records:raise ValueError('No sealed TEST episodes')
    if len({r['recipe']['episode_id'] for r in records})!=len(records):raise ValueError('Episode filename collision across datasets')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);provenance=root/'provenance';provenance.mkdir()
    sources=['evaluate_ridge_gaussian_inverse.py','ridge_gaussian_inverse.py','evaluate_gaussian_inverse.py',
             'gaussian_final_protocol.py','train_gaussian_inverse.py','gaussian_observed_features.py',
             'gaussian_observation.py','gaussian_operator.py','gaussian_modal_operator.py']
    source_hashes={}
    for name in sources:
        data=Path(__file__).with_name(name).read_bytes();(provenance/name).write_bytes(data);source_hashes[name]=hashlib.sha256(data).hexdigest()
    rows=[];input_hashes={}
    for record in records:
        episode=record['path'];inputs=load_public_observations(episode,model.observation_mode,tuple(metadata['size']),model.observation_cutoff_s)
        measured=extract_features(inputs)['features'];cameras=inputs['cameras']
        if model.texture_calibration is not None:raise ValueError('Final sealed ridge models use geometry_v1; do not guess texture backend')
        if model.observation_cutoff_s is not None:
            measured=align_prefix_features(model.feature_shape,model.observation_times_s,model.observation_cutoff_s,measured,cameras['times_s'])
        prediction=model.predict_features(measured)
        # Freeze physical predictions before opening query cameras or motion.
        physical_before=prediction['physical'].copy()
        with np.load(episode/'truth/cameras.npz') as raw:query={k:raw[k].copy() for k in raw.files}
        ref=inputs['reference'];rest=ref['rest_points_m']
        q=predict_forward(operator,prediction,rest,query['times_s'],'cpu')[0]
        if not np.isfinite(q).all():raise ValueError('Nonfinite frozen prediction')
        np.savez_compressed(root/(episode.name+'-prediction.npz'),positions_m=q,times_s=query['times_s'],frame_ids=query['frame_ids'],physical=physical_before,boundary_type=prediction['boundary_type'],force_type=prediction['force_type'])
        # Evaluation only: model and predicted physical coefficients stay fixed.
        with np.load(episode/'truth/trajectory.npz') as raw:target=raw['positions_m']
        np.testing.assert_array_equal(physical_before,prediction['physical'])
        gap=(query['times_s']>1)&(query['times_s']<2);late=(query['times_s']>=2)&(query['times_s']<=3)
        tet=ref['tetrahedra'];ratio=np.linalg.det(q[:,tet[:,1:]]-q[:,tet[:,:1]])/np.linalg.det(rest[tet[:,1:]]-rest[tet[:,:1]])
        bc,force=category_labels(record['recipe'])
        score_frames=np.array([10,20,30,40,50,60]);score_nodes=np.linspace(0,len(rest)-1,64).astype(int)
        row={'key':record['key'],'dataset':record['dataset'],'episode_id':episode.name,'group_id':record['recipe']['group_id'],
             'coordinate_rmse_m':float(np.sqrt(np.mean((q-target)**2))),
             'static_coordinate_rmse_m':float(np.sqrt(np.mean((rest-target)**2))),
             'gap_coordinate_rmse_m':float(np.sqrt(np.mean((q[gap]-target[gap])**2))),
             'late_2to3_coordinate_rmse_m':float(np.sqrt(np.mean((q[late]-target[late])**2))),
             'static_gap_coordinate_rmse_m':float(np.sqrt(np.mean((rest-target[gap])**2))),
             'static_late_2to3_coordinate_rmse_m':float(np.sqrt(np.mean((rest-target[late])**2))),
             'predictive_energy_score_6times64nodes_m':float(np.sqrt(np.mean((q[score_frames][:,score_nodes]-target[score_frames][:,score_nodes])**2))),
             'boundary_correct':bool(prediction['boundary_type'][0]==bc),'force_correct':bool(prediction['force_type'][0]==force),
             'mean_trajectory_finite':True,'mean_trajectory_positive_volume':bool((ratio>0).all()),
             'mean_trajectory_minimum_volume_ratio':float(ratio.min()),'mean_trajectory_inverted_tetrahedron_count':int((ratio<=0).sum()),
             'observed_frames':cameras['frame_ids'].tolist(),'max_observed_time_s':float(cameras['times_s'].max())}
        rows.append(row)
        public=[episode/'reference.npz',episode/'observations/cameras.npz']
        for frame in cameras['frame_ids']:
            public.extend(episode/'observations'/folder/(f'{frame:05d}'+suffix) for folder,suffix in [('rgb','.png'),('masks','.png')]+([('depth','.npz')] if model.observation_mode=='rgbd' else []))
        input_hashes.update({str(p):sha(p) for p in public})
        print(json.dumps(row),flush=True)
    keys=['coordinate_rmse_m','static_coordinate_rmse_m','gap_coordinate_rmse_m','late_2to3_coordinate_rmse_m','static_gap_coordinate_rmse_m','static_late_2to3_coordinate_rmse_m','predictive_energy_score_6times64nodes_m','boundary_correct','force_correct']
    def aggregate(selected):return {k:float(np.mean([r[k] for r in selected])) for k in keys}
    result={'episodes':rows,'aggregate':aggregate(rows),'aggregate_by_dataset':{d:aggregate([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})},
            'checkpoint_sha256':checkpoint_hashes,'inverse_checkpoint':str(inverse_checkpoint),'forward_checkpoint':str(forward_checkpoint),
            'final_test_protocol_sha256':sealed['protocol_sha256'],'source_sha256':source_hashes,'public_input_sha256':input_hashes,
            'model_parameters_updated':False,'observation_cutoff_s':model.observation_cutoff_s,'observation_mode':model.observation_mode,
            'future_input_pixels_opened':False if model.observation_cutoff_s is not None else 'Full10input condition',
            'uncertainty':'Deterministic point baseline; no calibrated intervals or class probabilities; energy score is point-mass Euclidean error',
            'seconds_elapsed':time.monotonic()-started,'device':'cpu','qualification':'Primary fresh64 and secondary original12 reported separately; no fitting, hyperparameter changes or TEST selection'}
    require_model(load_protocol(test_protocol),inverse_checkpoint,forward_checkpoint)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True);p.add_argument('--inverse-checkpoint',required=True);p.add_argument('--forward-checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--test-protocol',required=True);p.add_argument('--threads',type=int,default=2)
    result=evaluate(**vars(p.parse_args()));print(json.dumps(result['aggregate_by_dataset'],indent=2))
