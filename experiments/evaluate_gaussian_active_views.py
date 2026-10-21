"""Validation-only active RGB-D views, selected before outcomes are revealed.

Initial public video stops at1s. An actual saved-state camera at1.5s is acquired;
future motion and an independent same-state camera are evaluation outcomes.
The candidate pool assumes a repeatable replay/paused-state acquisition. It
does not claim a real robot can move instantaneously between camera poses.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

from evaluate_gaussian_inverse import load_model,encode_episode,posterior_samples,predict_particles
from gaussian_active_design import score_actions,condition,predictive_risk
from gaussian_acquisition_sensor import public_ray_queries,predict_ranges,measure_ranges,discrepancy_covariance
from isaac_acquisition_views import load_public_view_queries,selected_candidate_paths
from gaussian_silhouette_sensor import query as silhouette_query,predict_support,measure_support,covariance as silhouette_covariance


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as output:
        json.dump(value,output,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--inverse-checkpoint',type=Path,required=True);p.add_argument('--forward-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',nargs='*')
    p.add_argument('--samples',type=int,default=64);p.add_argument('--rays',type=int,default=24)
    p.add_argument('--fantasies',type=int,default=256);p.add_argument('--device',default='cpu')
    p.add_argument('--independent-noise-mm',type=float,default=2.)
    p.add_argument('--common-noise-mm',type=float,default=2.)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--sensor',choices=['range','silhouette'],default='range')
    p.add_argument('--angles',type=int,default=16)
    p.add_argument('--silhouette-noise-pixels',type=float,default=1.)
    p.add_argument('--target',choices=['future_motion','independent_view'],default='future_motion')
    p.add_argument('--random-expectation',action='store_true',help='Evaluate separate one-view random trials at every candidate; never combine their posteriors')
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--test-protocol',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'provenance').mkdir()
    for name in ['evaluate_gaussian_active_views.py','gaussian_active_design.py','gaussian_acquisition_sensor.py','gaussian_silhouette_sensor.py']:
        shutil.copy2(Path(__file__).with_name(name),a.output/'provenance'/name)
    plan=json.loads((a.dataset/'plan.json').read_text())
    allowed={c['episode_id'] for c in plan['episodes'] if c['split']==a.split}
    if a.split=='test':
        if not a.test_protocol:raise ValueError('TEST acquisition requires a frozen checkpoint protocol')
        from gaussian_final_protocol import load_protocol,require_model,collect_test_cases
        sealed=load_protocol(a.test_protocol);require_model(sealed,a.inverse_checkpoint,a.forward_checkpoint)
        collect_test_cases([a.dataset],sealed)
        allowed&=set(sealed['design']['view_episodes'])
    episodes=sorted(d.name for d in (a.pool/'public').iterdir() if d.is_dir())
    if a.episodes:episodes=[name for name in episodes if name in a.episodes]
    if not episodes or not set(episodes)<=allowed:raise ValueError('Acquisition episodes outside declared split/design')
    bundle=load_model(a.inverse_checkpoint,a.forward_checkpoint,a.device)
    expected_mode='rgbd' if a.sensor=='range' else 'rgb'
    if bundle['checkpoint']['model_kwargs'].get('observation_mode')!=expected_mode:
        raise ValueError('Range and silhouette acquisitions require matching RGBD/RGB checkpoints')
    covariance=(discrepancy_covariance(a.rays,a.independent_noise_mm/1000,a.common_noise_mm/1000)
                if a.sensor=='range' else silhouette_covariance(a.angles,a.silhouette_noise_pixels,a.silhouette_noise_pixels))
    protocol={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'episodes':episodes,'observation_prefix_s':1.,'acquisition_time_s':1.5,
              'target_times_s':[2.,2.2,2.4,2.6,2.8,3.],
              'policies':['predictive_risk','random','fixed0','max_observation_variance'],
              'sensor_covariance':covariance.tolist(),'sensor_units':'m' if a.sensor=='range' else 'pixels',
              'checkpoint_sha256':{'inverse':file_hash(a.inverse_checkpoint),'forward':file_hash(a.forward_checkpoint)},
              'qualification':a.split+' evaluation; working Gaussian sensor likelihood; calibration unestablished',
              'source_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in ['evaluate_gaussian_active_views.py','gaussian_active_design.py','gaussian_acquisition_sensor.py']}}
    atomic_json(a.output/'protocol.json',protocol);rows=[]
    for index,name in enumerate(episodes):
        started=time.monotonic();episode=a.dataset/'episodes'/name
        encoded=encode_episode(bundle,episode,max_time_s=1.)
        reference=encoded['inputs']['reference']
        particles=posterior_samples(bundle,encoded,a.samples,seed=a.seed+index)
        times=np.array([1.5,2.,2.2,2.4,2.6,2.8,3.])
        predicted=predict_particles(bundle,particles,reference,times).cpu().numpy()
        weights=np.full(a.samples,1/a.samples)
        public=load_public_view_queries(a.pool,name)
        candidates=np.flatnonzero(public['roles']=='candidate')
        evaluation=np.flatnonzero(public['roles']=='evaluation')
        if len(evaluation)!=1:raise ValueError('One independent evaluation camera required')
        if not np.allclose(public['times_s'][np.r_[candidates,evaluation]],1.5):
            raise ValueError('Candidate and independent cameras must observe the same physical state')
        def make_query(i):
            pose=public['camera_to_world_usd_row'][i]
            return (public_ray_queries(reference,pose,public['K'],a.rays,seed=a.seed)
                    if a.sensor=='range' else silhouette_query(reference,pose,public['K'],a.angles))
        def sensor_predictions(q):
            return predict_ranges(predicted[:,0],reference,q) if a.sensor=='range' else predict_support(predicted[:,0],q)
        queries=[make_query(i) for i in candidates]
        observation_predictions=np.stack([sensor_predictions(q) for q in queries])
        eval_id=int(evaluation[0]);eval_query=make_query(eval_id)
        eval_predictions=sensor_predictions(eval_query)
        targets=predicted[:,1:].reshape(a.samples,-1) if a.target=='future_motion' else eval_predictions
        scores=score_actions(targets,observation_predictions,covariance,weights,
                             fantasies=a.fantasies,seed=a.seed+index)
        selection={'episode_id':name,'selection':scores,
                   'chosen_indices':{'predictive_risk':scores['selected_index'],
                       'random':int(np.random.default_rng(a.seed+index).integers(len(candidates))),
                       'fixed0':0,
                       'nearest_evaluation_camera':int(np.argmax([
                           np.trace(public['camera_to_world_usd_row'][i,:3,:3]@public['camera_to_world_usd_row'][eval_id,:3,:3].T)
                           for i in candidates])),
                       'max_observation_variance':int(np.argmax(np.var(observation_predictions,axis=1).mean(-1)))},
                   'candidate_view_ids':public['view_ids'][candidates].tolist(),
                   'observed_frame_ids':encoded['inputs']['cameras']['frame_ids'].tolist(),
                   'measurement_files_read':[],
                   'selection_is_final_before_reveal':True}
        if a.random_expectation:
            for trial in range(len(candidates)):selection['chosen_indices']['random_trial'+str(trial)]=trial
        folder=a.output/name;folder.mkdir()
        # Durable, reviewable precommitment before ANY candidate pixels/depth
        # or evaluation state is loaded, including random/control policy choices.
        atomic_json(folder/'selection-before-reveal.json',selection)
        policy_weights={'passive':weights};revealed=[]
        for policy,chosen in selection['chosen_indices'].items():
            view_id=selection['candidate_view_ids'][chosen]
            paths=selected_candidate_paths(a.pool,name,view_id)
            if a.sensor=='range':
                with np.load(paths['depth_path']) as raw:measurement=measure_ranges(raw['depth_m'],queries[chosen])
                measurement_path=paths['depth_path']
            else:
                measurement_path=paths['mask_path']
                measurement=measure_support(np.asarray(Image.open(measurement_path))>127,queries[chosen])
            posterior,diagnostics=condition(observation_predictions[chosen],measurement,covariance,weights)
            policy_weights[policy]=posterior
            revealed.append({'policy':policy,'view_id':view_id,'measurement_sha256':file_hash(measurement_path),**diagnostics})
        # Outcomes are read only after all policy decisions are committed.
        with np.load(episode/'truth/trajectory.npz') as raw:
            ids=np.array([int(np.argmin(abs(raw['times_s']-t))) for t in times])
            if not np.allclose(raw['times_s'][ids],times):raise ValueError('Exact query times required')
            truth=raw['positions_m'][ids]
        eval_folder=a.pool/'truth'/name/'views'/str(public['view_ids'][eval_id])
        if a.sensor=='range':
            with np.load(eval_folder/'depth.npz') as raw:eval_truth=measure_ranges(raw['depth_m'],eval_query)
        else:eval_truth=measure_support(np.asarray(Image.open(eval_folder/'mask.png'))>127,eval_query)
        case={'episode_id':name,'policies':{},'revealed':revealed,
              'static_future_coordinate_rmse_m':float(np.sqrt(np.mean((reference['rest_points_m']-truth[1:])**2))),
              'seconds_elapsed':time.monotonic()-started}
        for policy,posterior in policy_weights.items():
            mean=np.tensordot(posterior,predicted,axes=1)
            case['policies'][policy]={'future_coordinate_rmse_m':float(np.sqrt(np.mean((mean[1:]-truth[1:])**2))),
                'independent_view_sensor_rmse':float(np.sqrt(np.mean((posterior@eval_predictions-eval_truth)**2))),
                'model_predictive_risk_m2':predictive_risk(targets,posterior),
                'effective_sample_size':float(1/np.square(posterior).sum())}
        if a.random_expectation:
            # Mean of errors over independent unit-budget policies, not an
            # ensemble mean conditioned on multiple acquired measurements.
            trials=[case['policies']['random_trial'+str(i)] for i in range(len(candidates))]
            case['policies']['random_expected']={key:float(np.mean([trial[key] for trial in trials])) for key in trials[0]}
        np.savez_compressed(folder/'posterior-predictions.npz',times_s=times,positions_m=predicted,
                            **{policy+'_weights':w for policy,w in policy_weights.items()})
        atomic_json(folder/'results.json',case);rows.append(case)
        print(json.dumps(case),flush=True)
    aggregate={policy:{metric:float(np.mean([row['policies'][policy][metric] for row in rows]))
                       for metric in ['future_coordinate_rmse_m','independent_view_sensor_rmse','effective_sample_size']}
               for policy in rows[0]['policies']}
    result={'episodes':rows,'aggregate':aggregate,'protocol':protocol}
    atomic_json(a.output/'results.json',result);print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
