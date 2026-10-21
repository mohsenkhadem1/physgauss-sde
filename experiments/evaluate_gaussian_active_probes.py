"""Choose actual calibrated reset experiments before revealing their RGB-D.

Material/support hypotheses come from passive observations. Known handle
locations are common extra side information, substituted into every policy's
forward query; the current encoder does not use this side information to refine
the latent material/support posterior. Controls replace the original unknown
forces. Querying a learned trajectory remains a direct neural evaluation.
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
from evaluate_gaussian_inverse import load_model,encode_episode,posterior_samples
from gaussian_active_design import score_actions,condition,predictive_risk
from gaussian_acquisition_sensor import public_ray_queries,predict_ranges,measure_ranges,discrepancy_covariance
from gaussian_operator import deformation_diagnostics
from isaac_calibrated_probes import record_selection,load_selected_measurement


def material_summary(particles,weights,case):
    """Evaluation only: weighted uncertainty and calibration after choices."""
    physical=particles['physical'].cpu().numpy();weights=np.asarray(weights)
    result={}
    for name,index,key in [('youngs_modulus_Pa',0,'youngs_modulus_Pa'),('density_kg_m3',1,'density_kg_m3')]:
        values=np.exp(physical[:,index]);order=np.argsort(values)
        cumulative=np.cumsum(weights[order]);cumulative[-1]=1.
        lower,median,upper=np.interp([.05,.5,.95],cumulative,values[order])
        truth=case['material'][key]
        result[name]={'lower90':float(lower),'median':float(median),'upper90':float(upper),
                      'width90':float(upper-lower),'covered90':bool(lower<=truth<=upper),'absolute_median_error':float(abs(median-truth))}
    bc=particles['boundary_type'].cpu().numpy()
    result['boundary_probabilities']=[float(weights[bc==i].sum()) for i in range(4)]
    return result


def predict_control(bundle,particles,reference,menu,action,chunk_size=8):
    device=bundle['device'];rest=torch.tensor(reference['rest_points_m'],device=device,dtype=torch.float32)
    times=torch.tensor(menu['sensor']['times_s'],device=device,dtype=torch.float32)
    physical=particles['effective_physical'].clone()
    physical[:,10:16]=physical.new_tensor(menu['known_handle_centers_m']).flatten()
    physical[:,16:22]=0;physical[:,16:19]=physical.new_tensor(action['peak_force_N'])
    family=torch.full_like(particles['force_type'],action.get('force_type_index',0));predictions=[]
    with torch.no_grad():
        for start in range(0,len(physical),chunk_size):
            stop=start+chunk_size
            predictions.append(bundle['forward'](rest,times,physical[start:stop],particles['boundary_type'][start:stop],family[start:stop]))
    return torch.cat(predictions).cpu().numpy()


def flatten_ranges(predicted,reference,query,frame_ids=None):
    frame_ids=range(1,predicted.shape[1]) if frame_ids is None else frame_ids
    return np.concatenate([predict_ranges(predicted[:,i],reference,query) for i in frame_ids],axis=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inverse-checkpoint',type=Path,required=True);p.add_argument('--forward-checkpoint',type=Path,required=True)
    p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--samples',type=int,default=128)
    p.add_argument('--rays',type=int,default=12);p.add_argument('--fantasies',type=int,default=512)
    p.add_argument('--noise-mm',type=float,default=.5);p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--sensor-frame-stride',type=int,default=1)
    p.add_argument('--random-expectation',action='store_true')
    p.add_argument('--censored-calibration',type=Path,help='TRAIN-only frozen mixed visibility/range model')
    a=p.parse_args();torch.set_num_threads(2)
    if a.sensor_frame_stride<1:p.error('Positive sensor frame stride required')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'provenance').mkdir()
    source_names=['evaluate_gaussian_active_probes.py','gaussian_active_design.py','gaussian_acquisition_sensor.py']
    if a.censored_calibration:source_names+=['censored_range_observation.py','calibrate_gaussian_probe_discrepancy.py']
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name),a.output/'provenance'/name)
    bundle=load_model(a.inverse_checkpoint,a.forward_checkpoint,a.device)
    if bundle['checkpoint']['model_kwargs'].get('observation_mode')!='rgbd':raise ValueError('Probe range acquisition requires explicitly RGBD inference')
    plan=json.loads((a.dataset/'plan.json').read_text());cases={c['episode_id']:c for c in plan['episodes']};allowed={c['episode_id'] for c in plan['episodes'] if c['split']=='validation'}
    if not set(a.episodes)<=allowed:raise ValueError('Development runner permits validation only')
    calibration=None
    if a.censored_calibration:
        from censored_range_observation import (CensoredRangeModel,predict_censored_ranges,measure_censored_depth,
            concatenate_observations,score_censored_actions)
        from calibrate_gaussian_probe_discrepancy import probe_discrepancy_covariance
        calibration=json.loads(a.censored_calibration.read_text())
        if calibration['forward_checkpoint_sha256']!=hashlib.sha256(a.forward_checkpoint.read_bytes()).hexdigest():
            raise ValueError('Frozen discrepancy calibration uses a different controlled teacher')
        for key,value in [('rays',a.rays),('sensor_frame_stride',a.sensor_frame_stride),('query_seed',a.seed)]:
            if calibration[key]!=value:raise ValueError('Acquisition query differs from TRAIN calibration: '+key)
        if calibration['roles_sha256']!='6d2781a71a24ebfa9cb0691ce0dd7ae889c15eae1b38ca3164618523d5552c84':
            raise ValueError('Unexpected discrepancy calibration roles')
        shutil.copy2(a.censored_calibration,a.output/'frozen-censored-calibration.json')
    protocol={key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}
    protocol.update({'initial_observations':'All10passiveframes; subsequent experiments reset the same object',
        'checkpoint_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('inverse',a.inverse_checkpoint),('forward',a.forward_checkpoint)]},
        'sensor_budget':'One action returns its declared RGBDsequence; same fixed frame stride and action count within probe policies',
        'known_hardware':'Both original handle centers are common public side information for all policies',
        'likelihood':'Working Gaussian on range residuals; within-frame common discrepancy; no exact physical-Gaussian claim',
        'qualifications':['Validation only','Substituting known hardware does not condition the flow on that geometry','Linear Green mechanics is approximate','No claim about equal view/probe time cost']})
    if calibration:protocol.update(likelihood='TRAIN-frozen mixed visibility and valid-range likelihood; same generative model in design/conditioning',
        calibration_sha256=hashlib.sha256(a.censored_calibration.read_bytes()).hexdigest(),
        max_variance_baseline='Mean variance of zero-filled valid residual/diagonal_sigma plus visibility bits; heuristic only')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=[]
    for index,name in enumerate(a.episodes):
        started=time.monotonic();episode=a.dataset/'episodes'/name
        passed=json.loads((a.pool/'private'/name/'validation.json').read_text())
        if not passed.get('passed'):raise ValueError('Actual probe pool has not passed independent audit')
        menu=json.loads((a.pool/'public'/name/'action-menu.json').read_text())
        encoded=encode_episode(bundle,episode);reference=encoded['inputs']['reference']
        particles=posterior_samples(bundle,encoded,a.samples,seed=a.seed+index)
        query=public_ray_queries(reference,np.array(menu['sensor']['camera_to_world_usd_row']),np.array(menu['sensor']['K']),a.rays,seed=a.seed)
        targets=predict_control(bundle,particles,reference,menu,menu['evaluation_query'])
        candidates=[predict_control(bundle,particles,reference,menu,action) for action in menu['candidates']]
        sensor_frame_ids=np.arange(1,targets.shape[1],a.sensor_frame_stride)
        count=len(sensor_frame_ids)
        weights=np.full(a.samples,1/a.samples)
        target_features=targets[:,1:].reshape(a.samples,-1)
        if calibration:
            candidate_features=[predict_censored_ranges(q,reference,query,sensor_frame_ids) for q in candidates]
            noise=probe_discrepancy_covariance(np.array(menu['sensor']['times_s'])[sensor_frame_ids],a.rays,calibration['covariance_parameters'])
            sensor_model=CensoredRangeModel(noise,calibration['visibility_error_probability'],
                calibration['background_std_m'],calibration['outlier_probability'])
            scores=score_censored_actions(target_features,candidate_features,sensor_model,weights,fantasies=a.fantasies,seed=a.seed+index)
            variance_score=[float((x.residual_m.var(0)/np.diag(noise)+x.visible.var(0)).mean()) for x in candidate_features]
        else:
            candidate_features=np.stack([flatten_ranges(q,reference,query,sensor_frame_ids) for q in candidates])
            noise=np.kron(np.eye(count),discrepancy_covariance(a.rays,a.noise_mm/1000,a.noise_mm/1000))
            scores=score_actions(target_features,candidate_features,noise,weights,fantasies=a.fantasies,seed=a.seed+index)
            variance_score=candidate_features.var(1).mean(-1)
        choices={'predictive_risk':scores['selected_index'],
                 'random':int(np.random.default_rng(a.seed+index).integers(len(candidates))),
                 'fixed0':0,'fixed_last':len(candidates)-1,
                 'max_observation_variance':int(np.argmax(variance_score))}
        if a.random_expectation:
            for trial in range(len(candidates)):choices['random_trial'+str(trial)]=trial
        folder=a.output/name;folder.mkdir()
        digest=hashlib.sha256(particles['physical'].cpu().numpy().tobytes()).hexdigest()
        # Commit ALL choices before reading ANY candidate outcome.
        selection_paths={policy:record_selection(a.pool,a.output.name+'_'+policy,name,menu['candidates'][chosen]['action_id'],digest)
                         for policy,chosen in choices.items()}
        (folder/'selection-before-reveal.json').write_text(json.dumps({'choices':choices,'scores':scores,
            'paths':{key:str(value) for key,value in selection_paths.items()},'observed_frames':encoded['inputs']['cameras']['frame_ids'].tolist(),
            'acquired_sensor_frame_ids':sensor_frame_ids.tolist()},indent=2)+'\n')
        posterior_weights={'passive':weights};revealed=[]
        for policy,chosen in choices.items():
            acquired=load_selected_measurement(a.pool,selection_paths[policy]);measured=[]
            for frame in sensor_frame_ids:
                path=acquired['depth_paths'][int(frame)]
                with np.load(path) as raw:
                    measured.append(measure_censored_depth(raw['depth_m'],query) if calibration else measure_ranges(raw['depth_m'],query))
            if calibration:
                measurement=concatenate_observations(measured)
                posterior,diagnostics=sensor_model.condition(candidate_features[chosen],measurement,weights)
            else:
                measurement=np.concatenate(measured)
                posterior,diagnostics=condition(candidate_features[chosen],measurement,noise,weights)
            posterior_weights[policy]=posterior;revealed.append({'policy':policy,'action_id':acquired['action']['action_id'],**diagnostics})
        # Independent control response is not an available acquisition.
        with np.load(a.pool/'private'/name/'evaluation'/'trajectory.npz') as raw:
            if not np.allclose(raw['times_s'],menu['sensor']['times_s']):raise ValueError('Query time mismatch')
            truth=raw['positions_m']
        row={'episode_id':name,'revealed':revealed,'policies':{},
             'static_coordinate_rmse_m':float(np.sqrt(np.mean((reference['rest_points_m']-truth)**2))),
             'seconds_elapsed':time.monotonic()-started}
        for policy,posterior in posterior_weights.items():
            mean=np.tensordot(posterior,targets,axes=1)
            diagnostics=deformation_diagnostics(torch.tensor(mean),torch.tensor(reference['rest_points_m']),torch.tensor(reference['tetrahedra'],dtype=torch.long))
            row['policies'][policy]={'withheld_control_coordinate_rmse_m':float(np.sqrt(np.mean((mean-truth)**2))),
                'model_predictive_risk_m2':predictive_risk(target_features,posterior),
                'effective_sample_size':float(1/np.square(posterior).sum()),'mean_trajectory':diagnostics}
            row['policies'][policy]['physical_uncertainty_evaluation_only']=material_summary(particles,posterior,cases[name])
        if a.random_expectation:
            trials=[row['policies']['random_trial'+str(i)] for i in range(len(candidates))]
            row['policies']['random_expected']={key:float(np.mean([r[key] for r in trials]))
                for key in ['withheld_control_coordinate_rmse_m','effective_sample_size']}
        np.savez_compressed(folder/'target-predictions.npz',positions_m=targets,times_s=menu['sensor']['times_s'],
                            **{policy+'_weights':w for policy,w in posterior_weights.items()})
        (folder/'results.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row);print(json.dumps(row),flush=True)
    aggregate={policy:{key:float(np.mean([r['policies'][policy][key] for r in results]))
                       for key in ['withheld_control_coordinate_rmse_m','effective_sample_size']}
               for policy in results[0]['policies']}
    (a.output/'results.json').write_text(json.dumps({'episodes':results,'aggregate':aggregate,'protocol':protocol},indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
