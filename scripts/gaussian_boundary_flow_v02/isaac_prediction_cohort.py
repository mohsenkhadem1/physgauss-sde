"""Prediction-only five-frame Isaac cohort; NEVER opens private test targets.

Freeze after both training candidates finish. A separate evaluator must require
barrier.json, verify its hashes, and only then open simulator trajectories/labels.
Exposed historical test cohort: this is not a fresh confirmation experiment.
"""
import argparse
import json
import re
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from evaluate_gaussian_inverse import load_public_observations,predict_particles,load_observation_posterior
from gaussian_observed_features import extract_features
from train_gaussian_inverse import load_frozen_operator
from gaussian_operator import canonicalize_physical
from ridge_gaussian_inverse import RidgeInverse
from evaluate_ridge_gaussian_inverse import align_prefix_features
from scripts.gaussian_boundary_flow_v02.isaac_bridge_campaign import MODELS,validate_recipe
from scripts.gaussian_boundary_flow_v02.isaac_prefix_cache import sha,write_json

KINDS=('flow','mdn','chart','25d','ridge')
ARCHIVED={'chart':'runs/gaussian-inverse-final-chart-rgbd-v01/best.pt','25d':'runs/gaussian-inverse-final-25d-rgbd-v01/best.pt','ridge':'runs/gaussian-ridge-final-rgbd-prefix-v01/best.npz'}
SOURCE_FILES=[Path(__file__)]+[ROOT/'experiments'/p for p in ['evaluate_gaussian_inverse.py','gaussian_observed_features.py','gaussian_operator.py','train_gaussian_inverse.py','train_gaussian_chart_inverse.py','gaussian_chart_posterior.py','gaussian_modal_operator.py','ridge_gaussian_inverse.py','evaluate_ridge_gaussian_inverse.py','gaussian_observation.py','gaussian_video_encoder.py','gaussian_texture_features.py']]


def source_hashes():return {str(p.resolve().relative_to(ROOT)):sha(p) for p in SOURCE_FILES}

def roster_from_seal(path):
    """Only committed identity/group metadata; no raw plans or case recipes."""
    seal=json.loads(Path(path).read_text());cases=[]
    for dataset in seal['datasets']:
        root=Path(dataset['root']).resolve();name=root.name
        for row in dataset['cases']:
            episode=row['episode_id']
            if row['split']!='test' or not re.fullmatch('[A-Za-z0-9_-]+',episode):raise ValueError('Invalid exposed-test identity')
            case=f'{name}--{episode}'
            cases.append(dict(case=case,dataset=name,group=row['group_id'],episode_id=episode,episode=str(root/'episodes'/episode)))
    if len(cases)!=76 or len({r['case'] for r in cases})!=76:raise ValueError('Exact76 unique exposed cases required')
    if sorted(sum(r['dataset']==name for r in cases) for name in {r['dataset'] for r in cases})!=[12,64]:raise ValueError('Expected64+12 exposed cases')
    return cases


def freeze(training,output):
    training=Path(training).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists():raise FileExistsError('Prediction protocol is immutable')
    recipe=validate_recipe(training);models={}
    training_seal=training/'training-sealed.json';sealed=json.loads(training_seal.read_text())
    if sealed['recipe_sha256']!=sha(training/'recipe.json'):raise ValueError('Training seal recipe mismatch')
    for kind in ('flow','mdn'):
        if sealed['models'][kind]['completed_steps']!=2400:raise ValueError('Training seal incomplete')
        if any(sha(ROOT/path)!=digest for path,digest in sealed['models'][kind]['artifact_sha256'].items()):raise ValueError('Sealed training artifact changed')
    for kind in ('flow','mdn'):
        status=training/kind/'status.json';state=json.loads(status.read_text())
        if state['state']!='complete' or state['step']!=2400 or state['recipe_sha256']!=sha(training/'recipe.json'):
            raise ValueError('Both new training runs must finish before prediction freezing')
        checkpoint=training/kind/'best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['kind']!=kind or saved['recipe_sha256']!=sha(training/'recipe.json') or not 1<=saved['step']<=2400:
            raise ValueError('Best checkpoint binding mismatch')
        models[kind]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),teacher_sha256=recipe['teacher_sha256'],completed_status_sha256=sha(status),best_step=saved['step'],samples=128,supervision='new label-supervised posterior')
    for kind,relative in ARCHIVED.items():
        checkpoint=(ROOT/relative).resolve()
        models[kind]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),teacher_sha256=recipe['teacher_sha256'],samples=1 if kind=='ridge' else 128,supervision='archived different-training comparator')
    cases=roster_from_seal(recipe['exposed_test_seal'])
    protocol=dict(version='isaac-five-frame-exposed-prediction-v01',training=str(training),training_recipe_sha256=sha(training/'recipe.json'),
                  training_seal=str(training_seal),training_seal_sha256=sha(training_seal),source_sha256=source_hashes(),exposed_seal=recipe['exposed_test_seal'],exposed_seal_sha256=recipe['exposed_test_seal_sha256'],
                  teacher=recipe['teacher'],teacher_sha256=recipe['teacher_sha256'],models=models,cases=cases,
                  input=dict(frames=5,times_s=[0,.25,.5,.75,1.],feature_shape=[5,119],size=[192,160],modality='public RGB-D, supplied masks/cameras and initial reference'),
                  forecast=dict(times_s=(np.arange(61)/20).tolist(),steps=32,seed=61033,primary_window_s=[2.,3.],all_nodes=455),
                  barrier='All76 cases x five declared methods must be terminal with verified successful prediction seals or preserved failure records; independent evaluator only afterwards',
                  qualification=['historically exposed cases, not new confirmation','shared learned teacher, not certified physics','global25D code, not spatial Gaussian boundary field','all force families known; no arbitrary future forcing inputs','same observation budget, archived supervision differs'],
                  storage='Float32 full sample trajectories retained locally; probabilistic128 samples, deterministic ridge1. No scoring or targets in this process.')
    validate_budget(protocol,recipe)
    # CPU-only archived interface/teacher provenance admission, no case data.
    for kind in ARCHIVED:load_predictor(kind,models[kind],'cpu')
    write_json(path,protocol);return protocol


def validate_budget(protocol,recipe):
    if set(protocol['models'])!=set(KINDS):raise ValueError('Declared model roster changed')
    for kind,model in protocol['models'].items():
        if model['samples']!=(1 if kind=='ridge' else recipe['evaluation']['samples']):raise ValueError('Frozen sample budget changed')
        if model['teacher_sha256']!=recipe['teacher_sha256']:raise ValueError('Model teacher binding differs')
    forecast=protocol['forecast']
    if forecast['steps']!=recipe['evaluation']['flow_steps'] or forecast['steps']!=32 or forecast['seed']!=recipe['evaluation']['seed']:
        raise ValueError('Frozen inference step/seed budget changed')
    if forecast['times_s']!=(np.arange(61)/20).tolist() or forecast['all_nodes']!=455 or forecast['primary_window_s']!=[2.,3.]:raise ValueError('Frozen forecast clock/node budget changed')
    inputs=protocol['input']
    if inputs['frames']!=5 or inputs['times_s']!=[0,.25,.5,.75,1.] or inputs['feature_shape']!=[5,119] or inputs['size']!=[192,160]:raise ValueError('Frozen public observation budget changed')


def verify_protocol(output):
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text())
    if protocol['source_sha256']!=source_hashes():raise ValueError('Prediction sources changed')
    training=Path(protocol['training']);recipe=validate_recipe(training);validate_budget(protocol,recipe)
    if sha(training/'recipe.json')!=protocol['training_recipe_sha256']:raise ValueError('Training recipe changed')
    if sha(protocol['training_seal'])!=protocol['training_seal_sha256']:raise ValueError('Training seal changed')
    training_seal=json.loads(Path(protocol['training_seal']).read_text())
    if any(sha(ROOT/path)!=digest for model in training_seal['models'].values() for path,digest in model['artifact_sha256'].items()):raise ValueError('Sealed training artifact changed')
    if sha(protocol['exposed_seal'])!=protocol['exposed_seal_sha256'] or protocol['cases']!=roster_from_seal(protocol['exposed_seal']):raise ValueError('Exposed cohort binding changed')
    if sha(protocol['teacher'])!=protocol['teacher_sha256'] or protocol['teacher_sha256']!=recipe['teacher_sha256']:raise ValueError('Shared teacher changed')
    for kind,model in protocol['models'].items():
        if sha(model['checkpoint'])!=model['checkpoint_sha256']:raise ValueError('Checkpoint changed after prediction freeze')
        if kind in MODELS and sha(training/kind/'status.json')!=model['completed_status_sha256']:raise ValueError('Training completion status changed')
    if set(protocol['models'])!=set(KINDS):raise ValueError('Declared models changed')
    return protocol


def public_case_cache(row,case_dir,protocol_hash):
    """Only reference + five public RGB/mask/depth frames; no recipe access."""
    case_dir=Path(case_dir);case_dir.mkdir(parents=True,exist_ok=True);path=case_dir/'public-prefix.npz';seal_path=case_dir/'public-prefix-seal.json'
    if path.exists() or seal_path.exists():
        if not(path.exists() and seal_path.exists()):raise ValueError('Partial public cache; preserve and inspect')
        seal=json.loads(seal_path.read_text())
        if seal['case']!=row['case'] or seal['protocol_sha256']!=protocol_hash or seal['cache_sha256']!=sha(path):raise ValueError('Public cache binding changed')
        if any(sha(p)!=v for p,v in seal['public_input_sha256'].items()):raise ValueError('Public inputs changed')
        with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files},seal
    episode=Path(row['episode']);inputs=load_public_observations(episode,'rgbd',size=(192,160),max_time_s=1.)
    times=np.asarray(inputs['cameras']['times_s']);frames=np.asarray(inputs['cameras']['frame_ids'])
    if times.shape!=(5,) or not np.allclose(times,np.arange(5)*.25,atol=1e-7,rtol=0):raise ValueError('Five-frame prefix clock changed')
    features=extract_features(inputs)['features'].astype(np.float32);rest=inputs['reference']['rest_points_m'].astype(np.float32)
    if features.shape!=(5,119) or rest.shape!=(455,3) or not np.isfinite(features).all() or not np.isfinite(rest).all():raise ValueError('Invalid public feature/reference geometry')
    files=[episode/'reference.npz',episode/'observations/cameras.npz']
    for frame in frames:
        files.extend(episode/'observations'/folder/f'{frame:05d}.{suffix}' for folder,suffix in [('rgb','png'),('masks','png'),('depth','npz')])
    arrays=dict(features=features,rest=rest,times=times,frame_ids=frames)
    temporary=case_dir/'public-prefix.tmp.npz';np.savez_compressed(temporary,**arrays);temporary.replace(path)
    seal=dict(case=row['case'],protocol_sha256=protocol_hash,cache_sha256=sha(path),public_input_sha256={str(p.resolve()):sha(p) for p in files},no_private_files=True)
    write_json(seal_path,seal);return arrays,seal


def load_predictor(kind,descriptor,device):
    if kind in MODELS:
        saved=torch.load(descriptor['checkpoint'],map_location=device,weights_only=False);model=MODELS[kind]().to(device);model.load_state_dict(saved['model']);model.eval();return model
    if kind=='ridge':
        model,metadata=RidgeInverse.load(descriptor['checkpoint'])
        if metadata['operator_checkpoint_sha256']!=descriptor['teacher_sha256']:raise ValueError('Archived ridge teacher mismatch')
        if model.observation_mode!='rgbd' or model.observation_cutoff_s!=1. or model.texture_calibration is not None:raise ValueError('Ridge not compatible with strict RGB-D prefix')
        # Source and teacher bindings are independently checked by freeze/run.
        for name in ['ridge_gaussian_inverse.py','gaussian_observed_features.py']:
            if sha(ROOT/'experiments'/name)!=metadata['source_sha256'][name]:raise ValueError('Archived ridge inference source changed')
        return model
    model,checkpoint=load_observation_posterior(descriptor['checkpoint'],device)
    if checkpoint.get('operator_checkpoint_sha256')!=descriptor['teacher_sha256']:raise ValueError('Archived posterior teacher mismatch')
    if checkpoint.get('archived_feature_source_path') or checkpoint['model_kwargs'].get('feature_dim')!=119 or checkpoint['model_kwargs'].get('observation_mode')!='rgbd':raise ValueError('Archived feature contract differs; no silent reinterpretation')
    return model


def infer_particles(kind,model,arrays,descriptor,device,seed):
    generator=torch.Generator(device=device).manual_seed(seed)
    if kind=='ridge':
        aligned=align_prefix_features(model.feature_shape,model.observation_times_s,model.observation_cutoff_s,arrays['features'],arrays['times'])
        pred=model.predict_features(aligned);p=torch.tensor(pred['physical'],device=device);bc=torch.tensor(pred['boundary_type'],device=device);force=torch.tensor(pred['force_type'],device=device)
        return dict(physical=p,effective_physical=canonicalize_physical(p,bc,force),boundary_type=bc,force_type=force)
    features=torch.tensor(arrays['features'][None],device=device)
    with torch.no_grad():
        if kind in MODELS:draw=model.sample(features,samples=descriptor['samples'],steps=32,generator=generator)
        else:draw=model.infer(dict(features=features),samples=descriptor['samples'],generator=generator)
    return {key:draw[key][0] for key in ('physical','effective_physical','boundary_type','force_type')}


def verify_terminal(case_dir,kind,protocol,protocol_hash):
    case_dir=Path(case_dir);run=case_dir/kind;status_path=run/'status.json'
    if not status_path.exists():return None
    status=json.loads(status_path.read_text())
    if status.get('protocol_sha256')!=protocol_hash or status.get('model')!=kind or status.get('case')!=case_dir.name:raise ValueError('Terminal identity mismatch')
    if status['state']=='failed':
        if not status.get('error_type') or not status.get('error'):raise ValueError('Failure must remain explicit')
        return status
    if status['state']!='complete':raise ValueError('Unexpected terminal state')
    seal_path=run/'prediction-seal.json';seal=json.loads(seal_path.read_text())
    if status['seal_sha256']!=sha(seal_path) or seal['prediction_sha256']!=sha(run/'predictions.npz'):raise ValueError('Prediction seal changed')
    expected=dict(case=case_dir.name,model=kind,protocol_sha256=protocol_hash,checkpoint_sha256=protocol['models'][kind]['checkpoint_sha256'],teacher_sha256=protocol['teacher_sha256'],samples=protocol['models'][kind]['samples'])
    if any(seal.get(k)!=v for k,v in expected.items()):raise ValueError('Prediction identity binding differs')
    if seal['public_prefix_seal_sha256']!=sha(case_dir/'public-prefix-seal.json'):raise ValueError('Prediction input binding changed')
    public=json.loads((case_dir/'public-prefix-seal.json').read_text())
    if public['case']!=case_dir.name or public['protocol_sha256']!=protocol_hash or public['cache_sha256']!=sha(case_dir/'public-prefix.npz'):raise ValueError('Cached public input changed')
    return status


def run(output,device,start=0,stop=76):
    output=Path(output);torch.set_num_threads(2);protocol=verify_protocol(output);protocol_hash=sha(output/'protocol.json')
    if not (0<=start<stop<=76):raise ValueError('Invalid fixed-cohort shard')
    forward=load_frozen_operator(protocol['teacher'],device);models={};errors={}
    for kind,descriptor in protocol['models'].items():
        try:
            models[kind]=load_predictor(kind,descriptor,device)
        except Exception as exc:errors[kind]=(type(exc).__name__,str(exc))
    for index in range(start,stop):
        row=protocol['cases'][index];case_dir=output/row['case'];case_dir.mkdir(exist_ok=True)
        # Already terminal records are immutable, including failures. No retries/replacements.
        pending=[kind for kind in KINDS if verify_terminal(case_dir,kind,protocol,protocol_hash) is None]
        if not pending:continue
        try:arrays,_=public_case_cache(row,case_dir,protocol_hash);input_error=None
        except Exception as exc:input_error=(type(exc).__name__,str(exc))
        for kind in pending:
            destination=case_dir/kind;destination.mkdir(exist_ok=True)
            status=dict(case=row['case'],model=kind,protocol_sha256=protocol_hash)
            try:
                if any(destination.iterdir()):raise ValueError('Unfinished prediction artifacts preserved; no overwrite or silent retry')
                if input_error:raise RuntimeError('Public prefix failure: '+repr(input_error))
                if kind in errors:raise RuntimeError('Model load failure: '+repr(errors[kind]))
                descriptor=protocol['models'][kind]
                particles=infer_particles(kind,models[kind],arrays,descriptor,device,protocol['forecast']['seed']+index)
                if any(not torch.isfinite(v).all() for v in particles.values()):raise ValueError('Nonfinite particle output')
                positions=predict_particles(dict(device=device,forward=forward),particles,dict(rest_points_m=arrays['rest']),protocol['forecast']['times_s'])
                if positions.shape!=(descriptor['samples'],61,455,3) or not torch.isfinite(positions).all():raise ValueError('Nonfinite or mismatched forecast')
                prediction=dict(positions_m=positions.cpu().numpy(),mean_positions_m=positions.mean(0).cpu().numpy(),times_s=np.asarray(protocol['forecast']['times_s'],np.float32),rest_points_m=arrays['rest'],input_times_s=arrays['times'],input_frame_ids=arrays['frame_ids'])
                prediction.update({key:value.cpu().numpy() for key,value in particles.items()})
                temporary=destination/'predictions.tmp.npz';np.savez_compressed(temporary,**prediction);temporary.replace(destination/'predictions.npz')
                seal=dict(case=row['case'],model=kind,protocol_sha256=protocol_hash,checkpoint_sha256=descriptor['checkpoint_sha256'],teacher_sha256=protocol['teacher_sha256'],samples=descriptor['samples'],
                          prediction_sha256=sha(destination/'predictions.npz'),public_prefix_seal_sha256=sha(case_dir/'public-prefix-seal.json'),shape=list(positions.shape),seed=protocol['forecast']['seed']+index,source_binding='protocol source_sha256 + bound training recipe sources',no_target_access=True)
                write_json(destination/'prediction-seal.json',seal);status.update(state='complete',seal_sha256=sha(destination/'prediction-seal.json'))
            except Exception as exc:status.update(state='failed',error_type=type(exc).__name__,error=str(exc))
            write_json(destination/'status.json',status)
        write_json(output/f'progress-{start:02d}-{stop:02d}.json',dict(last_index=index,case=row['case'],protocol_sha256=protocol_hash))


def barrier(output):
    output=Path(output);protocol=verify_protocol(output);protocol_hash=sha(output/'protocol.json');records=[];missing=[]
    for row in protocol['cases']:
        for kind in KINDS:
            case_dir=output/row['case'];status=verify_terminal(case_dir,kind,protocol,protocol_hash)
            if status is None:missing.append([row['case'],kind]);continue
            records.append(dict(case=row['case'],model=kind,state=status['state'],status_sha256=sha(case_dir/kind/'status.json')))
    if missing:raise ValueError(f'Prediction barrier incomplete: {len(missing)}/380 missing')
    result=dict(protocol_sha256=protocol_hash,expected_cases=76,expected_predictions=380,terminal=len(records),complete=sum(r['state']=='complete' for r in records),failed=sum(r['state']=='failed' for r in records),records=records,targets_may_be_opened_by_independent_evaluator_only=True)
    path=output/'barrier.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('Existing barrier changed')
    else:write_json(path,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','barrier']);p.add_argument('--training',default='runs/gaussian-boundary-isaac-bridge-v01');p.add_argument('--output',default='runs/gaussian-boundary-isaac-exposed-v01');p.add_argument('--device',default='cpu');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=76);a=p.parse_args()
    if a.command=='freeze':freeze(a.training,a.output)
    elif a.command=='run':run(a.output,a.device,a.start,a.stop)
    else:print(json.dumps(barrier(a.output)))
