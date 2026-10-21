"""Supervised deterministic ridge inverse baseline on public RGB/RGB-D features.

TRAIN alone fits feature/label normalization and linear coefficients. Validation
trajectory RMSE through the frozen neural forward operator selects ridge lambda.
Runtime inference accepts only public features and needs no physical label,
trajectory, FEM, per-scene optimization or simulation. This is a point baseline:
linear class scores and regression residuals are not calibrated uncertainty.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls,training_array):
        array=np.asarray(training_array,dtype=np.float64)
        if array.ndim!=2 or len(array)<2 or not np.isfinite(array).all():
            raise ValueError('At least two finite TRAIN rows required')
        mean=array.mean(0);scale=array.std(0)
        return cls(mean,np.where(scale>1e-8,scale,1.))

    def transform(self,array):return (np.asarray(array,dtype=np.float64)-self.mean)/self.scale
    def inverse(self,array):return np.asarray(array)*self.scale+self.mean


@dataclass
class RidgeInverse:
    feature_standardizer: Standardizer
    label_standardizer: Standardizer
    coefficients: np.ndarray
    regularization: float
    feature_shape: tuple
    observation_mode: str
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    texture_calibration: dict | None = None
    observation_cutoff_s: float | None = None
    observation_times_s: np.ndarray | None = None

    @classmethod
    def fit(cls,training_features,training_labels,regularization,observation_mode,
            physical_lower,physical_upper,observation_cutoff_s=None,observation_times_s=None):
        features=np.asarray(training_features,dtype=np.float64)
        labels=np.asarray(training_labels,dtype=np.float64)
        if features.ndim<2 or labels.ndim!=2 or len(features)!=len(labels) or labels.shape[-1]!=33:
            raise ValueError('Expected matching feature rows and 25+4+4 supervised labels')
        if regularization<=0 or observation_mode not in ('rgb','rgbd'):
            raise ValueError('Positive ridge penalty and explicit observation mode required')
        shape=features.shape[1:]
        if observation_cutoff_s is not None:
            times=np.asarray(observation_times_s,dtype=float)
            if features.ndim!=3 or times.shape!=(shape[0],) or not np.isfinite(times).all():raise ValueError('Prefix model needs its finite observation clock')
            keep=times<=observation_cutoff_s+1e-8
            if keep.sum()<2:raise ValueError('At least two prefix observations required')
            features=features.copy();features[:,~keep,:]=0
        flat=features.reshape(len(features),-1)
        x_normalizer=Standardizer.fit(flat);y_normalizer=Standardizer.fit(labels)
        x=x_normalizer.transform(flat);y=y_normalizer.transform(labels)
        # Objective mean squared error + lambda * squared coefficient norm.
        # Dual solve costs O(TRAIN rows^3), avoiding a feature_dim^3 solve.
        dual=np.linalg.solve(x@x.T+len(x)*regularization*np.eye(len(x)),y)
        coefficients=x.T@dual
        return cls(x_normalizer,y_normalizer,coefficients,float(regularization),shape,
                   observation_mode,np.asarray(physical_lower),np.asarray(physical_upper),
                   observation_cutoff_s=observation_cutoff_s,
                   observation_times_s=np.asarray(observation_times_s,dtype=float) if observation_cutoff_s is not None else None)

    def predict_features(self,features):
        """Public-array inference. Features contain no labels or episode IDs."""
        values=np.asarray(features,dtype=np.float64)
        if values.shape==self.feature_shape:values=values[None]
        if values.shape[1:]!=self.feature_shape:
            raise ValueError('Observation feature shape/schedule differs from training')
        if self.observation_cutoff_s is not None:
            values=values.copy();values[:,self.observation_times_s>self.observation_cutoff_s+1e-8,:]=0
        if not np.isfinite(values).all():raise ValueError('Nonfinite available observation features')
        raw=self.label_standardizer.inverse(self.feature_standardizer.transform(values.reshape(len(values),-1))@self.coefficients)
        width=self.physical_upper-self.physical_lower
        physical=np.clip(raw[:,:25],self.physical_lower+1e-5*width,self.physical_upper-1e-5*width)
        return {'physical':physical.astype(np.float32),'boundary_type':raw[:,25:29].argmax(-1),
                'force_type':raw[:,29:33].argmax(-1),'boundary_scores':raw[:,25:29],
                'force_scores':raw[:,29:33],
                'physical_clipped_fraction':np.mean(physical!=raw[:,:25],axis=-1)}

    def save(self,path,metadata=None):
        metadata=dict(metadata or {})
        if self.texture_calibration is not None:metadata.setdefault('texture_calibration',self.texture_calibration)
        if self.observation_cutoff_s is not None:
            metadata.setdefault('observation_cutoff_s',self.observation_cutoff_s)
            metadata.setdefault('observation_times_s',self.observation_times_s.tolist())
        np.savez_compressed(path,feature_mean=self.feature_standardizer.mean,
                            feature_scale=self.feature_standardizer.scale,
                            label_mean=self.label_standardizer.mean,label_scale=self.label_standardizer.scale,
                            coefficients=self.coefficients,regularization=self.regularization,
                            feature_shape=np.asarray(self.feature_shape),observation_mode=np.asarray(self.observation_mode),
                            physical_lower=self.physical_lower,physical_upper=self.physical_upper,
                            metadata_json=np.asarray(json.dumps(metadata or {},sort_keys=True)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as raw:
            model=cls(Standardizer(raw['feature_mean'],raw['feature_scale']),
                      Standardizer(raw['label_mean'],raw['label_scale']),raw['coefficients'],
                      float(raw['regularization']),tuple(raw['feature_shape'].tolist()),
                      str(raw['observation_mode']),raw['physical_lower'],raw['physical_upper'])
            metadata=json.loads(str(raw['metadata_json']))
        model.texture_calibration=metadata.get('texture_calibration')
        model.observation_cutoff_s=metadata.get('observation_cutoff_s')
        model.observation_times_s=np.asarray(metadata['observation_times_s'],dtype=float) if model.observation_cutoff_s is not None else None
        return model,metadata


def labels_from_rows(rows):
    physical=np.stack([r['physical'] for r in rows])
    bc=np.eye(4)[[r['bc'] for r in rows]];force=np.eye(4)[[r['force'] for r in rows]]
    return np.concatenate((physical,bc,force),-1)


def append_texture_features(features,observed_frame_ids,texture):
    """Match the inverse model's public texture backend on the same time grid."""
    if not np.array_equal(texture['frame_ids'],observed_frame_ids):
        raise ValueError('Texture/geometry observation schedules differ')
    values=np.asarray(texture['features'])
    if values.ndim!=2 or len(values)!=len(features) or not np.isfinite(values).all():
        raise ValueError('Finite public texture features at each observed time required')
    return np.concatenate((features,values),axis=-1)


def infer_episode(model,episode,size=(192,160)):
    """Read public observations only; never call supervised training loaders."""
    try:
        from gaussian_observed_features import load_observed_arrays,extract_features
    except ImportError:
        from experiments.gaussian_observed_features import load_observed_arrays,extract_features
    inputs=load_observed_arrays(episode,model.observation_mode,size)
    if model.observation_cutoff_s is not None:
        np.testing.assert_allclose(inputs['cameras']['times_s'],model.observation_times_s,atol=1e-7,rtol=0)
    features=extract_features(inputs)['features']
    if model.texture_calibration is not None:
        if model.observation_mode!='rgb':raise ValueError('Texture augmentation is declared RGB-only')
        try:from gaussian_texture_features import load_texture_features
        except ImportError:from experiments.gaussian_texture_features import load_texture_features
        features=append_texture_features(features,inputs['cameras']['frame_ids'],
                                        load_texture_features(episode,model.texture_calibration))
    return model.predict_features(features),inputs


def predict_forward(operator,prediction,rest,times,device='cpu'):
    import torch
    try:from gaussian_operator import canonicalize_physical
    except ImportError:from experiments.gaussian_operator import canonicalize_physical
    physical=torch.as_tensor(prediction['physical'],device=device,dtype=torch.float32)
    bc=torch.as_tensor(prediction['boundary_type'],device=device,dtype=torch.long)
    force=torch.as_tensor(prediction['force_type'],device=device,dtype=torch.long)
    physical=canonicalize_physical(physical,bc,force)
    with torch.no_grad():
        positions=operator(torch.as_tensor(rest,device=device,dtype=torch.float32),
                           torch.as_tensor(times,device=device,dtype=torch.float32),physical,bc,force)
    return positions.detach().cpu().numpy()


def validation_scores(model,rows,operator,device='cpu',target_window=None):
    """VAL-only selection on the exact six times/64 nodes used by flow trainer."""
    output=[]
    for row in rows:
        prediction=model.predict_features(row['features'])
        indices=np.linspace(0,len(row['rest'])-1,64).astype(int)
        selected=np.ones(len(row['target_times']),dtype=bool) if target_window is None else (row['target_times']>=target_window[0])&(row['target_times']<=target_window[1])
        if not selected.any():raise ValueError('Empty declared validation selection window')
        positions=predict_forward(operator,prediction,row['rest'][indices],row['target_times'][selected],device)[0]
        target=row['target_positions'][selected][:,indices]
        output.append({'key':row['key'],'trajectory_rmse_m':float(np.sqrt(np.mean((positions-target)**2))),
                       'boundary_correct':bool(prediction['boundary_type'][0]==row['bc']),
                       'force_correct':bool(prediction['force_type'][0]==row['force']),
                       'physical_clipped_fraction':float(prediction['physical_clipped_fraction'][0])})
    return {'trajectory_rmse_m':float(np.mean([r['trajectory_rmse_m'] for r in output])),
            'boundary_accuracy':float(np.mean([r['boundary_correct'] for r in output])),
            'force_accuracy':float(np.mean([r['force_correct'] for r in output])),
            'cases':output}


def train(catalog,operator_checkpoint,output,*,observation_mode='rgbd',
          penalties=(.0001,.001,.01,.1,1.,10.,100.,1000.),device='cpu',size=(192,160),threads=4,
          texture_calibration=None,observation_cutoff_s=None):
    import torch
    try:
        from train_gaussian_inverse import collect_cases,apply_catalog,load_cases,load_frozen_operator
        from gaussian_operator import PhysicalParameterTransform
    except ImportError:
        from experiments.train_gaussian_inverse import collect_cases,apply_catalog,load_cases,load_frozen_operator
        from experiments.gaussian_operator import PhysicalParameterTransform
    torch.set_num_threads(threads)
    texture_path=Path(texture_calibration) if texture_calibration else None
    texture=json.loads(texture_path.read_text()) if texture_path else None
    if texture is not None and observation_mode!='rgb':raise ValueError('Texture augmentation is declared RGB-only')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    frozen=json.loads(Path(catalog).read_text())
    if any(c['split'] not in ('train','validation') for c in frozen['cases']):
        raise ValueError('TEST cases forbidden in ridge training/selection catalog')
    cases=apply_catalog(collect_cases(frozen['roots']),catalog)
    train_cases=[c for c in cases if c['recipe']['split']=='train']
    validation_cases=[c for c in cases if c['recipe']['split']=='validation']
    if not train_cases or not validation_cases:raise ValueError('Both TRAIN and VAL required')
    source_names=['ridge_gaussian_inverse.py','train_gaussian_inverse.py','gaussian_observed_features.py',
                  'gaussian_operator.py','gaussian_observation.py','gaussian_video_encoder.py',
                  'gaussian_modal_operator.py','gaussian_texture_features.py']
    source_bytes={name:Path(__file__).with_name(name).read_bytes() for name in source_names}
    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    (root/'provenance').mkdir()
    for name,data in source_bytes.items():(root/'provenance'/name).write_bytes(data)
    protocol={'baseline':'supervised deterministic ridge physical inverse','observation_mode':observation_mode,
              'training_count':len(train_cases),'validation_count':len(validation_cases),'no_test_cases':True,
              'regularization_grid':list(penalties),'normalization':'TRAIN rows only, features and33labels',
              'labels':'25 continuous physical coefficients plus boundary/force onehot',
              'category_prediction':'argmax unconstrained linear class score; not calibrated probabilities',
              'checkpoint_selection':'minimum VAL mean coordinate RMSE, six times64nodes, frozen forward operator',
              'catalog_sha256':hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
              'operator_checkpoint_sha256':hashlib.sha256(Path(operator_checkpoint).read_bytes()).hexdigest(),
              'source_sha256':source_hashes,'device':device,'size':list(size),'threads':threads,
              'torch_disable_native_jit':os.environ.get('TORCH_DISABLE_NATIVE_JIT'),
              'torch_version':str(torch.__version__),
              'feature_version':'rgb_geometry_texture_v1' if texture else 'geometry_v1',
              'texture_calibration':texture,
              'texture_calibration_sha256':hashlib.sha256(texture_path.read_bytes()).hexdigest() if texture_path else None,
              'observation_cutoff_s':observation_cutoff_s,
              'selection_target_window_s':[2.,3.] if observation_cutoff_s is not None else None,
              'prefix_rule':'Future columns zero BEFORE TRAIN normalization/fit and again at every prediction' if observation_cutoff_s is not None else None,
              'uncertainty':'not implemented; no calibration or Bayesian posterior claim'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Only validation loads trajectory labels, for model selection/evaluation.
    if texture_path:(root/'provenance/texture-calibration.json').write_bytes(texture_path.read_bytes())
    training=load_cases(train_cases,size,input_mode='features',observation_mode=observation_mode,texture_calibration=texture)
    validation=load_cases(validation_cases,size,operator_targets=True,input_mode='features',observation_mode=observation_mode,texture_calibration=texture)
    x=np.stack([r['features'] for r in training]);y=labels_from_rows(training)
    observed_times=training[0]['times']
    for row in training+validation:np.testing.assert_allclose(row['times'],observed_times,atol=1e-7,rtol=0)
    protocol['observation_times_s']=observed_times.tolist()
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    limits=PhysicalParameterTransform();lower=limits.lower.numpy();upper=limits.upper.numpy()
    operator=load_frozen_operator(operator_checkpoint,device)
    scores=[];best_score=float('inf');best=None
    for penalty in penalties:
        candidate=RidgeInverse.fit(x,y,penalty,observation_mode,lower,upper,
                                   observation_cutoff_s=observation_cutoff_s,observation_times_s=observed_times)
        candidate.texture_calibration=texture
        score=validation_scores(candidate,validation,operator,device,
                                target_window=(2.,3.) if observation_cutoff_s is not None else None)
        scores.append({'regularization':float(penalty),**score})
        print(json.dumps({k:v for k,v in scores[-1].items() if k!='cases'}),flush=True)
        (root/'selection.json').write_text(json.dumps(scores,indent=2)+'\n')
        if score['trajectory_rmse_m']<best_score:
            best_score=score['trajectory_rmse_m'];best=candidate
    if best is None:raise RuntimeError('No finite ridge candidate')
    metadata={**protocol,'selected_regularization':best.regularization,'feature_names':training[0]['feature_names'],
              'catalog_path':str(Path(catalog).resolve()),'operator_checkpoint':str(Path(operator_checkpoint).resolve()),
              'training_keys':[r['key'] for r in training],'validation_keys':[r['key'] for r in validation]}
    best.save(root/'best.npz',metadata)
    # All parameter fitting/selection is complete before this dense VAL report.
    dense=[]
    for row in validation:
        prediction=best.predict_features(row['features'])
        with np.load(row['path']/'truth/cameras.npz') as raw:cameras={k:raw[k] for k in raw.files}
        positions=predict_forward(operator,prediction,row['rest'],cameras['times_s'],device)[0]
        filename=row['key'].replace('/','__')+'-prediction.npz'
        np.savez_compressed(root/filename,positions_m=positions,times_s=cameras['times_s'],
                            frame_ids=cameras['frame_ids'],physical=prediction['physical'],
                            boundary_type=prediction['boundary_type'],force_type=prediction['force_type'])
        with np.load(row['path']/'truth/trajectory.npz') as raw:target=raw['positions_m']
        gap=(cameras['times_s']>1)&(cameras['times_s']<2)
        forecast=(cameras['times_s']>=2)&(cameras['times_s']<=3)
        dense.append({'key':row['key'],'coordinate_rmse_m':float(np.sqrt(np.mean((positions-target)**2))),
                      'static_coordinate_rmse_m':float(np.sqrt(np.mean((row['rest'][None]-target)**2))),
                      'gap_coordinate_rmse_m':float(np.sqrt(np.mean((positions[gap]-target[gap])**2))),
                      'forecast_coordinate_rmse_m':float(np.sqrt(np.mean((positions[forecast]-target[forecast])**2))),
                      'boundary_correct':bool(prediction['boundary_type'][0]==row['bc']),
                      'force_correct':bool(prediction['force_type'][0]==row['force'])})
    aggregate={k:float(np.mean([r[k] for r in dense])) for k in ['coordinate_rmse_m','static_coordinate_rmse_m','gap_coordinate_rmse_m','forecast_coordinate_rmse_m','boundary_correct','force_correct']}
    result={'selected_regularization':best.regularization,'selection_trajectory_rmse_m':best_score,
            'validation_dense':dense,'aggregate':aggregate,'seconds_elapsed':time.monotonic()-started,
            'qualification':'TRAIN/VAL development baseline; no TEST opened; deterministic mean; no calibrated uncertainty',
            'protocol':protocol}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',required=True);parser.add_argument('--operator-checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--observation-mode',choices=['rgb','rgbd'],default='rgbd')
    parser.add_argument('--penalties',type=float,nargs='+',default=[.0001,.001,.01,.1,1.,10.,100.,1000.])
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--texture-calibration',help='Same frozen TRAIN-only RGB texture calibration as matched inverse model')
    parser.add_argument('--observation-cutoff-s',type=float,help='Train an independent prefix-only baseline; future feature columns always zero')
    args=parser.parse_args();result=train(args.catalog,args.operator_checkpoint,args.output,observation_mode=args.observation_mode,
                                         penalties=args.penalties,device=args.device,threads=args.threads,
                                         texture_calibration=args.texture_calibration,observation_cutoff_s=args.observation_cutoff_s)
    print(json.dumps({k:v for k,v in result.items() if k not in ('validation_dense','protocol')},indent=2))


if __name__=='__main__':main()
