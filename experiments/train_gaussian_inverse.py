"""Train amortized physical hypotheses from sparse RGB, masks and known cameras.

Training/evaluation labels are supervised simulator data. Runtime inference is
one image encoder followed by an invertible conditional posterior; no FEM,
trajectory truth, per-video fitting, or observation depth enters that API.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import ast
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gaussian_observation import load_rgb_inputs
from gaussian_video_encoder import GaussianVideoEncoder
from gaussian_operator import (PhysicalPosterior,physical_vector,category_labels,
                               PhysicalParameterTransform,NeuralGreenOperator,NeuralTrajectoryOperator)

LABEL_TOLERANCE=np.r_[[.015,.01,.0015,.02],np.full(12,.0003),np.full(6,.001),np.full(3,.00005)].astype(np.float32)
SOURCE_FILES=('train_gaussian_inverse.py','gaussian_observation.py','gaussian_video_encoder.py','gaussian_operator.py','gaussian_observed_features.py')


def source_hashes():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def collect_cases(roots):
    """Catalog only completed TRAIN/VAL cases; never open any test private file."""
    cases=[];seen=set()
    for root in map(Path,roots):
        plan=json.loads((root/'plan.json').read_text())
        for recipe in plan['episodes']:
            if recipe['split'] not in ('train','validation'):continue
            episode=root/'episodes'/recipe['episode_id']
            summary=episode/'summary.json'
            if not summary.exists() or (episode/'failure.json').exists():continue
            if not json.loads(summary.read_text()).get('completed'):continue
            audit=episode/'validation.json'
            if not audit.exists() or not json.loads(audit.read_text()).get('passed'):continue
            key=(str(root.resolve()),recipe['episode_id'])
            if key in seen:continue
            seen.add(key)
            cases.append({'path':episode,'recipe':recipe,'dataset':plan['dataset'],
                          'key':plan['dataset']+'/'+recipe['episode_id'],
                          'plan_sha256':plan.get('plan_sha256')})
    return cases


def snapshot_catalog(roots,output):
    cases=collect_cases(roots)
    rows=[{'key':r['key'],'split':r['recipe']['split'],'group_id':r['recipe']['group_id'],
           'recipe_sha256':r['recipe']['recipe_sha256'],'path':str(r['path'].resolve())} for r in cases]
    payload={'roots':[str(Path(r).resolve()) for r in roots],'cases':rows,'no_test_cases':True}
    payload['catalog_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Catalog snapshots are immutable; choose a fresh path')
    output.write_text(json.dumps(payload,indent=2)+'\n');return payload


def apply_catalog(cases,path):
    frozen=json.loads(Path(path).read_text());digest=frozen.pop('catalog_sha256')
    if hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest:
        raise ValueError('Catalog fingerprint mismatch')
    selected={r['key']:r for r in frozen['cases']}
    if any(r['split'] not in ('train','validation') for r in selected.values()):raise ValueError('Test cases forbidden in training catalog')
    available={r['key']:r for r in cases}
    if not selected.keys()<=available.keys():raise ValueError('Frozen catalog case is missing or incomplete')
    output=[]
    for key,wanted in selected.items():
        row=available[key]
        if row['recipe']['recipe_sha256']!=wanted['recipe_sha256'] or row['recipe']['split']!=wanted['split']:
            raise ValueError('Frozen catalog recipe or split changed')
        output.append(row)
    return output


def add_label_tolerance(physical,noise,transform):
    std=physical.new_tensor(LABEL_TOLERANCE)
    span=transform.upper-transform.lower
    return torch.maximum(torch.minimum(physical+noise*std,transform.upper-1e-5*span),transform.lower+1e-5*span)


class GaussianInverseModel(nn.Module):
    def __init__(self,context_dim=128,hidden=128,layers=8,input_mode="cnn",feature_dim=None,context_mode="features",observation_mode="rgb"):
        super().__init__();self.input_mode=input_mode;self.observation_mode=observation_mode
        if input_mode=="features":
            from gaussian_observed_features import ObservedFeatureEncoder
            self.encoder=ObservedFeatureEncoder(feature_dim,context_dim,hidden,mode=context_mode)
        else:self.encoder=GaussianVideoEncoder(context_dim)
        self.posterior=PhysicalPosterior(context_dim,hidden,layers)

    def context(self,batch,observation_mask=None):
        if self.input_mode=='features':return self.encoder(batch['features'],observation_mask)
        return self.encoder(batch['images'],batch['poses'],batch['K'],batch['times'],
                            batch['rest'],observation_mask,calibration_size=batch['calibration_size'])

    def infer(self,batch,samples=32,observation_mask=None,generator=None):
        """Public-input-only inference; no labels or file access are required."""
        return self.posterior.sample(self.context(batch,observation_mask),samples=samples,generator=generator)


def load_cases(cases,size=(192,160),operator_targets=False,input_mode="cnn",observation_mode="rgb"):
    loaded=[]
    for item in cases:
        recipe=item['recipe'];episode=item['path']
        if input_mode=='features':
            from gaussian_observed_features import load_observed_arrays,extract_features
            inputs=load_observed_arrays(episode,observation_mode,size)
            measured=extract_features(inputs)
        else:inputs=load_rgb_inputs(episode,size,roi=True)
        spec=json.loads((episode/'truth/boundaries.json').read_text())
        p=physical_vector(recipe,spec,inputs['reference']['rest_points_m']);bc,force=category_labels(recipe)
        images=np.concatenate((inputs['rgb'],inputs['mask'][...,None]),-1).transpose(0,3,1,2)
        c=inputs['cameras']
        row={**item,'images':images,'poses':c['camera_to_world_usd_row'].astype(np.float32),
             'K':c['K'].astype(np.float32),'times':c['times_s'].astype(np.float32),
             'rest':inputs['reference']['rest_points_m'].astype(np.float32),
             'physical':p,'bc':bc,'force':force,'calibration_size':tuple(size)}
        if input_mode=='features':
            row['features']=measured['features'];row['feature_names']=measured['feature_names']
            row['feature_audits']=measured['audits']
            row['images']=np.zeros((len(images),1,1,1),dtype=np.float32)
        if operator_targets:
            with np.load(episode/'truth/trajectory.npz') as f:
                ids=np.array([10,20,30,40,50,60]);row['target_times']=f['times_s'][ids].astype(np.float32)
                row['target_positions']=f['positions_m'][ids].astype(np.float32)
        loaded.append(row)
    return loaded


def batch_rows(rows,device):
    result={key:torch.as_tensor(np.stack([r[key] for r in rows]),device=device) for key in ['images','poses','K','times','rest','physical','bc','force']}
    if all('features' in r for r in rows):result['features']=torch.as_tensor(np.stack([r['features'] for r in rows]),device=device)
    result['bc']=result['bc'].long();result['force']=result['force'].long();result['calibration_size']=rows[0]['calibration_size']
    if all('target_positions' in r for r in rows):
        for key in ['target_times','target_positions']:result[key]=torch.as_tensor(np.stack([r[key] for r in rows]),device=device)
    return result


def predictive_energy_score(predictions,target,scale=.01):
    """Unbiased sample energy score using distinct sample pairs.

    The norm is Euclidean per trajectory divided bysqrt(coordinates), then by
    the declared lengthscale. Finite-sample estimates may be negative. Unlike
    single-draw MSE, its expectation rewards a predictive distribution.
    """
    count=predictions.shape[1]
    if count<2:raise ValueError('Energy score needs at least two samples')
    difference=(predictions-target[:,None]).flatten(2)/scale
    data_norm=difference.square().mean(-1).clamp_min(1e-12).sqrt().mean(1)
    pair=(predictions[:,:,None]-predictions[:,None,:]).flatten(3)/scale
    pair_norm=pair.square().mean(-1).clamp_min(1e-12).sqrt()
    off_diagonal=1-torch.eye(count,device=predictions.device,dtype=predictions.dtype)
    pair_mean=(pair_norm*off_diagonal).sum((1,2))/(count*(count-1))
    return (data_norm-.5*pair_mean).mean()


def forward_draws(operator,posterior,context,batch,samples=4,known_categories=False,point_ids=None,time_ids=None,generator=None):
    arguments={'boundary_type':batch['bc'],'force_type':batch['force']} if known_categories else {}
    draw=posterior.sample(context,samples=samples,generator=generator,**arguments)
    rest=batch['rest'];times=batch['target_times'];target=batch['target_positions']
    if point_ids is not None:rest=rest[:,point_ids];target=target[:,:,point_ids]
    if time_ids is not None:times=times[:,time_ids];target=target[:,time_ids]
    b=len(rest)
    prediction=operator(rest.repeat_interleave(samples,0),times.repeat_interleave(samples,0),
                        draw['effective_physical'].reshape(b*samples,-1),draw['boundary_type'].reshape(-1),draw['force_type'].reshape(-1))
    prediction=prediction.reshape(b,samples,len(times[0]),len(rest[0]),3)
    return prediction,target


def validation_scores(model,rows,device,noise_seed=93211,batch_size=8,reference_only=False,operator=None,predictive_samples=32):
    model.eval();scores=[];generator=torch.Generator(device=device).manual_seed(noise_seed)
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            batch=batch_rows(rows[start:start+batch_size],device)
            mask=None
            if reference_only:
                mask=torch.zeros(batch['images'].shape[:2],device=device,dtype=torch.bool);mask[:,0]=True
            context=model.context(batch,mask)
            # Fixed independent dequantization draws make checkpoint selection repeatable.
            nll=[]
            for _ in range(4):
                noise=torch.randn(batch['physical'].shape,device=device,generator=generator)
                labels=add_label_tolerance(batch['physical'],noise,model.posterior.transform)
                nll.append(-model.posterior.log_prob(labels,context,batch['bc'],batch['force']))
            bc,force=model.posterior.categorical_logits(context)
            predictive=None
            if operator is not None and 'target_positions' in batch:
                points=torch.linspace(0,batch['rest'].shape[1]-1,64,device=device).long()
                predicted,target=forward_draws(operator,model.posterior,context,batch,predictive_samples,point_ids=points,generator=generator)
                predictive=((predicted.mean(1)-target).square().mean((1,2,3))).sqrt()
            for index,row in enumerate(rows[start:start+batch_size]):
                scores.append({'key':row['key'],'nll':float(torch.stack(nll)[:,index].mean()),
                               'boundary_correct':bool(bc[index].argmax()==batch['bc'][index]),
                               'force_correct':bool(force[index].argmax()==batch['force'][index]),
                               'trajectory_rmse_m':float(predictive[index]) if predictive is not None else None})
    model.train()
    result={'nll':float(np.mean([r['nll'] for r in scores])),
            'boundary_accuracy':float(np.mean([r['boundary_correct'] for r in scores])),
            'force_accuracy':float(np.mean([r['force_correct'] for r in scores])),'cases':scores}
    if operator is not None:result['trajectory_rmse_m']=float(np.mean([r['trajectory_rmse_m'] for r in scores]))
    return result


def load_inverse_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint['source_sha256']!=source_hashes():
        # Training-only catalog changes need not invalidate identical inference
        # code. Require exact archived sources and identical model-class AST;
        # every imported inference dependency must still match its fingerprint.
        provenance=Path(path).parent/'provenance'
        for name,digest in checkpoint['source_sha256'].items():
            archived=provenance/name
            if not archived.exists() or hashlib.sha256(archived.read_bytes()).hexdigest()!=digest:
                raise ValueError('Inverse checkpoint has missing or mismatched archived provenance')
            if name!='train_gaussian_inverse.py' and source_hashes().get(name)!=digest:
                if name!='gaussian_observed_features.py':raise ValueError('Inverse inference dependency changed')
                def encoder_ast(text):
                    node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ObservedFeatureEncoder')
                    return ast.dump(node,include_attributes=False)
                if encoder_ast(archived.read_text())!=encoder_ast(Path(__file__).with_name(name).read_text()):
                    raise ValueError('Feature encoder architecture changed')
                checkpoint['archived_feature_source_path']=str(archived.resolve())
        def model_ast(text):
            node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='GaussianInverseModel')
            return ast.dump(node,include_attributes=False)
        if model_ast((provenance/'train_gaussian_inverse.py').read_text())!=model_ast(Path(__file__).read_text()):
            raise ValueError('Inverse model class differs from archived inference code')
        checkpoint['inference_provenance_verified']=True
    model=GaussianInverseModel(**checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    return model,checkpoint


def load_frozen_operator(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    classes={'NeuralGreenOperator':NeuralGreenOperator,'NeuralTrajectoryOperator':NeuralTrajectoryOperator}
    name=checkpoint.get('model_class')
    if name not in classes:raise ValueError('Operator checkpoint needs model_class and model_kwargs')
    kwargs=dict(checkpoint['model_kwargs'])
    if 'geometry_reference' in checkpoint['model_state']:
        kwargs['reference_points']=checkpoint['model_state']['geometry_reference'].detach().cpu().numpy()
    frozen_source=checkpoint.get('source_sha256',{})
    if isinstance(frozen_source,dict) and 'gaussian_operator.py' in frozen_source:
        if frozen_source['gaussian_operator.py']!=source_hashes()['gaussian_operator.py']:
            raise ValueError('Forward operator source fingerprint differs')
    model=classes[name](**kwargs).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model


def train(roots,output,steps=2000,gpu=3,resume=None,operator_checkpoint=None,
          batch_size=8,eval_every=50,learning_rate=3e-4,seed=61003,operator_weight=0.,size=(192,160),
          input_mode="cnn",observation_mode="rgb",context_mode="features",hidden=128,flow_layers=8,catalog=None,selection_metric="nll",operator_samples=4,category_weight=0.):
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=f'cuda:{gpu}' if gpu>=0 else 'cpu';output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=source_hashes();cases=collect_cases(roots)
    catalog_path=str(catalog) if catalog else None
    if catalog:cases=apply_catalog(cases,catalog)
    loaded=load_cases(cases,size,operator_targets=bool(operator_checkpoint and (operator_weight or selection_metric=="trajectory_rmse")),input_mode=input_mode,observation_mode=observation_mode)
    train_rows=[r for r in loaded if r['recipe']['split']=='train'];val_rows=[r for r in loaded if r['recipe']['split']=='validation']
    if not train_rows or not val_rows:raise ValueError('Need completed training and validation cases')
    model_kwargs={'context_dim':128,'hidden':hidden,'layers':flow_layers,'input_mode':input_mode,
                  'feature_dim':train_rows[0]['features'].shape[-1] if input_mode=='features' else None,
                  'context_mode':context_mode,'observation_mode':observation_mode}
    model=GaussianInverseModel(**model_kwargs).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    start_step=0;best=float('inf')
    if resume:
        _,checkpoint=load_inverse_checkpoint(resume,device);model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state']);start_step=checkpoint['step'];best=checkpoint.get('best_selection_score',checkpoint['best_validation_nll'])
        old_validation={r['key'] for r in checkpoint['cases'] if r['split']=='validation'}
        if old_validation!={r['key'] for r in val_rows}:best=float('inf')
        old_recipes={r['key']:r['recipe_sha256'] for r in checkpoint['cases']}
        for row in loaded:
            if row['key'] in old_recipes and old_recipes[row['key']]!=row['recipe']['recipe_sha256']:
                raise ValueError('Existing supervised case recipe changed across resume')
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu());np.random.set_state(checkpoint['numpy_rng_state'])
        if checkpoint.get('cuda_rng_state') is not None and gpu>=0:torch.cuda.set_rng_state(checkpoint['cuda_rng_state'],device)
        if not (output/'best.pt').exists():best=float('inf')
    operator=load_frozen_operator(operator_checkpoint,device) if operator_checkpoint else None
    if selection_metric=='trajectory_rmse' and operator is None:raise ValueError('Trajectory selection needs a frozen operator checkpoint')
    catalog=[{'key':r['key'],'split':r['recipe']['split'],'group_id':r['recipe']['group_id'],
              'recipe_sha256':r['recipe']['recipe_sha256'],'plan_sha256':r['plan_sha256']} for r in loaded]
    metadata={'roots':[str(Path(r).resolve()) for r in roots],'cases':catalog,'train_count':len(train_rows),'validation_count':len(val_rows),
              'input_protocol':'Public sparse RGB/mask, known reference, restprojectionROI, known cameras/time; depth only for declared RGBD mode',
              'input_mode':input_mode,'observation_mode':observation_mode,'context_mode':context_mode,
              'training_catalog_path':catalog_path,
              'training_catalog_sha256':hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() if catalog_path else None,
              'feature_names':train_rows[0].get('feature_names'),
              'feature_audits':{r['key']:r.get('feature_audits') for r in loaded},
              'label_tolerance_std':LABEL_TOLERANCE.tolist(),'tolerance_note':'Supervised label dequantization and inactive padded dimensions; not posterior uncertainty',
              'source_sha256':frozen,'model_kwargs':model_kwargs,'seed':seed,'size':list(size),
              'training':{'steps_this_run':steps,'batch_size':batch_size,'eval_every':eval_every,
                          'learning_rate':learning_rate,'optimizer':'AdamW','weight_decay':1e-4,
                          'gradient_clip_norm':10.,'random_frame_drop_probability':.35,'always_keep_reference_frame':True},
              'operator_checkpoint':str(operator_checkpoint) if operator_checkpoint else None,'operator_weight':operator_weight,
              'selection_metric':selection_metric,'operator_samples':operator_samples,'category_auxiliary_weight':category_weight,
              'predictive_objective':'Unbiased multi-sample energy score, known TRAIN categories only; validation samples joint unknown categories',
              'operator_checkpoint_sha256':hashlib.sha256(Path(operator_checkpoint).read_bytes()).hexdigest() if operator_checkpoint else None,
              'resume_checkpoint':str(resume) if resume else None,
              'qualification':'Conditional density estimate from finite simulated supervision; uncertainty is uncalibrated',
              'versions':{'numpy':np.__version__,'torch':torch.__version__}}
    (output/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    provenance=output/'provenance';provenance.mkdir(exist_ok=True)
    for name in SOURCE_FILES:(provenance/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    history=[];t0=time.perf_counter();model.train()
    for step in range(start_step,start_step+steps+1):
        if step>start_step:
            ids=np.random.choice(len(train_rows),min(batch_size,len(train_rows)),replace=False)
            batch=batch_rows([train_rows[i] for i in ids],device)
            keep=torch.rand(batch['images'].shape[:2],device=device)>.35;keep[:,0]=True
            context=model.context(batch,keep)
            labels=add_label_tolerance(batch['physical'],torch.randn_like(batch['physical']),model.posterior.transform)
            loss=-model.posterior.log_prob(labels,context,batch['bc'],batch['force']).mean()
            supervised=torch.zeros((),device=device)
            if operator is not None and operator_weight and 'target_positions' in batch:
                point_ids=torch.randperm(batch['rest'].shape[1],device=device)[:64]
                time_ids=torch.randperm(batch['target_times'].shape[1],device=device)[:3]
                prediction,target=forward_draws(operator,model.posterior,context,batch,operator_samples,True,point_ids,time_ids)
                supervised=predictive_energy_score(prediction,target);loss=loss+operator_weight*supervised
            if category_weight:
                bc_logits,force_logits=model.posterior.categorical_logits(context)
                loss=loss+category_weight*(nn.functional.cross_entropy(bc_logits,batch['bc'])+nn.functional.cross_entropy(force_logits,batch['force']))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite inverse training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
        if step==start_step or step%eval_every==0 or step==start_step+steps:
            if source_hashes()!=frozen:raise RuntimeError('Source changed during inverse training')
            score=validation_scores(model,val_rows,device,batch_size=batch_size,operator=operator)
            metric=score['nll'] if selection_metric=='nll' else score['trajectory_rmse_m']
            improved=metric<best
            if improved:best=metric
            record={'step':step,'seconds':time.perf_counter()-t0,'validation':score,
                    'training_loss':float(loss.detach()) if step>start_step else None,
                    'supervised_trajectory_loss':float(supervised.detach()) if step>start_step else None,
                    'best_selection_score':best,'selection_metric':selection_metric}
            history.append(record);print(json.dumps({k:v for k,v in record.items() if k!='validation'},default=str),flush=True)
            checkpoint={**metadata,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                        'step':step,'best_validation_nll':score['nll'],'best_selection_score':best,'validation':score,
                        'torch_rng_state':torch.get_rng_state(),'numpy_rng_state':np.random.get_state(),
                        'cuda_rng_state':torch.cuda.get_rng_state(device) if gpu>=0 else None}
            torch.save(checkpoint,output/'latest.pt')
            if improved:torch.save(checkpoint,output/'best.pt')
            (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=False)['model_state'])
    full=validation_scores(model,val_rows,device,batch_size=batch_size,operator=operator)
    reference=validation_scores(model,val_rows,device,batch_size=batch_size,reference_only=True,operator=operator)
    ablation={'full_observations':full,'reference_frame_only':reference,
              'nll_improvement_from_motion_video':reference['nll']-full['nll'],
              'qualification':'Validation sensitivity to additional observed frames; not uncertainty calibration or active-policy evaluation'}
    (output/'observation-ablation.json').write_text(json.dumps(ablation,indent=2)+'\n')
    return {'best_selection_score':best,'selection_metric':selection_metric,'last_step':step,'train_count':len(train_rows),'validation_count':len(val_rows),'seconds':time.perf_counter()-t0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--gpu',type=int,default=3);parser.add_argument('--resume');parser.add_argument('--operator-checkpoint')
    parser.add_argument('--operator-weight',type=float,default=0.);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--eval-every',type=int,default=50);parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--selection-metric',choices=['nll','trajectory_rmse'],default='nll')
    parser.add_argument('--operator-samples',type=int,default=4);parser.add_argument('--category-weight',type=float,default=0.)
    parser.add_argument('--catalog',help='Immutable common training/validation catalog snapshot')
    parser.add_argument('--input-mode',choices=['cnn','features'],default='cnn')
    parser.add_argument('--observation-mode',choices=['rgb','rgbd'],default='rgb')
    parser.add_argument('--context-mode',choices=['features','prior'],default='features')
    parser.add_argument('--hidden',type=int,default=128);parser.add_argument('--flow-layers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=61003);parser.add_argument('--size',type=int,nargs=2,default=[192,160])
    args=parser.parse_args();print(json.dumps(train(**vars(args)),indent=2))
