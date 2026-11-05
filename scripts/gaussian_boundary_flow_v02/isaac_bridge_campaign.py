"""Fresh label-supervised global-code posteriors; no test target access.

Prepare freezes all contracts. Training is a separate explicit command. Validation
selection is teacher-to-teacher consistency, NOT measured simulator accuracy.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from train_gaussian_inverse import add_label_tolerance,load_frozen_operator
from gaussian_operator import PhysicalParameterTransform,canonicalize_physical
from evaluate_gaussian_inverse import predict_particles
from scripts.gaussian_boundary_flow_v02.isaac_prefix_cache import sha,write_json
from scripts.gaussian_boundary_flow_v02.isaac_flow_adapter import IsaacConditionalFlowAdapter,IsaacDiagonalMixtureAdapter

MODELS={'flow':IsaacConditionalFlowAdapter,'mdn':IsaacDiagonalMixtureAdapter}
SOURCE_PATHS=[__file__,ROOT/'scripts/gaussian_boundary_flow_v02/isaac_flow_adapter.py',ROOT/'scripts/gaussian_boundary_flow_v02/isaac_prefix_cache.py',ROOT/'scripts/gaussian_boundary_flow/flow.py']+[ROOT/'experiments'/p for p in ['train_gaussian_inverse.py','evaluate_gaussian_inverse.py','gaussian_operator.py','gaussian_observed_features.py','gaussian_observation.py','gaussian_modal_operator.py']]

def sources():return {str(Path(p).resolve().relative_to(ROOT)):sha(p) for p in SOURCE_PATHS}

def load_cache(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text());contract=json.loads((path/'contract.json').read_text())
    if manifest['contract_sha256']!=sha(path/'contract.json'):raise ValueError('Cache contract changed')
    if any(sha(ROOT/p)!=v for p,v in contract['sources'].items()):raise ValueError('Cache preprocessing sources changed')
    # Check the entire role roster BEFORE opening any shard.
    if len(manifest['cases'])!=196 or any(r['role'] not in ('train','validation') for r in manifest['cases']):raise ValueError('Only frozen train/validation roster allowed')
    if sum(r['role']=='train' for r in manifest['cases'])!=152:raise ValueError('Training roster changed')
    catalog_path=ROOT/'runs/gaussian-final-passive-catalog.json'
    if sha(catalog_path)!=contract['catalog_sha256']:raise ValueError('Original catalog binding changed')
    catalog=json.loads(catalog_path.read_text());expected={r['key']:r for r in catalog['cases']}
    actual={r['key']:r for r in manifest['cases']}
    if len(actual)!=len(manifest['cases']) or set(actual)!=set(expected):raise ValueError('Duplicate or changed case roster')
    roles_by_group={}
    for key,row in actual.items():
        ref=expected[key]
        if (row['role'],row['group'],row['recipe_sha256'],row['dataset'])!=(ref['split'],ref['group_id'],ref['recipe_sha256'],key.rsplit('/',1)[0]):
            raise ValueError('Case role/group/recipe differs from bound catalog')
        roles_by_group.setdefault((row['dataset'],row['group']),set()).add(row['role'])
    if any(len(roles)>1 for roles in roles_by_group.values()):raise ValueError('Layout group spans train and validation')
    rows=[]
    for row in manifest['cases']:
        shard=path/row['cache']
        if sha(shard)!=row['cache_sha256']:raise ValueError('Cache shard changed')
        with np.load(shard,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        if arrays['features'].shape!=(5,119) or not np.isfinite(arrays['features']).all():raise ValueError('Invalid strict-prefix cache')
        rows.append(dict(row,**arrays))
    return rows


def fit_logit_standardization(physical,seed=61013,draws=32):
    transform=PhysicalParameterTransform();g=torch.Generator().manual_seed(seed)
    if not transform.contains(physical).all():raise ValueError("Original TRAIN labels outside support; do not hide by clipping")
    expanded=physical.repeat_interleave(draws,0)
    noisy=add_label_tolerance(expanded,torch.randn(expanded.shape,generator=g),transform)
    logits=transform.physical_to_unconstrained(noisy,strict=True)
    return logits.mean(0),logits.std(0,unbiased=False).clamp_min(.1)


def prepare(cache,output,teacher):
    cache=Path(cache).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'recipe.json').exists():raise FileExistsError('Recipe already frozen; verify rather than replace')
    rows=load_cache(cache);train=[r for r in rows if r['role']=='train']
    support_audit={}
    transform=PhysicalParameterTransform()
    for role in ("train","validation"):
        p=torch.tensor(np.stack([r["physical"] for r in rows if r["role"]==role]))
        invalid=(p<=transform.lower)|(p>=transform.upper)|~torch.isfinite(p)
        support_audit[role]=dict(rows=int(len(p)),invalid_rows=int(invalid.any(-1).sum()),invalid_by_coordinate=invalid.sum(0).tolist())
        if invalid.any():raise ValueError("Original labels outside declared support; inspect before proceeding")
    mean,scale=fit_logit_standardization(torch.tensor(np.stack([r['physical'] for r in train])))
    np.savez(output/'standardization.npz',mean=mean.numpy(),scale=scale.numpy())
    recipe=dict(version='isaac-global-code-flow-v01',scope='Global25D physical program posterior, not spatial Gaussian boundary field',
        cache=str(cache),cache_manifest_sha256=sha(cache/'manifest.json'),teacher=str(Path(teacher).resolve()),teacher_sha256=sha(teacher),
        exposed_test_seal=str((ROOT/'reports/gaussian-sealed-study-v01/passive-seal.json').resolve()),
        exposed_test_seal_sha256=sha(ROOT/'reports/gaussian-sealed-study-v01/passive-seal.json'),
        counts=dict(train=152,validation=44,exposed_test_neural=64,exposed_test_orbit=12),original_label_support_audit=support_audit,
        source_sha256=sources(),standardization_sha256=sha(output/'standardization.npz'),
        standardization=dict(source='152 TRAIN physical labels only',seed=61013,dequantization_draws_per_label=32,logit_std_floor=.1),
        models={k:dict(parameters=sum(p.numel() for p in cls().parameters())) for k,cls in MODELS.items()},
        train=dict(seed=61003,steps=2400,batch=16,learning_rate=.0001,weight_decay=.0001,gradient_clip=10.,category_weight=3.,validation_every=50),
        validation=dict(samples=32,flow_steps=32,seed=61023,times_s=[2.,2.5,3.],selection='mean coordinate RMSE against shared teacher conditioned on validation true physical code; not simulator truth'),
        evaluation=dict(samples=128,flow_steps=32,seed=61033,forecast_times_s=[2.,2.5,3.],clock_s='np.arange(61)/20',
                        cohort='76 historically exposed fixed cases; no fresh confirmation claim',metrics=['coordinate_RMSE','rigid_removed_shape_RMSE','90_percent_predictive_interval_coverage_width_score','parameter_coverage_width','category_accuracy','failed_case_denominators','near_bound_saturation'],
                        gate='seal all candidate predictions before independent evaluator reads test labels or trajectories'),
        losses=dict(flow='mean standardized-logit flow matching +3 boundary CE +3 force CE',mdn='per-coordinate standardized-logit mixture NLL +3 boundary CE +3 force CE',
                    both='same labels, tolerances, prefix features, train batches, validation criterion; no teacher-gradient training'),
        archived_comparators='chart/25D operator-supervised checkpoints and prefix ridge remain explicitly different-training archived references')
    write_json(output/'recipe.json',recipe);return recipe


def validate_recipe(output):
    output=Path(output);recipe=json.loads((output/'recipe.json').read_text())
    if recipe['source_sha256']!=sources():raise ValueError('Frozen source binding changed')
    for path,digest in [(recipe['teacher'],recipe['teacher_sha256']),(Path(recipe['cache'])/'manifest.json',recipe['cache_manifest_sha256']),
                        (output/'standardization.npz',recipe['standardization_sha256']),(recipe['exposed_test_seal'],recipe['exposed_test_seal_sha256'])]:
        if sha(path)!=digest:raise ValueError('Frozen artifact binding changed')
    return recipe


def validation_score(model,rows,forward,device,config):
    """Uses validation labels only; no simulator trajectory file is opened."""
    errors=[];saturation=[];model.eval()
    with torch.no_grad():
        for i,row in enumerate(rows):
            x=torch.tensor(row['features'][None],device=device)
            particles={k:v[0] for k,v in model.sample(x,samples=config['samples'],steps=config['flow_steps'],generator=torch.Generator(device=device).manual_seed(config['seed']+i)).items()}
            p=torch.tensor(row['physical'][None],device=device);bc=torch.tensor([int(row['boundary'])],device=device);f=torch.tensor([int(row['force'])],device=device)
            truth=dict(physical=p,effective_physical=canonicalize_physical(p,bc,f),boundary_type=bc,force_type=f)
            bundle=dict(device=device,forward=forward);reference=dict(rest_points_m=row['rest'])
            forecast=predict_particles(bundle,particles,reference,config['times_s'])
            target=predict_particles(bundle,truth,reference,config['times_s'])[0]
            if not torch.isfinite(forecast).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite validation teacher output')
            errors.append(float((forecast.mean(0)-target).square().mean().sqrt()))
            fraction=(particles['physical']-model.transform.lower)/(model.transform.upper-model.transform.lower)
            saturation.append(float(((fraction<1e-4)|(fraction>1-1e-4)).float().mean()))
    if not errors or not np.isfinite(errors).all() or not np.isfinite(saturation).all():raise ValueError('Nonfinite or empty validation score')
    model.train();return dict(teacher_coordinate_rmse_m=float(np.mean(errors)),per_case_rmse_m=errors,near_bound_fraction=float(np.mean(saturation)),cases=len(rows))


def train(output,kind,device):
    torch.set_num_threads(2);output=Path(output);recipe=validate_recipe(output);config=recipe['train'];rows=load_cache(recipe['cache'])
    train_rows=[r for r in rows if r['role']=='train'];val_rows=[r for r in rows if r['role']=='validation']
    torch.manual_seed(config['seed']);model=MODELS[kind]().to(device)
    with np.load(output/'standardization.npz') as f:model.set_standardization(torch.tensor(f['mean'],device=device),torch.tensor(f['scale'],device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    run=output/kind;run.mkdir(exist_ok=True);last=run/'last.pt';recipe_hash=sha(output/'recipe.json')
    data_generator=torch.Generator().manual_seed(config['seed']+1);start=0;best=float('inf')
    if last.exists():
        state=torch.load(last,map_location=device,weights_only=False)
        if state['recipe_sha256']!=recipe_hash or state['kind']!=kind:raise ValueError('Checkpoint binding changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best']
        data_generator.set_state(state['data_rng'].cpu());torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
    forward=load_frozen_operator(recipe['teacher'],device)
    features=torch.tensor(np.stack([r['features'] for r in train_rows]),device=device);physical=torch.tensor(np.stack([r['physical'] for r in train_rows]),device=device)
    boundary=torch.tensor([int(r['boundary']) for r in train_rows],device=device);force=torch.tensor([int(r['force']) for r in train_rows],device=device)
    started=time.time();model.train()
    for step in range(start+1,config['steps']+1):
        ids=torch.randint(len(train_rows),(config['batch'],),generator=data_generator).to(device)
        noise=torch.randn((config['batch'],25),generator=data_generator).to(device)
        target=add_label_tolerance(physical[ids],noise,model.transform)
        optimizer.zero_grad();loss=model.loss(features[ids],target,boundary[ids],force[ids],category_weight=config['category_weight'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; no silent fallback')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        if step%config['validation_every']==0 or step==config['steps']:
            score=validation_score(model,val_rows,forward,device,recipe['validation']);better=score['teacher_coordinate_rmse_m']<best
            if better:best=score['teacher_coordinate_rmse_m']
            state=dict(recipe_sha256=recipe_hash,kind=kind,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
                       data_rng=data_generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None)
            torch.save(state,run/'last.tmp');(run/'last.tmp').replace(last)
            if better:torch.save(state,run/'best.pt')
            progress=dict(step=step,loss=float(loss),gradient_norm=float(norm),best_teacher_rmse_m=best,validation=score,elapsed_s=time.time()-started,state='complete' if step==config['steps'] else 'running',recipe_sha256=recipe_hash)
            write_json(run/'status.json',progress)
            with (run/'history.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--output',default='runs/gaussian-boundary-isaac-bridge-v01');p.add_argument('--cache',default='runs/gaussian-boundary-isaac-prefix-v01');p.add_argument('--teacher',default='runs/gaussian-inverse-final-forward.pt');p.add_argument('--model',choices=list(MODELS),default='flow');p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.command=='prepare':prepare(a.cache,a.output,a.teacher)
    else:
        try:train(a.output,a.model,a.device)
        except Exception as exc:
            failure=Path(a.output)/a.model;failure.mkdir(parents=True,exist_ok=True)
            write_json(failure/'failure.json',dict(state='failed',error_type=type(exc).__name__,error=str(exc)))
            raise
