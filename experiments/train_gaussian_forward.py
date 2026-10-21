"""Train a physical-input neural trajectory surrogate using TRAIN simulation labels.

The validation metric below supplies true physical inputs and measures forward
surrogate error. It is NOT inverse inference performance. Test cases are never
opened. Model runtime contains no FEM solve, simulator, or image shortcut.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from gaussian_operator import (NeuralTrajectoryOperator,NeuralGreenOperator,
    physical_vector,category_labels,deformation_diagnostics)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_records(roots,splits=('train','validation')):
    """Freeze a snapshot of independently validated records; never read test labels."""
    records=[]
    for root in map(Path,roots):
        plan=json.loads((root/'plan.json').read_text())
        for case in plan['episodes']:
            if case['split'] not in splits:continue
            episode=root/'episodes'/case['episode_id']
            if not (episode/'validation.json').exists():continue
            if not json.loads((episode/'validation.json').read_text()).get('passed'):continue
            if not json.loads((episode/'summary.json').read_text()).get('completed'):continue
            # These are explicitly TRAIN/VAL physical supervision, never inputs
            # of the image-conditioned inverse prediction pipeline.
            with np.load(episode/'reference.npz') as raw:rest=raw['rest_points_m'].astype(np.float32);tets=raw['tetrahedra']
            with np.load(episode/'truth/trajectory.npz') as raw:q=raw['positions_m'].astype(np.float32);times=raw['times_s'].astype(np.float32)
            spec=json.loads((episode/'truth/boundaries.json').read_text())
            bc,force=category_labels(case);physical=physical_vector(case,spec,rest)
            records.append({'path':str(episode),'split':case['split'],'episode_id':case['episode_id'],
                'group_id':str(root)+':'+case['group_id'],'physical':physical,'bc':bc,'force':force,
                'rest':rest,'tets':tets,'q':q,'times':times,
                'scale_m':max(float(np.sqrt(np.mean((q-rest)**2))),.0004),
                'trajectory_sha256':sha(episode/'truth/trajectory.npz'),'recipe_sha256':case['recipe_sha256']})
    if not records:raise ValueError('No completed TRAIN/VAL episodes')
    return records


def make_model(model_class,kwargs,rest):
    classes={'NeuralTrajectoryOperator':NeuralTrajectoryOperator,'NeuralGreenOperator':NeuralGreenOperator}
    return classes[model_class](**kwargs,reference_points=rest)


def evaluate(model,records,device):
    model.eval();rows=[]
    with torch.no_grad():
        for record in records:
            rest=torch.as_tensor(record['rest'],device=device);times=torch.as_tensor(record['times'],device=device)
            physical=torch.as_tensor(record['physical'][None],device=device)
            bc=torch.tensor([record['bc']],device=device);force=torch.tensor([record['force']],device=device)
            predicted=model(rest,times,physical,bc,force)[0]
            target=torch.as_tensor(record['q'],device=device)
            error=float(torch.sqrt((predicted-target).square().mean()))
            static=float(torch.sqrt((rest-target).square().mean()))
            gap=(times>1)&(times<2)
            gap_error=float(torch.sqrt((predicted[gap]-target[gap]).square().mean()))
            diagnostics=deformation_diagnostics(predicted,rest,torch.as_tensor(record['tets'],device=device,dtype=torch.long))
            rows.append({'episode_id':record['episode_id'],'split':record['split'],'group_id':record['group_id'],
                'coordinate_rmse_m':error,'gap_coordinate_rmse_m':gap_error,'static_coordinate_rmse_m':static,
                'relative_error_to_static':error/max(static,1e-9),**diagnostics})
    return {'mean_coordinate_rmse_m':float(np.mean([r['coordinate_rmse_m'] for r in rows])),
        'mean_static_coordinate_rmse_m':float(np.mean([r['static_coordinate_rmse_m'] for r in rows])),
        'median_relative_error_to_static':float(np.median([r['relative_error_to_static'] for r in rows])),
        'positive_volume_episodes':sum(r['positive_volume'] for r in rows),'episodes':rows,
        'qualification':'True physical parameters supplied: forward surrogate validation, not inverse prediction'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=['green','nonlinear'],default='green')
    parser.add_argument('--steps',type=int,default=12000);parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--points',type=int,default=160);parser.add_argument('--times',type=int,default=12)
    parser.add_argument('--hidden',type=int,default=128);parser.add_argument('--rank',type=int,default=24)
    parser.add_argument('--depth',type=int,default=3);parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--eval-every',type=int,default=500);parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--resume',type=Path)
    parser.add_argument('--catalog',type=Path,help='Immutable existing record snapshot; recipe hashes must match')
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    root=args.output;root.mkdir(parents=True,exist_ok=False);device=torch.device(args.device)
    records=collect_records(args.roots)
    if args.catalog:
        frozen_catalog=json.loads(args.catalog.read_text())
        keys={(str(Path(r['path']).resolve()),r['recipe_sha256']) for r in frozen_catalog}
        records=[r for r in records if (str(Path(r['path']).resolve()),r['recipe_sha256']) in keys]
        if len(records)!=len(keys):raise ValueError('Frozen forward catalog has unavailable/changed recipes')
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation']
    if not train or not val:raise ValueError('Need both TRAIN and VAL snapshots')
    shapes={(r['rest'].shape,r['q'].shape) for r in records}
    if len(shapes)!=1:raise ValueError('This initial batching runner requires shared mesh topology/time grid')
    model_class='NeuralGreenOperator' if args.model=='green' else 'NeuralTrajectoryOperator'
    kwargs={'hidden':args.hidden,'rank':args.rank,'depth':args.depth,'seconds':3.,'displacement_scale_m':.03}
    model=make_model(model_class,kwargs,train[0]['rest']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        if checkpoint['model_class']!=model_class or checkpoint['model_kwargs']!=kwargs:raise ValueError('Resume architecture mismatch')
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:optimizer.load_state_dict(checkpoint['optimizer_state'])
    names=['train_gaussian_forward.py','gaussian_operator.py']
    sources={name:Path(__file__).with_name(name).read_bytes() for name in names}
    (root/'provenance').mkdir()
    for name,content in sources.items():(root/'provenance'/name).write_bytes(content)
    source_hashes={name:hashlib.sha256(content).hexdigest() for name,content in sources.items()}
    catalog=[{k:r[k] for k in ['path','episode_id','split','group_id','trajectory_sha256','recipe_sha256']} for r in records]
    (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    protocol={**vars(args),'output':str(root),'resume':str(args.resume) if args.resume else None,'catalog':str(args.catalog) if args.catalog else None,
        'train_episodes':len(train),'validation_episodes':len(val),'no_test_labels_opened':True,
        'source_sha256':source_hashes,'model_class':model_class,'model_kwargs':kwargs,
        'supervision':'TRAIN nodal motion and physical parameters; per-case displacement-normalized MSE, 0.4mm scale floor',
        'checkpoint_selection':'Minimum mean coordinate RMSE on validation only','runtime':'Direct time neural evaluation; no FEM or numerical integration',
        'torch_disable_native_jit':os.environ['TORCH_DISABLE_NATIVE_JIT']}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rest=torch.as_tensor(np.stack([r['rest'] for r in train]),device=device)
    q=torch.as_tensor(np.stack([r['q'] for r in train]),device=device)
    physical=torch.as_tensor(np.stack([r['physical'] for r in train]),device=device)
    bc=torch.tensor([r['bc'] for r in train],device=device);force=torch.tensor([r['force'] for r in train],device=device)
    scales=torch.tensor([r['scale_m'] for r in train],device=device)
    times_all=torch.as_tensor(train[0]['times'],device=device)
    started=time.monotonic();best=float('inf');history=[]
    for step in range(1,args.steps+1):
        model.train();ids=torch.as_tensor(rng.integers(len(train),size=args.batch),device=device)
        node_ids=torch.as_tensor(rng.choice(rest.shape[1],size=min(args.points,rest.shape[1]),replace=False),device=device)
        time_ids=torch.as_tensor(np.sort(rng.choice(np.arange(1,len(times_all)),size=min(args.times,len(times_all)-1),replace=False)),device=device)
        prediction=model(rest[ids][:,node_ids],times_all[time_ids],physical[ids],bc[ids],force[ids])
        target=q[ids][:,time_ids][:,:,node_ids]
        loss=((prediction-target)/scales[ids,None,None,None]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite forward training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%100==0:print(json.dumps({'step':step,'train_loss':float(loss.detach()),'gradient':float(gradient),'elapsed_s':time.monotonic()-started}),flush=True)
        if step==1 or step%args.eval_every==0 or step==args.steps:
            validation=evaluate(model,val,device);metric=validation['mean_coordinate_rmse_m']
            history.append({'step':step,'train_loss':float(loss.detach()),'validation_rmse_m':metric,'static_rmse_m':validation['mean_static_coordinate_rmse_m'],
                'positive_volume_episodes':validation['positive_volume_episodes'],'seconds_elapsed':time.monotonic()-started})
            checkpoint={'model_class':model_class,'model_kwargs':kwargs,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                'step':step,'validation':validation,'source_sha256':source_hashes,'catalog':catalog,'protocol':protocol}
            torch.save(checkpoint,root/'latest.pt')
            if metric<best:best=metric;torch.save(checkpoint,root/'best.pt');(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
            (root/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps({'validation_step':step,'rmse_mm':metric*1000,'static_mm':validation['mean_static_coordinate_rmse_m']*1000,'best_mm':best*1000}),flush=True)
            if any(Path(__file__).with_name(name).read_bytes()!=content for name,content in sources.items()):raise RuntimeError('Training source changed; preserve run and restart under fresh version')
    print(json.dumps({'completed':True,'best_validation_rmse_mm':best*1000,'train_episodes':len(train),'validation_episodes':len(val),'seconds_elapsed':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
