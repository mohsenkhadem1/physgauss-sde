"""Train a fixed-object modal neural trajectory decoder using TRAIN motion only.

Validation trajectories select weights and report the TRAIN-basis projection
floor. Neither validation nor test motion enters basis construction. Inference
queries a physical/time MLP and a frozen spatial basis, without mechanics solves.
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
from train_gaussian_forward import collect_records,evaluate
from gaussian_modal_operator import fit_modal_basis,NeuralModalGreenOperator,NeuralModalOperator


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--catalog',type=Path)
    p.add_argument('--model',choices=['green','nonlinear'],default='green')
    p.add_argument('--rank',type=int,default=48);p.add_argument('--hidden',type=int,default=256);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--steps',type=int,default=8000);p.add_argument('--batch',type=int,default=16);p.add_argument('--times',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=500);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=20261107)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    a.output.mkdir(parents=True,exist_ok=False);records=collect_records(a.roots)
    if a.catalog:
        catalog=json.loads(a.catalog.read_text())
        keys={(str(Path(r['path']).resolve()),r['recipe_sha256']) for r in catalog}
        records=[r for r in records if (str(Path(r['path']).resolve()),r['recipe_sha256']) in keys]
        if len(records)!=len(keys):raise ValueError('Frozen catalog contains unavailable or changed recipes')
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation']
    if not train or not val:raise ValueError('Completed TRAIN/VAL data required')
    device=torch.device(a.device)
    basis=fit_modal_basis(train[0]['rest'],[{'split':'train','episode_id':r['episode_id'],'positions_m':r['q']} for r in train],rank=a.rank,seed=a.seed)
    classes={'green':NeuralModalGreenOperator,'nonlinear':NeuralModalOperator};cls=classes[a.model]
    model=cls(basis,hidden=a.hidden,depth=a.depth).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    source_names=['train_gaussian_modal.py','train_gaussian_forward.py','gaussian_modal_operator.py','gaussian_operator.py']
    sources={name:Path(__file__).with_name(name).read_bytes() for name in source_names}
    provenance=a.output/'provenance';provenance.mkdir()
    for name,content in sources.items():(provenance/name).write_bytes(content)
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in sources.items()}
    catalog=[{k:r[k] for k in ['path','episode_id','split','group_id','trajectory_sha256','recipe_sha256']} for r in records]
    (a.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    (a.output/'basis-metadata.json').write_text(json.dumps(basis['metadata'],indent=2)+'\n')
    rest=torch.as_tensor(train[0]['rest'],device=device)
    target=torch.as_tensor(np.stack([r['q'] for r in train]),device=device)
    physical=torch.as_tensor(np.stack([r['physical'] for r in train]),device=device)
    bc=torch.tensor([r['bc'] for r in train],device=device);force=torch.tensor([r['force'] for r in train],device=device)
    scales=torch.tensor([r['scale_m'] for r in train],device=device)
    times=torch.as_tensor(train[0]['times'],device=device)
    # Validate basis capacity separately from coefficient regression accuracy.
    floor=[]
    with torch.no_grad():
        modes=model.basis.modes
        for record in val:
            displacement=torch.tensor(record['q'],device=device)-rest
            coefficients=model.basis.project_displacements(displacement)
            reconstructed=torch.einsum('tr,rnc->tnc',coefficients,modes)
            floor.append({'episode_id':record['episode_id'],'projection_rmse_m':float((reconstructed-displacement).square().mean().sqrt())})
    (a.output/'validation-basis-floor.json').write_text(json.dumps({'mean_projection_rmse_m':float(np.mean([r['projection_rmse_m'] for r in floor])),'episodes':floor,'qualification':'Validation evaluation only; no basis refitting'},indent=2)+'\n')
    protocol={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'train_episodes':len(train),'validation_episodes':len(val),
        'model_class':cls.__name__,'source_sha256':hashes,'basis_fit':'TRAIN-only randomizedSVD, per-case RMS normalization, .4mm floor',
        'supervision':'TRAIN full-mesh coordinate MSE divided by per-case displacement RMS, .4mm floor',
        'checkpoint_selection':'Minimum mean validation full61-time coordinate RMSE',
        'qualification':'One known mesh; directtime neural decoder, no numerical integration or mechanics solve; tests locked'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    best=float('inf');history=[];started=time.monotonic()
    for step in range(1,a.steps+1):
        model.train();ids=torch.tensor(rng.integers(len(train),size=a.batch),device=device)
        tid=torch.tensor(np.sort(rng.choice(np.arange(1,len(times)),size=min(a.times,len(times)-1),replace=False)),device=device)
        prediction=model(rest,times[tid],physical[ids],bc[ids],force[ids])
        loss=((prediction-target[ids][:,tid])/scales[ids,None,None,None]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite modal training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-started}),flush=True)
        if step==1 or step%a.eval_every==0 or step==a.steps:
            validation=evaluate(model,val,device);metric=validation['mean_coordinate_rmse_m']
            checkpoint={'model_class':cls.__name__,'model_kwargs':model.checkpoint()['config'],'model_state':model.state_dict(),
                'modal_checkpoint':model.checkpoint(),'optimizer_state':optimizer.state_dict(),'step':step,'validation':validation,
                'source_sha256':hashes,'catalog':catalog,'protocol':protocol}
            torch.save(checkpoint,a.output/'latest.pt')
            if metric<best:
                best=metric;torch.save(checkpoint,a.output/'best.pt');(a.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
            row={'step':step,'validation_rmse_m':metric,'static_rmse_m':validation['mean_static_coordinate_rmse_m'],'best_rmse_m':best,'seconds':time.monotonic()-started}
            history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
            if any(Path(__file__).with_name(name).read_bytes()!=content for name,content in sources.items()):raise RuntimeError('Frozen modal training source changed')
    print(json.dumps({'completed':True,'best_validation_mm':1000*best,'train_episodes':len(train),'validation_episodes':len(val)}),flush=True)


if __name__=='__main__':main()
