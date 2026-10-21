"""Structured cuboid surface-chart posterior; separate from the25D baseline."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gaussian_chart_posterior import SurfaceChartPosterior,encode_labels,dequantize_chart,CHART_NAMES,TOLERANCE,LOWER,UPPER
from gaussian_observed_features import ObservedFeatureEncoder
import train_gaussian_inverse as base

SOURCE_FILES=tuple(dict.fromkeys((*base.SOURCE_FILES,'gaussian_chart_posterior.py','train_gaussian_chart_inverse.py')))


def source_hashes():return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


class ChartGaussianInverseModel(nn.Module):
    def __init__(self,feature_dim,context_dim=128,hidden=64,layers=4,observation_mode='rgb',context_mode='features',center=(0,0,.25),size=(.24,.12,.08),input_mode='features'):
        super().__init__();self.observation_mode=observation_mode;self.input_mode='features'
        self.encoder=ObservedFeatureEncoder(feature_dim,context_dim,hidden,context_mode)
        self.posterior=SurfaceChartPosterior(context_dim,hidden,layers,center,size)
    def context(self,batch,observation_mask=None):return self.encoder(batch['features'],observation_mask)
    def infer(self,batch,samples=32,observation_mask=None,generator=None):return self.posterior.sample(self.context(batch,observation_mask),samples,generator=generator)


def load_chart_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get('model_class')!='ChartGaussianInverseModel':raise ValueError('Wrong structured inverse checkpoint class')
    if checkpoint['source_sha256']!=source_hashes():raise ValueError('Structured inverse source changed; archived source replay is required')
    model=ChartGaussianInverseModel(**checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval();return model,checkpoint


def load_rows(cases,mode,size):
    rows=base.load_cases(cases,size,operator_targets=True,input_mode='features',observation_mode=mode)
    for row in rows:
        spec=json.loads((row['path']/'truth/boundaries.json').read_text())
        labels=encode_labels(row['recipe'],spec,row['rest'])
        row.update({k:labels[k] for k in ('chart','support_face','load_faces')})
    return rows


def batch(rows,device):
    result=base.batch_rows(rows,device)
    for name in ('chart','support_face','load_faces'):
        result[name]=torch.as_tensor(np.stack([r[name] for r in rows]),device=device)
    result['chart']=result['chart'].float();result['support_face']=result['support_face'].long();result['load_faces']=result['load_faces'].long()
    return result


def draw_observation_mask(times,generator=None):
    keep=torch.rand(times.shape,device=times.device,generator=generator)>.35
    prefix=torch.rand(len(times),device=times.device,generator=generator)<.5
    keep=torch.where(prefix[:,None],times<=1.+1e-8,keep);keep[:,0]=True
    return keep,prefix


def forward_samples(operator,model,context,data,samples,known=False,generator=None,point_ids=None,time_ids=None):
    categories={'boundary_type':data['bc'],'force_type':data['force'],'support_face':data['support_face'],'load_faces':data['load_faces']} if known else {}
    draw=model.posterior.sample(context,samples=samples,generator=generator,**categories)
    points=data['rest'];times=data['target_times'];target=data['target_positions']
    if point_ids is not None:points=points[:,point_ids];target=target[:,:,point_ids]
    if time_ids is not None:times=times[:,time_ids];target=target[:,time_ids]
    b=len(points)
    q=operator(points.repeat_interleave(samples,0),times.repeat_interleave(samples,0),draw['effective_physical'].reshape(b*samples,25),draw['boundary_type'].reshape(-1),draw['force_type'].reshape(-1))
    return q.reshape(b,samples,len(times[0]),len(points[0]),3),target


def validate(model,rows,operator,device,batch_size=8,reference_only=False,prefix_only=False):
    model.eval();generator=torch.Generator(device=device).manual_seed(93211);records=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            chosen=rows[start:start+batch_size];data=batch(chosen,device)
            mask=None
            if reference_only:
                mask=torch.zeros(data['features'].shape[:2],device=device,dtype=torch.bool);mask[:,0]=True
            elif prefix_only:mask=data['times']<=1.+1e-8
            context=model.context(data,mask);nll=[]
            for _ in range(4):
                labels=dequantize_chart(data['chart'],data['bc'],data['force'],generator)
                nll.append(-model.posterior.log_prob_chart(labels,context,data['bc'],data['force'],data['support_face'],data['load_faces']))
            bc,force=model.posterior.categorical_logits(context)
            points=torch.linspace(0,data['rest'].shape[1]-1,64,device=device).long()
            q,target=forward_samples(operator,model,context,data,32,generator=generator,point_ids=points)
            error=(q.mean(1)-target).square().mean((1,2,3)).sqrt()
            for i,row in enumerate(chosen):records.append({'key':row['key'],'nll_intrinsic':float(torch.stack(nll)[:,i].mean()),
                'trajectory_rmse_m':float(error[i]),'boundary_correct':bool(bc[i].argmax()==data['bc'][i]),'force_correct':bool(force[i].argmax()==data['force'][i])})
    model.train()
    return {'trajectory_rmse_m':float(np.mean([r['trajectory_rmse_m'] for r in records])),
            'nll_intrinsic':float(np.mean([r['nll_intrinsic'] for r in records])),
            'boundary_accuracy':float(np.mean([r['boundary_correct'] for r in records])),
            'force_accuracy':float(np.mean([r['force_correct'] for r in records])),'cases':records}


def train(roots,catalog,output,operator_checkpoint,observation_mode='rgb',steps=2400,gpu=-1,batch_size=16,
          eval_every=50,learning_rate=1e-4,hidden=64,layers=4,operator_weight=1000.,operator_samples=4,
          category_weight=3.,seed=61003,context_mode='features',size=(192,160),resume=None):
    if operator_samples<2:raise ValueError('Unbiased energy scoring needs at least two samples')
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=f'cuda:{gpu}' if gpu>=0 else 'cpu';output=Path(output)
    if output.exists() and not resume:raise FileExistsError('Choose a new structured inverse run directory')
    output.mkdir(parents=True,exist_ok=True);frozen=source_hashes()
    rows=load_rows(base.apply_catalog(base.collect_cases(roots),catalog),observation_mode,size)
    train_rows=[r for r in rows if r['recipe']['split']=='train'];val_rows=[r for r in rows if r['recipe']['split']=='validation']
    if not train_rows or not val_rows:raise ValueError('Need completed TRAIN and VAL supervision')
    p=train_rows[0]['rest'];center=(p.max(0)+p.min(0))/2;extent=p.max(0)-p.min(0)
    if any(not np.allclose(r['rest'].min(0),p.min(0)) or not np.allclose(r['rest'].max(0),p.max(0)) for r in rows):raise ValueError('This chart version requires one shared known cuboid')
    kwargs={'feature_dim':train_rows[0]['features'].shape[-1],'context_dim':128,'hidden':hidden,'layers':layers,
            'observation_mode':observation_mode,'context_mode':context_mode,'center':center.tolist(),'size':extent.tolist(),'input_mode':'features'}
    model=ChartGaussianInverseModel(**kwargs).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    operator=base.load_frozen_operator(operator_checkpoint,device);start=0;best=float('inf')
    if resume:
        old,checkpoint=load_chart_checkpoint(resume,device);model.load_state_dict(old.state_dict());optimizer.load_state_dict(checkpoint['optimizer_state'])
        if checkpoint['training_catalog_sha256']!=hashlib.sha256(Path(catalog).read_bytes()).hexdigest():raise ValueError('Resume catalog changed')
        start=checkpoint['step'];best=checkpoint['best_selection_score'];torch.set_rng_state(checkpoint['torch_rng_state'].cpu());np.random.set_state(checkpoint['numpy_rng_state'])
        if gpu>=0 and checkpoint.get('cuda_rng_state') is not None:torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(),device)
        if not (output/'best.pt').exists():best=float('inf')
    metadata={'model_class':'ChartGaussianInverseModel','model_kwargs':kwargs,'source_sha256':frozen,'train_count':len(train_rows),'validation_count':len(val_rows),
              'cases':[{'key':r['key'],'split':r['recipe']['split'],'recipe_sha256':r['recipe']['recipe_sha256']} for r in rows],
              'training_catalog_path':str(catalog),'training_catalog_sha256':hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
              'operator_checkpoint':str(operator_checkpoint),'operator_checkpoint_sha256':hashlib.sha256(Path(operator_checkpoint).read_bytes()).hexdigest(),
              'input_mode':'features','observation_mode':observation_mode,'feature_version':'intrinsic_surface_chart_v1','size':list(size),'seed':seed,
              'resume_checkpoint':str(resume) if resume else None,
              'chart_names':CHART_NAMES,'chart_lower':LOWER.tolist(),'chart_upper':UPPER.tolist(),'active_label_tolerance_std':TOLERANCE.tolist(),
              'inactive_label_policy':'Uniform independent bounded nuisance coefficients; decoder removes inactive physical controls',
              'density_measure':'16D intrinsic coefficients plus joint BC/load16 and autoregressive support/load-face6 categories; NOT25D physical Lebesgue density',
              'geometry_policy':'Known fixed cuboid only; support1 oppositeface only two_patches. Load tangents clipped from[-.55,.55] to finite[-.5,.5]face: noninvertible projection, explicitly outside any density Jacobian.',
              'training':{'steps':steps,'batch_size':batch_size,'learning_rate':learning_rate,'operator_weight':operator_weight,'operator_samples':operator_samples,'category_weight':category_weight},
              'observation_mask_distribution':{'exact_prefix_times_le_1s_probability':.5,'random_frame_dropout_probability':.35,
                  'random_dropout_component_probability':.5,'always_keep_reference_frame':True,
                  'note':'New chart training differs from preserved25D baseline random-dropout-only masks; evaluation conditions are matched explicitly.'},
              'predictive_objective':'Unbiased energy score; known TRAIN categories/faces only; validation samples all unknown categories jointly',
              'qualification':'Physical hypotheses satisfy declared surface/direction/range constraints; posterior accuracy/calibration remain empirical questions, not guaranteed.'}
    (output/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n');provenance=output/'provenance';provenance.mkdir(exist_ok=True)
    for name in SOURCE_FILES:(provenance/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    history=[];t0=time.perf_counter();model.train()
    for step in range(start,start+steps+1):
        loss=None;energy=None
        if step>start:
            ids=np.random.choice(len(train_rows),min(batch_size,len(train_rows)),replace=False);data=batch([train_rows[i] for i in ids],device)
            keep,_=draw_observation_mask(data['times'])
            context=model.context(data,keep)
            chart=dequantize_chart(data['chart'],data['bc'],data['force'])
            discrete=model.posterior.discrete_log_prob(context,data['bc'],data['force'],data['support_face'],data['load_faces'])
            nll=-model.posterior.log_prob_chart(chart,context,data['bc'],data['force'],data['support_face'],data['load_faces']).mean()
            points=torch.randperm(data['rest'].shape[1],device=device)[:64];times=torch.randperm(6,device=device)[:3]
            q,target=forward_samples(operator,model,context,data,operator_samples,known=True,point_ids=points,time_ids=times)
            energy=base.predictive_energy_score(q,target)
            loss=nll-category_weight*discrete.mean()+operator_weight*energy
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        if step%eval_every==0 or step==start+steps:
            if source_hashes()!=frozen:raise RuntimeError('Structured training source changed')
            validation=validate(model,val_rows,operator,device);score=validation['trajectory_rmse_m'];improved=score<best;best=min(best,score)
            log={'step':step,'seconds':time.perf_counter()-t0,'training_loss':float(loss.detach()) if loss is not None else None,
                 'energy_score':float(energy.detach()) if energy is not None else None,'validation':validation,'best_selection_score':best}
            history.append(log);checkpoint={**metadata,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'step':step,
                'best_selection_score':best,'validation':validation,'torch_rng_state':torch.get_rng_state(),'numpy_rng_state':np.random.get_state(),
                'cuda_rng_state':torch.cuda.get_rng_state(device) if gpu>=0 else None}
            torch.save(checkpoint,output/'latest.pt')
            if improved:torch.save(checkpoint,output/'best.pt')
            (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(json.dumps({k:v for k,v in log.items() if k!='validation'}|{'boundary_accuracy':validation['boundary_accuracy'],'force_accuracy':validation['force_accuracy']}),flush=True)
    model,_=load_chart_checkpoint(output/'best.pt',device)
    full=validate(model,val_rows,operator,device);reference=validate(model,val_rows,operator,device,reference_only=True)
    prefix=validate(model,val_rows,operator,device,prefix_only=True)
    (output/'observation-ablation.json').write_text(json.dumps({'full':full,'reference_only':reference,'prefix_le_1s':prefix,'qualification':'Development-set sensitivity, not held-out calibration'},indent=2)+'\n')
    return {'best_selection_score':best,'last_step':step,'train_count':len(train_rows),'validation_count':len(val_rows),'seconds':time.perf_counter()-t0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True);p.add_argument('--catalog',required=True)
    p.add_argument('--output',required=True);p.add_argument('--operator-checkpoint',required=True);p.add_argument('--observation-mode',choices=['rgb','rgbd'],default='rgb')
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--gpu',type=int,default=-1);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=50);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=4);p.add_argument('--operator-weight',type=float,default=1000.)
    p.add_argument('--operator-samples',type=int,default=4);p.add_argument('--category-weight',type=float,default=3.)
    p.add_argument('--seed',type=int,default=61003);p.add_argument('--context-mode',choices=['features','prior'],default='features')
    p.add_argument('--size',type=int,nargs=2,default=[192,160]);p.add_argument('--resume');a=p.parse_args();print(json.dumps(train(**vars(a)),indent=2))
