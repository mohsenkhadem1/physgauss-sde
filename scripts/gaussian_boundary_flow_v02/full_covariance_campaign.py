"""Predeclared full-covariance posterior ablation, preserving matched v02 results.

A 64-case stability validation precedes checkpoint sealing. All four frozen models
are then compared on a new128-case confirmation cohort, never used for tuning.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow.model import observe, PARAMETERS
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow_v02.probabilistic import GaussianMixtureRegressor, ParameterRegressor
from scripts.gaussian_boundary_flow_v02.full_covariance import FullCovarianceMixtureRegressor
from scripts.gaussian_boundary_flow_v02.noise import noisy_context

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/gaussian-boundary-flow-v02-matched'
OUT=ROOT/'runs/gaussian-boundary-flow-v02-full-covariance'
REPORT=ROOT/'reports/gaussian-boundary-flow-v02/full-covariance-results.json'
METHODS=('flow','mixture','full_covariance','point')
PROGRAMS=('sine','release','triangle','reverse')
SIGMA=.00005


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp);tmp.replace(path)
def model_for(name):
    return {'flow':ConditionalFlow,'mixture':GaussianMixtureRegressor,
            'full_covariance':FullCovarianceMixtureRegressor,'point':ParameterRegressor}[name](144)

def frozen_plan():
    sources=[Path(__file__).resolve(),ROOT/'scripts/gaussian_boundary_flow_v02/full_covariance.py',
             ROOT/'scripts/gaussian_boundary_flow_v02/probabilistic.py',ROOT/'scripts/gaussian_boundary_flow_v02/noise.py',
             ROOT/'scripts/gaussian_boundary_flow/flow.py',ROOT/'scripts/gaussian_boundary_flow/model.py',
             ROOT/'scripts/gaussian_boundary_flow/refined.py',ROOT/'scripts/gaussian_boundary_flow/qualification.py',
             ROOT/'src/physgauss/gaussian_boundary.py',ROOT/'src/physgauss/physics.py']
    return {'scope':'Controlled conditional block identification, new confirmation cohort; not real-data adequacy',
            'hypothesis':'A two-component full-covariance posterior may improve support-location coverage relative to the four-component diagonal mixture; no assumed success',
            'training_bank_sha256':sha(BASE/'training.pt'),'training_seed':43091,'training_count':8192,
            'steps':15000,'batch_size':256,'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.00001,
            'gradient_clip':5.,'batch_noise_seed':6135,'initialization_seed':7304,'sigma_m':SIGMA,
            'validation_seed':63119,'validation_count':64,'validation_use':'Finite training/likelihood/samples/rollouts only; no model selection or hyperparameter tuning',
            'confirmation_seed':13401,'confirmation_count':128,'context_noise_seed':13402,'sample_seed':13403,
            'samples':32,'device':'cuda:2','precision':'float32',
            'methods':{name:{'parameters':sum(p.numel()for p in model_for(name).parameters()),'samples':1 if name=='point' else 32}for name in METHODS},
            'existing_checkpoint_sha256':{name:sha(BASE/f'{name}-checkpoint.pt')for name in ('flow','mixture','point')},
            'source_sha256':{str(path.relative_to(ROOT)):sha(path)for path in sources},
            'primary_parameter_check':'support x-location coverage at index3; widths, errors and proper scores reported together',
            'known':['geometry','density','initial state','camera','Gaussian widths','top/bottom boundary topology','force direction','future command'],
            'inferred_parameters':PARAMETERS}

def check_sources(plan):
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Source changed: {name}')
    for name,digest in plan['existing_checkpoint_sha256'].items():
        if sha(BASE/f'{name}-checkpoint.pt')!=digest:raise RuntimeError('Existing checkpoint changed')
    if sha(BASE/'training.pt')!=plan['training_bank_sha256']:raise RuntimeError('Clean bank changed')


def prepare(bank,plan):
    path=OUT/'validation-data.pt'
    if not path.exists():
        rng=torch.Generator().manual_seed(plan['validation_seed'])
        truth=torch.logit(torch.rand(64,6,generator=rng).clamp(.001,.999))
        block=RefinedBlock(device='cpu')
        with torch.no_grad():
            trajectory=block.rollout(truth)
            context=noisy_context(observe(trajectory,noise=False),bank['mean'],bank['scale'],SIGMA,
                                  torch.Generator().manual_seed(plan['validation_seed']+1))
        save(path,{'truth_z':truth,'trajectory':trajectory,'context':context})
    torch.manual_seed(plan['initialization_seed']);model=FullCovarianceMixtureRegressor(144)
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    stream=torch.Generator().manual_seed(plan['batch_noise_seed']);start=0;checkpoint=OUT/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False)
        if state['plan_sha256']!=sha(OUT/'plan.json'):raise RuntimeError('Checkpoint plan mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        stream.set_state(state['stream_rng']);torch.set_rng_state(state['rng']);start=state['step']
    losses=[]
    for step in range(start,plan['steps']):
        ids=torch.randint(len(bank['z']),(256,),generator=stream)
        context=noisy_context(bank['clean'][ids],bank['mean'],bank['scale'],SIGMA,stream)
        loss=model.loss(bank['z'][ids],context)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss; do not evaluate confirmation')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        if (step+1)%500==0:
            save(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                            'rng':torch.get_rng_state(),'stream_rng':stream.get_state(),'plan_sha256':sha(OUT/'plan.json')})
            progress={'phase':'training','step':step+1,'recent_loss':float(np.mean(losses[-500:]))};write(OUT/'progress.json',progress);print(progress,flush=True)
    data=torch.load(path,weights_only=True);model.eval()
    with torch.no_grad():
        logits,means,L=model(data['context']);nll=-model.log_prob(data['truth_z'],data['context'])
        samples=model.sample(data['context'],32,generator=torch.Generator().manual_seed(63121))
        stable=bool(torch.isfinite(nll).all() and torch.isfinite(samples).all() and (L.diagonal(dim1=-2,dim2=-1)>0).all())
        val={'scope':'Validation only; no hyperparameter selection','seed':63119,'finite_positive_factors_and_samples':stable,
             'NLL_mean':float(nll.mean()),'min_cholesky_diagonal':float(L.diagonal(dim1=-2,dim2=-1).min()),
             'max_abs_sample':float(samples.abs().max()),'checkpoint_sha256':sha(checkpoint)}
    save(OUT/'validation-samples.pt',samples);write(OUT/'validation-stability.json',val)
    if not stable:raise RuntimeError('Validation stability failed; preserve checkpoint and stop')
    check_sources(plan)


def metrics(pred,target):
    mean=pred.mean(1);error=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
    row={'future_rmse_mm_mean':float(error.mean()),'per_case_mm':error.tolist()}
    if pred.shape[1]>1:
        lo,hi=torch.quantile(pred[:,:,17:],torch.tensor([.05,.95]),dim=1);truth=target[:,17:]
        inside=(truth>=lo)&(truth<=hi)
        score=hi-lo+20*(lo-truth).clamp_min(0)+20*(truth-hi).clamp_min(0)
        row.update(coordinate_coverage_90=float(inside.float().mean()),trajectory_coverage_90=float(inside.flatten(1).all(1).float().mean()),interval_width_mm=float((hi-lo).mean()*1000),interval_score_90_mm=float(score.mean()*1000))
    return row


def propagate(block,latent,kind):
    flat=latent.flatten(0,1);parts=[]
    for start in range(0,len(flat),128):parts.append(block.rollout(flat[start:start+128].to(block.device),kind=kind).cpu())
    return torch.cat(parts).reshape(len(latent),latent.shape[1],41,8,3)


def evaluate(bank,plan):
    check_sources(plan);validation=json.loads((OUT/'validation-stability.json').read_text())
    if not validation['finite_positive_factors_and_samples']:raise RuntimeError('Validation failed')
    if validation['checkpoint_sha256']!=sha(OUT/'checkpoint.pt'):raise RuntimeError('Checkpoint changed since validation')
    block=RefinedBlock(device='cuda:2')
    with torch.no_grad():
        # No confirmation data is created until the validation physical stability check passes.
        val_data=torch.load(OUT/'validation-data.pt',weights_only=True)
        val_samples=torch.load(OUT/'validation-samples.pt',weights_only=True)
        val_motion=propagate(block,val_samples,'sine')
        if not bool(torch.isfinite(val_motion).all()):raise RuntimeError('Validation rollout nonfinite')
        validation['rollout_metrics']=metrics(val_motion,val_data['trajectory']);validation['finite_rollouts']=True
        write(OUT/'validation-stability.json',validation)
        freeze={'plan_sha256':sha(OUT/'plan.json'),'full_covariance_checkpoint_sha256':sha(OUT/'checkpoint.pt'),
                'existing_checkpoint_sha256':plan['existing_checkpoint_sha256'],'validation_sha256':sha(OUT/'validation-stability.json')}
        f=OUT/'confirmation-freeze.json'
        if f.exists() and json.loads(f.read_text())!=freeze:raise RuntimeError('Confirmation freeze changed')
        write(f,freeze)
        rng=torch.Generator().manual_seed(plan['confirmation_seed']);truth=torch.logit(torch.rand(128,6,generator=rng).clamp(.001,.999))
        base=block.rollout(truth.to(block.device)).cpu();context=noisy_context(observe(base,noise=False),bank['mean'],bank['scale'],SIGMA,torch.Generator().manual_seed(plan['context_noise_seed']))
        save(OUT/'confirmation-data.pt',{'truth_z':truth,'context':context})
        samples={};parameter_scores={};physical_truth=block.parameters(truth);qtruth=torch.sigmoid(truth)
        for name in METHODS:
            model=model_for(name);path=OUT/'checkpoint.pt' if name=='full_covariance' else BASE/f'{name}-checkpoint.pt'
            model.load_state_dict(torch.load(path,weights_only=False)['model']);model.eval()
            sample=model(context)[:,None] if name=='point' else model.sample(context,count=32,generator=torch.Generator().manual_seed(plan['sample_seed']))
            samples[name]=sample;physical=block.parameters(sample);q=torch.sigmoid(sample)
            energy=(q-qtruth[:,None]).norm(dim=-1).mean(1)
            if sample.shape[1]>1:
                pair=(q[:,:,None]-q[:,None,:]).norm(dim=-1)
                energy-=pair.sum((1,2))/(2*32*31)
            row={'absolute_mean_error':(physical.mean(1)-physical_truth).abs().mean(0).tolist(),
                 'normalized_parameter_energy_score_mean':float(energy.mean()),'units':'Log SI parameters except x locations in metres'}
            if sample.shape[1]>1:
                lo,hi=torch.quantile(physical,torch.tensor([.05,.95]),dim=1)
                score=hi-lo+20*(lo-physical_truth).clamp_min(0)+20*(physical_truth-hi).clamp_min(0)
                row.update(coverage_90=((physical_truth>=lo)&(physical_truth<=hi)).float().mean(0).tolist(),interval_width=(hi-lo).mean(0).tolist(),interval_score_90=score.mean(0).tolist())
            if name in ('mixture','full_covariance'):row['logit_NLL_mean']=float(-model.log_prob(truth,context).mean())
            parameter_scores[name]=row
        save(OUT/'parameter-samples.pt',samples);programs={}
        for kind in PROGRAMS:
            target=block.rollout(truth.to(block.device),kind=kind).cpu();rows={};means={}
            for name in METHODS:
                pred=propagate(block,samples[name],kind);rows[name]=metrics(pred,target);means[name]=pred.mean(1)
            programs[kind]=rows;save(OUT/f'predictions-{kind}.pt',{'truth':target,'means':means});write(OUT/f'scores-{kind}.json',rows)
            print(kind,{name:row['future_rmse_mm_mean']for name,row in rows.items()},flush=True)
        cpu=RefinedBlock(device='cpu');cpu_difference=float((cpu.rollout(truth[:4])-base[:4]).abs().max())
    rng=np.random.default_rng(13404);comparisons={}
    for kind,rows in programs.items():
        comparisons[kind]={};new=np.array(rows['full_covariance']['per_case_mm'])
        for other in ('flow','mixture','point'):
            difference=np.array(rows[other]['per_case_mm'])-new
            bootstrap=difference[rng.integers(0,128,(5000,128))].mean(1)
            comparisons[kind][other]={'other_minus_full_covariance_mm':float(difference.mean()),'paired_bootstrap_95CI_mm':np.quantile(bootstrap,[.025,.975]).tolist(),'full_covariance_wins':int((difference>0).sum()),'cases':128}
    check_sources(plan)
    result={'scope':plan['scope'],'hypothesis':plan['hypothesis'],'plan_sha256':sha(OUT/'plan.json'),
            'confirmation_freeze':freeze,'validation':validation,'confirmation_seed':13401,'confirmation_count':128,
            'methods':plan['methods'],'parameter_scores':parameter_scores,'programs':programs,'paired_comparisons':comparisons,
            'cpu_gpu_max_displacement_difference_m_4cases':cpu_difference,'source_hashes_unchanged':True,
            'same_context_all_methods':True,'limitations':['Known conditional block mechanism; no real-world or unknown-topology claim',
            'Full covariance ablation also changes component count four to two; architecture choice was fixed before validation',
            '32-sample quantiles have Monte Carlo error; coordinate intervals are not simultaneous trajectory guarantees',
            'No tuning or checkpoint selection on confirmation cohort']}
    write(OUT/'results.json',result);write(REPORT,result);write(OUT/'progress.json',{'phase':'complete'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','evaluate','all'),default='prepare');args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;OUT.mkdir(parents=True,exist_ok=True)
    plan=frozen_plan();p=OUT/'plan.json'
    if p.exists():
        if json.loads(p.read_text())!=plan:raise RuntimeError('Immutable plan changed')
    else:
        write(p,plan)
        for relative in plan['source_sha256']:
            target=OUT/'source-snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
    bank=torch.load(BASE/'training.pt',weights_only=True);start=time.time()
    if args.phase in ('prepare','all'):prepare(bank,plan)
    if args.phase in ('evaluate','all'):evaluate(bank,plan)
    print('elapsed_this_invocation_seconds',time.time()-start,flush=True)

if __name__=='__main__':main()
