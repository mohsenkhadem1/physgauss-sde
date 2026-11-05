"""Bounded training-bank reconstruction diagnostic; no real future targets."""
import argparse,json,signal,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.package_campaign_v03 import (load_package,sha,write,load_transform,load_bound_model,relative_observation_noise,GeometryAwareGaussianBoundaryBackend,WideGaussianTableBoundaryBackend,ConditionalFlow,GaussianMixtureRegressor)
ROOT=Path(__file__).resolve().parents[2]
CASES=('single_push_sloth','object-soft-task1-demo0')
class DeadlineExceeded(Exception):pass
@contextmanager
def deadline(seconds):
    def fail(*args):raise DeadlineExceeded('Declared per-rollout deadline exceeded')
    old=signal.signal(signal.SIGALRM,fail);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def metrics(pred,true):
    pred=np.asarray(pred,float);true=np.asarray(true,float)
    if pred.shape!=true.shape or pred.ndim!=3 or not np.isfinite(pred).all():raise ValueError('Invalid trajectories')
    ca=pred.mean(1);cb=true.mean(1);a=pred-ca[:,None];b=true-cb[:,None]
    u,_,vh=np.linalg.svd(a.transpose(0,2,1)@b);c=np.tile(np.eye(3),(len(a),1,1));c[:,-1,-1]=np.linalg.det(u@vh);rot=u@c@vh
    return dict(coordinate_rmse_mm=float(np.sqrt(np.mean((pred-true)**2))*1000),initial_coordinate_rmse_mm=float(np.sqrt(np.mean((pred[0]-true[0])**2))*1000),centroid_rmse_mm=float(np.sqrt(np.mean((ca-cb)**2))*1000),aligned_shape_rmse_mm=float(np.sqrt(np.mean((a@rot-b)**2))*1000),rigid_alignment_angle_mean_deg=float(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1)).mean()*180/np.pi))

def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():raise FileExistsError('Plan immutable')
    artifacts={};cases=[];sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for case in CASES:
        run=ROOT/'runs/gaussian-boundary-flow-v03-package'/case;old=json.loads((run/'plan.json').read_text())
        for name,digest in old['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Original numerical source changed: '+name)
            sources[name]=digest
        for name in ('plan.json','bank.npz','transform.pt','flow.pt','mixture.pt'):
            p=run/name;artifacts[str(p)]=sha(p)
        package=Path(old['input_package'])
        for p in package.iterdir():
            if p.is_file():artifacts[str(p)]=sha(p)
        cases.append(dict(case=case,run=str(run),bank_index=0))
    plan=dict(scope='TRAINING reconstruction on two fixed existing bank worlds, not generalization or real target accuracy',cases=cases,artifacts=artifacts,sources=sources,draws=8,noise_sigma_m=.001,noise_seed=717,posterior_seed=138,rollout_deadline_s=30,selection='bank index0 each; no replacements',nearest='raw masked relative feature MSE',forecast='full existing known command clock, same initial rest/theta backend; synthetic truth only',failure_policy='retain every requested draw; means conditional on survivors explicitly, no threshold changes')
    write(output/'plan.json',plan)

def run(output,device):
    output=Path(output);plan=json.loads((output/'plan.json').read_text())
    if any(sha(p)!=h for p,h in plan['artifacts'].items()) or any(sha(ROOT/p)!=h for p,h in plan['sources'].items()):raise ValueError('Frozen binding changed')
    torch.set_num_threads(2)
    for spec in plan['cases']:
        dest=output/spec['case'];dest.mkdir(exist_ok=False);source=Path(spec['run']);old=json.loads((source/'plan.json').read_text());data=load_package(old['input_package'])
        backend=GeometryAwareGaussianBoundaryBackend if old['args']['geometry']=='filtered' else WideGaussianTableBoundaryBackend;b=backend(data,device)
        with np.load(source/'bank.npz') as raw:zbank=raw['z'];features=raw['features']
        frames=np.array(old['prefix_frames']);ids=np.array(old['point_ids']);mask=np.repeat(data['observed_mask'][frames][:,ids,None],3,-1).reshape(-1)
        tr=load_transform(source/'transform.pt',bank_path=source/'bank.npz',plan_path=source/'plan.json',device=device)
        noise=relative_observation_noise(1,len(frames),len(ids),plan['noise_sigma_m'],mask=mask,generator=torch.Generator(device=device).manual_seed(plan['noise_seed']),device=device)[0]
        clean=torch.tensor(features[0],dtype=torch.float32,device=device);observed=clean+noise;context=((observed-tr['mean'])@tr['basis'])/tr['scale']
        models={'flow':ConditionalFlow(len(tr['scale']),b.parameter_dim),'mixture':GaussianMixtureRegressor(len(tr['scale']),b.parameter_dim)};draws={}
        for name,model in models.items():
            model=model.to(device);load_bound_model(source/f'{name}.pt',model,transform_path=source/'transform.pt',bank_path=source/'bank.npz',plan_path=source/'plan.json');model.eval()
            draws[name]=model.sample(context[None],plan['draws'],generator=torch.Generator(device=device).manual_seed(plan['posterior_seed']))[0].detach().cpu().numpy()
        nearest=int(np.argmin(np.mean((features-observed.cpu().numpy())**2,axis=1)));draws['nearest']=zbank[nearest:nearest+1]
        np.savez_compressed(dest/'inference-sealed.npz',truth=zbank[0],clean_prefix=features[0],noisy_prefix=observed.cpu().numpy(),**draws)
        write(dest/'inference-sealed.json',dict(sha256=sha(dest/'inference-sealed.npz'),plan_sha256=sha(output/'plan.json'),nearest_bank_index=nearest))
        rows=[];truth=None;allframes=np.arange(len(data['times']))
        for name,latents in [('truth',zbank[:1])]+list(draws.items()):
            worlds=[]
            for index,z in enumerate(latents):
                started=time.time();record=dict(method=name,draw=index,requested=len(latents))
                try:
                    with deadline(plan['rollout_deadline_s']):positions,_=b.rollout(z,allframes)
                    record.update(state='complete',diagnostics=dict(b.last_path_diagnostics))
                    if name=='truth':
                        truth=positions;recomputed=(b.observation_features(positions[frames],frames,ids));recomputed=(recomputed-recomputed[:1]).reshape(-1)[mask];record['bank_feature_reproduction_max_m']=float(np.max(np.abs(recomputed-features[0])))
                        if not np.allclose(recomputed,features[0],atol=1e-8,rtol=1e-6):raise ValueError('Frozen bank feature reproduction differs')
                    if truth is not None:record['metrics']=metrics(positions,truth)
                    worlds.append(positions)
                except (RuntimeError,ValueError,DeadlineExceeded) as exc:record.update(state='failed',error_type=type(exc).__name__,error=str(exc))
                record['elapsed_s']=time.time()-started;rows.append(record);write(dest/'progress.json',dict(records=rows));print(spec['case'],name,index,record['state'],flush=True)
            if worlds:np.savez_compressed(dest/f'{name}-trajectories.npz',positions=np.stack(worlds))
            if name=='truth' and rows[-1]['state']!='complete':truth=None;break
        summary={}
        for name in draws:
            selected=[r for r in rows if r['method']==name];valid=[r for r in selected if r['state']=='complete'];p=dest/f'{name}-trajectories.npz'
            entry=dict(requested=len(draws[name]),complete=len(valid),failed=len(selected)-len(valid),not_run=len(draws[name])-len(selected))
            if valid and truth is not None:
                with np.load(p) as f:values=f['positions']
                entry['conditional_mean_prediction']=metrics(values.mean(0),truth);entry['draw_coordinate_rmse_mm']=[r['metrics']['coordinate_rmse_mm'] for r in valid]
            summary[name]=entry
        write(dest/'result.json',dict(case=spec['case'],plan_sha256=sha(output/'plan.json'),scope=plan['scope'],records=rows,methods=summary))
    write(output/'complete.json',dict(complete=True,cases=list(CASES),plan_sha256=sha(output/'plan.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);p.add_argument('--output',default='runs/gaussian-boundary-bank-roundtrip-v01');p.add_argument('--device',default='cpu');a=p.parse_args()
    freeze(a.output) if a.command=='freeze' else run(a.output,a.device)
