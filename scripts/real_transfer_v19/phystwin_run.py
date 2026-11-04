"""Prefix-only fitting and frozen forecasts for the real PhysTwin release."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.external_benchmark.data import phystwin, predictor_payload
from scripts.external_benchmark.fem import ExternalFEM
from scripts.real_transfer_v19.neural_fem import NeuralFEM, parameters, BULK, ROOT

BASE=ROOT/'runs/real-transfer-v19'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending.json');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def export(case,nodes=120):
    destination=BASE/'inputs'/f'{case}-{nodes}'
    if destination.exists():
        meta=json.loads((destination/'manifest.json').read_text())
        if sha(destination/'prefix.npz')!=meta['prefix_sha256']:
            raise ValueError('Modified prefix')
        return destination
    data=phystwin(case,nodes);payload=predictor_payload(data)
    destination.mkdir(parents=True)
    np.savez_compressed(destination/'prefix.npz',**{k:v for k,v in payload.items() if isinstance(v,np.ndarray)})
    sources=[ROOT/'scripts/external_benchmark/data.py',Path(data['source']),Path(data['source']).parent/'split.json']
    write(destination/'manifest.json',dict(name=case,dataset='PhysTwin',train_end=int(data['train_end']),
         floor=float(data['floor']),prefix_sha256=sha(destination/'prefix.npz'),
         source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
         evaluation_geometry_exported=False,force_labels_exported=False,
         historical_exposure='Eight plush sequences were previously evaluated; this is an exploratory rebenchmark.'))
    return destination

def load(case,nodes,development):
    folder=export(case,nodes);meta=json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'prefix.npz',allow_pickle=False) as f:
        data={k:f[k].copy() for k in f.files}
    data.update(name=case,dataset='PhysTwin',train_end=meta['train_end'],floor=meta['floor'])
    if development:
        full=int(data['train_end']);cut=max(3,int(.7*full));data['train_end']=cut
        for key in ('observed','observed_mask'):data[key]=data[key][:cut]
        for key in ('times','controls','controller_targets','attachment_active'):data[key]=data[key][:full]
    if len(data['observed'])!=data['train_end'] or 'evaluation' in data:
        raise ValueError('Strict prefix violation')
    return data,meta

def run(args):
    torch.set_num_threads(1);out=BASE/'fits'/args.label/args.case/args.method
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    data,meta=load(args.case,args.nodes,args.development)
    learned=args.method=='neural'
    m=NeuralFEM(data,args.device) if learned else ExternalFEM(data,args.device,field=False,nonconvex=True,materials=6)
    theta0,bounds=parameters(m,learned);selected=np.unique(np.linspace(0,data['train_end']-1,args.frames,dtype=int))
    A=m.t(data['observation_matrix']);observed=m.t(data['observed']/m.scale)
    initial=m.t(data['observed_initial']/m.scale);mask=torch.as_tensor(data['observed_mask'],device=args.device)
    history=[];failures=[];best={};fit_deadline=start+args.fit_seconds
    sources=[Path(__file__),ROOT/'scripts/real_transfer_v19/neural_fem.py',ROOT/'scripts/external_benchmark/fem.py',BULK,
             ROOT/'scripts/compositional_neural_mechanics/materials.py',ROOT/'scripts/compositional_neural_mechanics/scale_separated_material.py']
    plan=dict(args=vars(args),input=meta,selected_frames=selected.tolist(),bounds=bounds,
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
              scope='Arbitrary tetrahedral adapter of frozen bulk decoder; quadratic explicit attachments, gravity and floor; quasistatic; no rendering claim.',
              frozen_decoder=True,held_targets_read=False)
    write(out/'plan.json',plan)
    def objective(values):
        if time.monotonic()>fit_deadline:raise TimeoutError('Fitting budget exhausted')
        theta=m.t(values).requires_grad_();x=None;losses=[]
        try:
            for frame in selected:
                if time.monotonic()>fit_deadline:raise TimeoutError('Fitting budget exhausted')
                x=m.predict(theta,int(frame),None if x is None else x.detach())
                prediction=initial+A@(x-m.rest);valid=mask[frame]
                if valid.any():losses.append(((prediction-observed[frame])[valid]**2).mean())
            loss=torch.stack(losses).mean()+1e-6*(theta[m.i_material:m.i_field]**2).sum()
            loss.backward();value=float(loss.detach());grad=theta.grad.detach().cpu().numpy()
            if not np.isfinite(value) or not np.isfinite(grad).all():raise ValueError('Nonfinite fit')
            row=dict(call=len(history),loss=value,coordinate_rmse_mm=np.sqrt(value)*m.scale*1000,
                     elapsed_s=time.monotonic()-start,parameters=values.tolist())
            history.append(row)
            if not best or value<best['loss']:
                best.update(row);np.savez_compressed(out/'best.npz',theta=values,nodes=data['nodes'])
            write(out/'progress.json',dict(state='fitting',best=best,calls=len(history),failed_trials=len(failures)))
            print(json.dumps(row),flush=True)
            return value,grad
        except (RuntimeError,ValueError) as error:
            failures.append(dict(parameters=values.tolist(),reason=str(error)))
            center=np.asarray(best.get('parameters',theta0));delta=values-center
            print('REJECT',error,flush=True)
            return 1e3+float(delta@delta),2*delta
    optimizer={}
    try:
        result=minimize(objective,theta0,jac=True,method='L-BFGS-B',bounds=bounds,
                        options=dict(maxiter=args.iterations,maxfun=args.iterations*5,maxls=10,ftol=1e-9,gtol=1e-7))
        optimizer=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
    except TimeoutError as error:optimizer=dict(success=False,message=str(error))
    write(out/'fit.json',dict(best=best,history=history,failures=failures,optimizer=optimizer,elapsed_s=time.monotonic()-start))
    if not best:raise RuntimeError('No valid fitted checkpoint')
    # The fitting checkpoint is sealed before forecasting; only commands enter below.
    theta=m.t(best['parameters']);pred=[];diagnostics=[];x=None
    for frame in range(len(data['times'])):
        try:
            x,_,d=m.solve(theta,frame,x);p=x.detach().cpu().numpy()*m.scale+m.origin
        except (RuntimeError,ValueError) as error:
            p=np.full_like(data['nodes'],np.nan);d=dict(failed=True,reason=str(error))
        pred.append(p);diagnostics.append(d)
        if frame%10==0:
            write(out/'progress.json',dict(state='forecasting',frame=frame,total=len(data['times']),best=best))
            print('forecast',frame,flush=True)
    np.savez_compressed(out/'prediction.npz',predicted=np.asarray(pred),nodes=data['nodes'],theta=np.asarray(best['parameters']))
    write(out/'complete.json',dict(completed=True,fit_end=int(data['train_end']),frames=len(pred),
          failed_frames=[i for i,d in enumerate(diagnostics) if d.get('failed')],diagnostics=diagnostics,
          elapsed_s=time.monotonic()-start,prediction_sha256=sha(out/'prediction.npz'),fit_sha256=sha(out/'fit.json'),
          plan_sha256=sha(out/'plan.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--label',required=True)
    p.add_argument('--method',choices=['neural','analytic'],default='neural');p.add_argument('--device',default='cuda:0')
    p.add_argument('--nodes',type=int,default=120);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--fit-seconds',type=float,default=900)
    p.add_argument('--development',action='store_true');p.add_argument('--total-seconds',type=float,default=2400)
    args=p.parse_args()
    def alarm(*_):raise TimeoutError('Job wall-clock limit')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,args.total_seconds)
    try:run(args)
    except Exception as error:
        write(BASE/'fits'/args.label/args.case/args.method/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
