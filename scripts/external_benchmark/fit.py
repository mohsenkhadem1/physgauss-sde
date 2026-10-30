"""Fit only observed prefixes. Test targets are not passed into the physics model."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from scipy.optimize import minimize
from scripts.external_benchmark.common import *
from scripts.external_benchmark.data import phystwin,softvt,predictor_payload
from scripts.external_benchmark.fem import ExternalFEM

def load_data(args):
    return phystwin(args.case,args.nodes) if args.dataset=='phystwin' else softvt(args.subset,args.task,args.episode)

def fit(data,out,device='cuda:0',iterations=30,frames=18,dev=False,field=True,contact=False,nonconvex=False,materials=1):
    check_deadline();out.mkdir(parents=True,exist_ok=True)
    save_provenance(out,data)
    payload=predictor_payload(data);m=ExternalFEM(payload,device=device,field=field,contact=contact,nonconvex=nonconvex,materials=materials)
    end=int(data['train_end']*.7) if dev else data['train_end']
    selected=np.unique(np.linspace(0,end-1,frames,dtype=int))
    A=m.t(data['observation_matrix']);y=m.t(data['observed']/m.scale);initial=m.t(data['observed_initial']/m.scale)
    mask=torch.as_tensor(data['observed_mask'],device=device)
    start=time.time();history=[];best={'value':float('inf')};failures=[]
    theta0=np.array([-4]+[2]*len(m.weights)+[5]+[0]*materials+[0],float)
    bounds=[(-9.,1.)]+[(-3.,8.)]*len(m.weights)+[(0.,9.)]+[(-2.,2.)]*materials+[(-3.,3.)]
    if not field:bounds[m.i_field]=(0.,0.)
    if contact:
        theta0=np.r_[theta0,1.];bounds.append((-5.,7.))
    def objective(values):
        check_deadline();theta=m.t(values).requires_grad_();x=None;losses=[];diagnostics=[]
        try:
            for frame in selected:
                x=m.predict(theta,int(frame),None if x is None else x.detach())
                observed=initial+A@(x-m.rest)
                valid=mask[frame]
                if valid.any():losses.append(((observed-y[frame])[valid]**2).mean())
                diagnostics.append(m.last_diagnostic)
            fitting=torch.stack(losses).mean()
            regularizer=1e-6*(theta[m.i_material:m.i_field+1]**2).sum()
            loss=fitting+regularizer;loss.backward();value=float(loss.detach());gradient=theta.grad.detach().cpu().numpy()
            if not np.isfinite(value) or not np.isfinite(gradient).all():raise RuntimeError('Nonfinite fit/gradient')
            record=dict(call=len(history),loss=value,fit_rmse_mm=float(torch.sqrt(fitting.detach()))*m.scale*1000,
                        theta=values.tolist(),elapsed_seconds=time.time()-start,max_residual=max(d['residual'] for d in diagnostics))
            history.append(record)
            if value<best['value']:
                best.update(value=value,theta=values.copy(),record=record)
                np.savez_compressed(out/'best.npz',theta=values,selected_frames=selected,nodes=m.nodes)
                write_json(out/'progress.json',dict(state='fitting',best=record,failures=failures,fit_end=end))
            print(data['name'],record['call'],round(record['fit_rmse_mm'],4),round(record['elapsed_seconds'],1),flush=True)
            return value,gradient
        except (RuntimeError,ValueError) as e:
            failures.append(dict(theta=values.tolist(),reason=str(e)))
            # A finite rejection barrier directs trial steps back toward the last valid iterate.
            center=best.get('theta',theta0);delta=values-center
            print('REJECT',str(e),flush=True)
            return 1e3+float(delta@delta),2*delta
    result=minimize(objective,theta0,jac=True,method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=iterations,maxls=12,ftol=1e-10,gtol=1e-7,maxfun=iterations*6))
    if 'theta' not in best:raise RuntimeError('No valid fitting iterate')
    summary=dict(state='fit_complete',name=data['name'],dataset=data['dataset'],fit_end=end,
        selected_frames=selected.tolist(),best=best['record'],optimizer=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit)),
        history=history,failed_trials=failures,elapsed_seconds=time.time()-start,field=field,contact=contact,nonconvex=nonconvex,materials=materials,
        interpretation='Effective gravity/material and support/material ratios; initial observed shape treated as stress-free; quasistatic tetrahedral FEM, optional geometry-pruned Delaunay meshing, implicit-Euler release dynamics; no true force or future geometry inputs')
    write_json(out/'fit.json',summary)
    return best['theta']

def predict(data,out,theta,device='cuda:0',field=True,stop=None,contact=False,nonconvex=False,materials=1):
    model=ExternalFEM(predictor_payload(data),device=device,field=field,contact=contact,nonconvex=nonconvex,materials=materials);th=model.t(theta);x=None;pred=[];diags=[];failed=[]
    end=len(data['times']) if stop is None else stop
    previous=None;velocity=None
    for frame in range(end):
        check_deadline()
        try:
            release=(data['dataset']=='SoftVTBench' and frame>data['grasp'] and data['gripper_command'][frame]>0)
            if release and x is not None:
                dt=float(data['times'][frame]-data['times'][frame-1])/5
                if velocity is None:velocity=torch.zeros_like(x)
                for substep in range(5):
                    model.inertial_target=x+dt*velocity;model.inertial_dt=dt
                    old=x;x,H,d=model.solve(th,frame,x);velocity=(x-old)/dt
                d['release_substeps']=5
            else:
                model.inertial_target=None
                old=x;x,H,d=model.solve(th,frame,x)
                if old is not None:velocity=(x-old)/float(data['times'][frame]-data['times'][frame-1])
            p=x.cpu().numpy()*model.scale+model.origin
        except (RuntimeError,ValueError) as e:
            failed.append(dict(frame=frame,reason=str(e)));p=np.full_like(data['nodes'],np.nan);d=dict(failed=True)
        pred.append(p);diags.append(d)
        if frame%10==0:print('predict',data['name'],frame,flush=True)
    np.savez_compressed(out/'prediction.npz',predicted=np.array(pred),nodes=data['nodes'],theta=theta)
    write_json(out/'prediction.json',dict(frames=end,failed_frames=failed,diagnostics=diags))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['phystwin','softvt'],required=True)
    p.add_argument('--case',default='single_lift_sloth');p.add_argument('--nodes',type=int,default=120)
    p.add_argument('--subset',default='object-soft');p.add_argument('--task',type=int,default=0);p.add_argument('--episode',type=int,default=0)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--frames',type=int,default=18);p.add_argument('--device',default='cuda:0')
    p.add_argument('--development',action='store_true');p.add_argument('--no-field',action='store_true');p.add_argument('--predict-only',action='store_true');p.add_argument('--tag',default='pilot')
    p.add_argument('--contact',action='store_true');p.add_argument('--nonconvex',action='store_true');p.add_argument('--materials',type=int,default=1)
    a=p.parse_args();data=load_data(a);out=BASE/'fits'/a.tag/data['name']
    theta=np.load(out/'best.npz')['theta'] if a.predict_only else fit(data,out,a.device,a.iterations,a.frames,a.development,not a.no_field,a.contact,a.nonconvex,a.materials)
    predict(data,out,theta,a.device,not a.no_field,stop=data['train_end'] if a.development else None,contact=a.contact,nonconvex=a.nonconvex,materials=a.materials)
