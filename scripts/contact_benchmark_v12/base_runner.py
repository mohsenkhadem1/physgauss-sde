"""Frozen v11 mechanics in isolated paths, with independently fitted prefix CV."""
import sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import BASE,check_deadline,write_json,provenance
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.external_benchmark import fit as oldfit
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scipy.optimize import minimize

def feasible(fun,x0,*args,**kwargs):
    guesses=[np.array(x0).copy()]
    for g,a in [(-6.,-1.),(-6.,-3.),(-8.,-3.)]:
        q=np.array(x0).copy();q[0]=g;q[1:3]=a;guesses.append(q)
    for k,q in enumerate(guesses):
        v,_=fun(q);print('FEASIBILITY_START',k,v,flush=True)
        if np.isfinite(v) and v<1000:return minimize(fun,q,*args,**kwargs)
    raise RuntimeError('All declared prefix-only feasible starts failed')

def rollout(data,theta,settings,device):
    model=DynamicFEM(predictor_payload(data),device=device,**settings);th=model.t(theta)
    x=None;velocity=None;pred=[];diagnostics=[];failed=[]
    for frame in range(len(data['times'])):
        check_deadline();release=data['dataset']=='SoftVTBench' and frame>data['grasp'] and data['gripper_command'][frame]>0
        try:
            if release and x is not None:
                if velocity is None:velocity=x*0
                errors=[]
                for steps in [5,10,20,40,80,160]:
                    trial=x.clone();v=velocity.clone();dt=float(data['times'][frame]-data['times'][frame-1])/steps
                    try:
                        for _ in range(steps):
                            model.inertial_target=trial+dt*v;model.inertial_dt=dt
                            before=trial;trial,_,diag=model.solve(th,frame,trial);v=(trial-before)/dt
                        x=trial;velocity=v;diag['release_substeps']=steps;break
                    except (ValueError,RuntimeError) as e:errors.append(str(e))
                else:raise RuntimeError(str(errors))
            else:
                model.inertial_target=None;before=x;x,_,diag=model.solve(th,frame,x)
                if before is not None:velocity=(x-before)/float(data['times'][frame]-data['times'][frame-1])
            points=x.cpu().numpy()*model.scale+model.origin
        except (ValueError,RuntimeError) as e:
            failed.append(dict(frame=frame,reason=str(e)));points=np.full_like(data['nodes'],np.nan);diag=dict(failed=True)
        pred.append(points);diagnostics.append(diag)
    return np.asarray(pred),dict(failed_frames=failed,diagnostics=diagnostics)

def run(spec,validation=False,device='cpu'):
    check_deadline();data=load(spec,validation);tag='validation' if validation else 'full';out=BASE/'base'/tag/data['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    oldfit.check_deadline=check_deadline;oldfit.save_provenance=provenance
    oldfit.minimize=feasible if data['dataset']=='SoftVTBench' else minimize
    settings=dict(field=True,contact=True,nonconvex=data['dataset']=='PhysTwin',materials=6 if data['dataset']=='PhysTwin' else 1)
    start=time.time()
    if (out/'best.npz').exists():theta=np.load(out/'best.npz')['theta']
    else:theta=oldfit.fit(data,out,device=device,iterations=100,frames=36,**settings)
    pred,diagnostic=rollout(data,theta,settings,device)
    np.savez_compressed(out/'prediction.npz',predicted=pred,nodes=data['nodes'],theta=theta)
    write_json(out/'prediction.json',diagnostic);write_json(out/'complete.json',dict(name=data['name'],fit_end=data['train_end'],elapsed_seconds=time.time()-start,failed_frames=len(diagnostic['failed_frames'])))
    print(data['name'],tag,'complete',len(diagnostic['failed_frames']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--subset');p.add_argument('--task',type=int);p.add_argument('--episode',type=int);p.add_argument('--validation',action='store_true');p.add_argument('--device',default='cpu');a=p.parse_args()
    spec=dict(case=a.case) if a.case else dict(subset=a.subset,task=a.task,episode=a.episode)
    run(spec,a.validation,a.device)
