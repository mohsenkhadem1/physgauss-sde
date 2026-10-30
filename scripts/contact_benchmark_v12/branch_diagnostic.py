"""Position-only equilibrium-branch diagnostic, not a benchmark predictor.

The same fitted energy is continued through interpolated commanded attachment
positions. Optional implicit inertia uses the existing source mass/gravity ratio;
no parameter refit, target loss, or test-error-selected branch is permitted.
"""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,check_deadline,sha256,write_json


def properties(model,x,theta,frame):
    saved=model.inertial_target;model.inertial_target=None
    try:
        E,r,H=model.evaluate(x,theta,frame,True)
        eigen=torch.linalg.eigvalsh(H)
        return dict(static_energy=float(E),static_residual=float(torch.linalg.norm(r)),
                    minimum_static_hessian_eigenvalue=float(eigen[0]),
                    negative_static_eigenvalues=int((eigen< -1e-8).sum()),min_J=model.min_jacobian(x))
    finally:model.inertial_target=saved


def continue_trajectory(payload,base,theta,start,stop,steps=1,inertia=False,field=True,contact=True,inspect_substeps=False):
    if 'evaluation' in payload:raise ValueError('No evaluator targets allowed')
    if not 1<=start<stop<=len(base) or steps<1:raise ValueError('Invalid bounded interval or step count')
    model=DynamicFEM(payload,device='cpu',field=field,contact=contact,nonconvex=True,materials=6)
    model.targets=model.targets.clone();model.active=model.active.clone()
    original_targets=model.targets.clone();original_active=model.active.clone();theta=model.t(theta)
    x=model.t((base[start-1]-model.origin)/model.scale)
    velocity=model.t((base[start-1]-base[start-2])/model.scale/(payload['times'][start-1]-payload['times'][start-2])) if start>1 else torch.zeros_like(x)
    predictions=[];rows=[];failed=[]
    for frame in range(start,stop):
        check_deadline();previous=x.clone();before=properties(model,previous,theta,frame)
        dt=float(payload['times'][frame]-payload['times'][frame-1])/steps;path=[]
        try:
            for sub in range(steps):
                fraction=(sub+1)/steps
                model.targets[frame]=(1-fraction)*original_targets[frame-1]+fraction*original_targets[frame]
                model.active[frame]=(1-fraction)*original_active[frame-1]+fraction*original_active[frame]
                model.inertial_target=x+dt*velocity if inertia else None;model.inertial_dt=dt
                old=x;x,_,diag=model.solve(theta,frame,x)
                velocity=(x-old)/dt
                path.append(dict(fraction=fraction,step_coordinate_rms_mm=float(torch.sqrt(((x-old)**2).mean()))*model.scale*1000,**diag))
                if inspect_substeps:path[-1]['static_properties']=properties(model,x,theta,frame)
            model.targets[frame]=original_targets[frame];model.active[frame]=original_active[frame]
            after=properties(model,x,theta,frame)
            displacement=(x-previous).cpu().numpy()*model.scale
            tool=(original_targets[frame]-original_targets[frame-1]).cpu().numpy()*model.scale
            rows.append(dict(frame=frame,node_mean_step_mm=float(np.linalg.norm(displacement,axis=1).mean()*1000),
                node_coordinate_step_mm=float(np.sqrt(np.mean(displacement**2))*1000),
                control_coordinate_step_mm=float(np.sqrt(np.mean(tool**2))*1000),
                centroid_displacement_mm=(displacement.mean(0)*1000).tolist(),
                previous_under_new_command=before,final=after,substeps=path))
            predictions.append(x.cpu().numpy()*model.scale+model.origin)
        except (ValueError,RuntimeError) as error:
            failed.append(dict(frame=frame,reason=str(error),completed_substeps=path));break
    return np.asarray(predictions),dict(start=start,stop=stop,steps=steps,inertia=inertia,field=field,contact=contact,frames=rows,failed=failed,
        scope='No target geometry scoring; dynamic mass inherited from fitted gravity/material ratio; not independently identified density')


def run(case,start,stop,steps,inertia,tag,validation=False,field=True,contact=True,inspect_substeps=False):
    data=load(dict(case=case),validation=validation);payload=predictor_payload(data);source=BASE/'base'/('validation' if validation else 'full')/case
    base=np.load(source/'prediction.npz')['predicted'];theta=np.load(source/'best.npz')['theta'];out=BASE/'branch-diagnostic'/tag;out.mkdir(parents=True,exist_ok=True)
    tic=time.monotonic();pred,record=continue_trajectory(payload,base,theta,start,stop,steps,inertia,field,contact,inspect_substeps)
    record.update(case=case,source=str(source.relative_to(ROOT)),checkpoint_sha256=sha256(source/'best.npz'),source_prediction_sha256=sha256(source/'prediction.npz'),elapsed_seconds=time.monotonic()-tic)
    np.savez_compressed(out/'prediction.npz',predicted=pred);write_json(out/'diagnostic.json',record)
    print(json.dumps(dict(tag=tag,case=case,steps=steps,inertia=inertia,failed=record['failed'],mean_steps=[r['node_mean_step_mm'] for r in record['frames']],elapsed=record['elapsed_seconds'])),flush=True)
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);p.add_argument('--steps',type=int,default=1);p.add_argument('--inertia',action='store_true');p.add_argument('--tag',required=True);p.add_argument('--validation',action='store_true');p.add_argument('--no-field',action='store_true');p.add_argument('--no-contact',action='store_true');p.add_argument('--inspect-substeps',action='store_true');a=p.parse_args();run(a.case,a.start,a.stop,a.steps,a.inertia,a.tag,a.validation,not a.no_field,not a.no_contact,a.inspect_substeps)
