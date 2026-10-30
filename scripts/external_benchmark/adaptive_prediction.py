"""Numerical release refinement, with frozen fitted parameters and no target scoring."""
import sys,argparse,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from scripts.external_benchmark.common import *
from scripts.external_benchmark.data import softvt,predictor_payload
from scripts.external_benchmark.fem import ExternalFEM

def run(subset,task,episode,source_tag):
    check_deadline();data=softvt(subset,task,episode);src=BASE/'fits'/source_tag/data['name']
    tag='adaptive' if source_tag=='primary' else 'adaptive-no-field';out=BASE/'fits'/tag/data['name'];out.mkdir(exist_ok=True,parents=True)
    fit=json.loads((src/'fit.json').read_text());theta=np.load(src/'best.npz')['theta'];save_provenance(out,data)
    shutil.copyfile(src/'best.npz',out/'best.npz');fit['prediction_refinement']='Adaptive release step subdivision, no refitting';write_json(out/'fit.json',fit)
    model=ExternalFEM(predictor_payload(data),device='cpu',field=fit['field'],contact=fit['contact'],materials=fit['materials'])
    th=model.t(theta);x=None;velocity=None;pred=[];diagnostics=[];failed=[];subdivisions=[]
    for frame in range(len(data['times'])):
        check_deadline();release=frame>data['grasp'] and data['gripper_command'][frame]>0
        attempts=[]
        try:
            if release and x is not None:
                if velocity is None:velocity=torch.zeros_like(x)
                accepted=False
                for steps in [5,10,20,40,80,160]:
                    trial=x.clone();v=velocity.clone();dt=float(data['times'][frame]-data['times'][frame-1])/steps
                    try:
                        for _ in range(steps):
                            model.inertial_target=trial+dt*v;model.inertial_dt=dt
                            before=trial;trial,H,d=model.solve(th,frame,trial);v=(trial-before)/dt
                        x=trial;velocity=v;accepted=True;d['release_substeps']=steps
                        if steps!=5:subdivisions.append(dict(frame=frame,substeps=steps,rejected_attempts=attempts))
                        break
                    except (RuntimeError,ValueError) as error:attempts.append(dict(substeps=steps,reason=str(error)))
                if not accepted:raise RuntimeError('All adaptive release subdivisions failed: '+str(attempts))
            else:
                model.inertial_target=None;before=x;x,H,d=model.solve(th,frame,x)
                if before is not None:velocity=(x-before)/float(data['times'][frame]-data['times'][frame-1])
            points=x.cpu().numpy()*model.scale+model.origin
        except (RuntimeError,ValueError) as error:
            failed.append(dict(frame=frame,reason=str(error)));points=np.full_like(data['nodes'],np.nan);d=dict(failed=True)
        pred.append(points);diagnostics.append(d)
    np.savez_compressed(out/'prediction.npz',predicted=np.array(pred),nodes=data['nodes'],theta=theta)
    write_json(out/'prediction.json',dict(frames=len(pred),failed_frames=failed,diagnostics=diagnostics,adaptive_subdivisions=subdivisions,source_parameters=str(src/'best.npz'),source_parameters_sha256=sha256(src/'best.npz')))
    print(data['name'],tag,'failed',len(failed),'refined_frames',len(subdivisions),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subset',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--source-tag',choices=['primary','primary-no-field'],required=True);a=p.parse_args()
    run(a.subset,a.task,a.episode,a.source_tag)
