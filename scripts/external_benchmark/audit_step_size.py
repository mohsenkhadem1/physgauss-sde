"""Development-only release integration sensitivity; never uses future target errors."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from scripts.external_benchmark.common import *
from scripts.external_benchmark.data import softvt,predictor_payload
from scripts.external_benchmark.fem import ExternalFEM
p=argparse.ArgumentParser();p.add_argument('--physical-release',action='store_true');p.add_argument('--positive-volume',action='store_true');p.add_argument('--local-grip',action='store_true');p.add_argument('--source-tag',default='calibrated-dev');p.add_argument('--output',default='development-time-step-audit');args=p.parse_args()
if args.positive_volume:
    from scripts.external_benchmark.fem_positive_volume import PositiveVolumeFEM as ExternalFEM
if args.physical_release:
    from scripts.external_benchmark.dynamic_fem import DynamicFEM as ExternalFEM
if args.local_grip:
    from scripts.external_benchmark.contact_adapter import softvt

data=softvt('spatial-soft',0,1);source=BASE/'fits'/args.source_tag/data['name'];fit=json.loads((source/'fit.json').read_text());theta=np.load(source/'best.npz')['theta']
release=int(np.flatnonzero((np.arange(len(data['times']))>data['grasp'])&(data['gripper_command']>0))[0]);assert (data['gripper_command'][release:]>0).all()
m=ExternalFEM(predictor_payload(data),device='cpu',field=fit['field'],contact=fit['contact'],materials=fit['materials']);th=m.t(theta);x=None;v=None
for frame in range(release):
    check_deadline();old=x;x,_,_=m.solve(th,frame,x)
    if old is not None:v=(x-old)/float(data['times'][frame]-data['times'][frame-1])
initial=x.clone();velocity=v.clone();predictions={};records=[]
for steps in [5,20,80,320]:
    check_deadline();x=initial.clone();v=velocity.clone();pred=[];start=time.time();failed=None
    for frame in range(release,len(data['times'])):
        dt=float(data['times'][frame]-data['times'][frame-1])/steps
        try:
            for _ in range(steps):
                m.inertial_target=x+dt*v;m.inertial_dt=dt;old=x;x,_,_=m.solve(th,frame,x);v=(x-old)/dt
            pred.append(x.cpu().numpy()*m.scale+m.origin)
        except (RuntimeError,ValueError) as error:failed=dict(frame=frame,reason=str(error));break
    if failed is None:predictions[steps]=np.array(pred)
    records.append(dict(substeps=steps,elapsed_seconds=time.time()-start,failure=failed));print(records[-1],flush=True)
if 320 in predictions:
    for row in records:
        steps=row['substeps']
        if steps in predictions:
            difference=predictions[steps]-predictions[320];row['coordinate_rmse_difference_to_320_mm']=float(np.sqrt(np.mean(difference**2))*1000)
            row['max_frame_coordinate_rmse_difference_to_320_mm']=float(np.sqrt(np.mean(difference**2,axis=(1,2))).max()*1000)
write_json(REPORT/(args.output+'.json'),dict(solver=ExternalFEM.__name__,local_grip=args.local_grip,name=data['name'],split='development',source_parameters=str(source/'best.npz'),release_frame=release,records=records,scope='Prediction-to-prediction numerical sensitivity only; no future target errors used. One development sequence, not a population accuracy claim.'))
print(json.dumps(records,indent=2))
