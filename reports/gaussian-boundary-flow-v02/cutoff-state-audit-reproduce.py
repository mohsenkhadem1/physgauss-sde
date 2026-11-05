import json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.object_score import expand_points,aligned_shape_rmse

summary=json.loads(Path('reports/gaussian-boundary-flow-v02/compactexpansion-results.json').read_text());rows=[]
for case in summary['cases']:
    root=Path(case['verified_channel_identity']['learned-raw']['run']);plan=json.loads((root/'plan.json').read_text());data=load_package(plan['input_package']);ids=np.asarray(plan['point_ids']);frames=np.asarray(plan['prefix_frames']);last=data['train_end']-1
    contactroot=Path(case['verified_channel_identity']['learned-contact']['run'])
    with np.load(root/'prediction.npz')as z, np.load(contactroot/'prediction.npz')as p:
        item={'case':case['case'],'out_of_bank_warning':case['out_of_bank_warning'],'cutoff':last,'families':{}}
        for family in ('flow','mixture'):
            pred=expand_points(z[family+'_raw'][frames],data['nodes'],data['observation_matrix'][ids],data['observed_initial'][ids]);target=data['observed'][frames][:,ids];mask=data['observed_mask'][frames][:,ids]
            def rmse(v,m):return float(np.sqrt(np.mean(v[m]**2))*1000)
            pose=expand_points(p[family+'_pose'][last],data['nodes'],data['observation_matrix'][ids],data['observed_initial'][ids]);valid=mask[-1]
            item['families'][family]={'initial_raw_rmse_mm':rmse(pred[0]-target[0],mask[0]),'cutoff_raw_rmse_mm':rmse(pred[-1]-target[-1],valid),'cutoff_pose_rmse_mm':rmse(pose-target[-1],valid),'cutoff_raw_shape_rmse_mm':aligned_shape_rmse(pred[-1,valid],target[-1,valid]),'cutoff_pose_shape_rmse_mm':aligned_shape_rmse(pose[valid],target[-1,valid]),'prefix_relative_rmse_mm':rmse((pred-pred[:1])-(target-target[:1]),mask)}
        rows.append(item)
Path('reports/gaussian-boundary-flow-v02/cutoff-state-audit.json').write_text(json.dumps({'scope':'Frozen predictions versus SAME declared prefix points/frames only; no forecast targets read or new fits','rows':rows},indent=2)+'\n')
for row in rows:print(row['case'],row['families']['flow'])
