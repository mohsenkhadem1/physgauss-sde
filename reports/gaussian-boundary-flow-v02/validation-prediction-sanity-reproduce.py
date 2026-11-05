from pathlib import Path
import datetime,json
import numpy as np
from scripts.gaussian_boundary_flow_v02.prediction_sanity import audit_run
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.external_benchmark.data import rigid_map

root=Path('runs/gaussian-boundary-flow-v04-validation')
recipe=json.loads(Path('reports/gaussian-boundary-flow-v02/softvt-validation/recipe.json').read_text())
rows=[];pending=[]
for case in recipe['cases']:
    folder=root/case;raw=folder/'physics';contact=folder/'learned_contact'
    if not(raw/'complete.json').exists():pending.append(case);continue
    report=audit_run(raw,contact if(contact/'complete.json').exists()else None)
    plan=json.loads((raw/'plan.json').read_text());data=load_package(plan['input_package']);ids=np.asarray(plan['point_ids']);mask=data['observed_mask'][0,ids];A=data['observation_matrix'][ids][mask];ref=data['observed_initial'][ids][mask]
    with np.load(raw/'prediction.npz')as pred:
        for method,row in report['methods'].items():
            p=ref+A@(pred[method][0]-data['nodes']);R,t=rigid_map(p,ref)
            row['initial_rigid_aligned_shape_rmse_mm']=float(np.sqrt(np.mean((p@R+t-ref)**2))*1000)
    rows.append(report)
summary={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Completed fresh-validation predictions and sealed initial/prefix packages only; no evaluation targets or raw HDF5 opened','expected_cases':len(recipe['cases']),'completed_physics_cases':len(rows),'not_yet_complete':pending,'rows':rows,'future_target_access':False,'frozen_sources_or_runs_modified':False,'new_model_selection':False}
path=Path('reports/gaussian-boundary-flow-v02/validation-prediction-sanity-snapshot.json')
if path.exists():raise FileExistsError('Snapshot immutable; choose new version')
path.write_text(json.dumps(summary,indent=2)+'\n')
for r in rows:
    print(r['case'])
    for k,v in r['methods'].items():
        print(k,'initial',v['mean_prediction']['initial_observation_coordinate_rmse_mm']['mean'],'initialshape',v['initial_rigid_aligned_shape_rmse_mm'],'penetration',v['mean_prediction']['allframe_maximum_floor_penetration_mm'],'worstsample',v.get('sample_diagnostics',{}).get('allframe_maximum_floor_penetration_mm'),'valid',v['surviving_draws']['valid'],'correction',v.get('pose_contact_correction',{}).get('coordinate_rms_mm'))
print('pending',pending)
