"""Read targets only after all declared development variant predictions commit."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
from scripts.compositional_dynamics_v17.forecast_metrics import compare_surface_forecast
from scripts.compositional_dynamics_v17.surface_physics_metrics import diagnostics,compare_diagnostics
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.compositional_dynamics_v17.render_metrics import compare_renderings
from scripts.physical_refinement_v15.observations import Camera,box_mesh
PUBLIC=ROOT/'runs/compositional-dynamics-v17/calibration-public/metadata.json'
OURS=ROOT/'runs/compositional-dynamics-v17/matched-material-forecast'
NATIVE=ROOT/'runs/compositional-dynamics-v17/native-phystwin-graph-recovery'
TRUTH=ROOT/'runs/compositional-dynamics-v17/calibration-development/forecast'
OUT=REPORT/'matched-forecast-development-results.json'


def committed_predictions():
    base=ROOT/'runs/compositional-dynamics-v17'
    specs=[('ours_coarse',base/'matched-material-forecast','prediction-commit.json','surface-prediction.npz','surface_prediction_sha256','positions_m'),
           ('ours_fine',base/'matched-material-fine-forecast','prediction-commit.json','surface-prediction.npz','surface_prediction_sha256','positions_m'),
           ('original125',NATIVE,'predict-complete.json','forecast.npz','forecast_sha256','tracked_positions_m'),
           ('capacity729',base/'native-phystwin-capacity-forecast','complete.json','prediction.npz','prediction_sha256','tracked_positions_m'),
           ('extended125',base/'native-phystwin-budget-forecast','complete.json','prediction.npz','prediction_sha256','tracked_positions_m')]
    plans={
        'ours_coarse':'matched-material-forecast-plan.json',
        'ours_fine':'matched-material-fine-forecast-plan.json',
        'original125':'native-phystwin-fit-plan.json',
        'capacity729':'native-phystwin-capacity-plan.json',
        'extended125':'native-phystwin-budget-plan.json'}
    records=[];evidence=[]
    # Require all terminal prediction records before loading ANY prediction or target.
    for name,folder,status_file,data_file,digest_key,position_key in specs:
        status=folder/status_file
        if not status.exists() and name=='extended125':
            terminal=base/'native-phystwin-budget-stage/complete.json'
            stage=json.loads(terminal.read_text())
            if stage['completed'] or stage['plan_sha256']!=sha(REPORT/plans[name]):raise ValueError('Budget stage identity/completion inconsistent with missing forecast')
            record=dict(completed=False,plan_sha256=stage['plan_sha256'],error=stage.get('error','Budget failed before forecast'))
            evidence.append(terminal)
        else:
            record=json.loads(status.read_text());evidence.append(status)
        expected_plan=REPORT/plans[name];evidence.append(expected_plan)
        if record.get('plan_sha256')!=sha(expected_plan):raise ValueError(f'Foreign forecast plan: {name}')
        if not isinstance(record.get('completed'),bool):raise ValueError(f'Explicit terminal status required: {name}')
        path=folder/data_file
        if record['completed']:
            if sha(path)!=record[digest_key]:raise ValueError(f'Prediction integrity mismatch: {name}')
            evidence.append(path)
        records.append((name,record,path,position_key))
    recovery_path=NATIVE/'recovery-complete.json';recovery=json.loads(recovery_path.read_text());native=records[2][1]
    if recovery['completed']!=native['completed'] or (native['completed'] and (recovery['prediction_commit_sha256']!=sha(NATIVE/'predict-complete.json') or recovery['forecast_sha256']!=sha(NATIVE/'forecast.npz'))):raise ValueError('Native graph recovery seal mismatch')
    recovery_plan=REPORT/'native-phystwin-graph-recovery-plan.json'
    if recovery['recovery_plan_sha256']!=sha(recovery_plan):raise ValueError('Foreign graph recovery')
    evidence.extend([recovery_path,recovery_plan])
    selection_path=REPORT/'native-budget-calibration-comparison.json';selection=json.loads(selection_path.read_text());evidence.append(selection_path)
    budget_plan=REPORT/'native-phystwin-budget-plan.json';budget=json.loads(budget_plan.read_text())
    if selection['plan_sha256']!=sha(budget_plan):raise ValueError('Foreign native selection')
    selection_source='scripts/compositional_dynamics_v17/native_budget_report.py'
    if sha(ROOT/selection_source)!=budget['source_sha256'][selection_source]:raise ValueError('Native selection source changed')
    if sha(PUBLIC)!=budget['input_sha256'][str(PUBLIC.relative_to(ROOT))]:raise ValueError('Public metadata changed')
    for name,digest in selection['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Calibration selection evidence changed')
    if selection['selected_recipe'] not in {'original125','capacity729','extended125'}:raise ValueError('Native recipe selection required before target access')
    return records,evidence,selection['selected_recipe']


def main():
    if OUT.exists():raise ValueError('Preserve previous evaluation')
    records,evidence,native_primary=committed_predictions()  # BEFORE private target access.
    public=json.loads(PUBLIC.read_text());q=np.asarray(public['material_queries']);dims=np.array([.12,.06,.06]);rest=q*dims+[0,-.03,-.03]
    target_record=json.loads((TRUTH/'complete.json').read_text())
    if not target_record['completed'] or target_record['trajectory_sha256']!=sha(TRUTH/'trajectory.npz'):raise ValueError('Complete pinned reference required for scoring')
    dense=face_grid(17);point_indices=[np.flatnonzero(np.all(dense==point,axis=1))[0] for point in q]
    with np.load(TRUTH/'trajectory.npz') as a:
        ids=np.arange(0,len(a['times_s']),10);times=a['times_s'][ids];positions=a['surface_frames'][ids].reshape(len(ids),-1,3)[:,point_indices]
    reference_physics=diagnostics(times,positions,q,dims,public['forecast_public_setup']['floor_z_m'])
    methods={}
    for name,record,path,key in records:
        if record['completed']:
            with np.load(path) as a:pt=a['times_s'].copy();pred=a[key].copy()
        else:pt=np.empty(0);pred=np.empty((0,98,3))
        methods[name]=dict(record_completed=record['completed'],geometry=compare_surface_forecast(rest,q,dims,times,positions,pt,pred),render_rows=[])
        methods[name]['physical_diagnostics']=compare_diagnostics(reference_physics,diagnostics(pt,pred,q,dims,public['forecast_public_setup']['floor_z_m']))
        methods[name]['prediction_positions']=pred
    cameras=[Camera(tuple(c['eye']),tuple(c['target']),c['width'],c['height'],c['focal']) for c in public['cameras'][:2]]+[Camera((.20,-.27,.065),(.06,0,-.025),256,192,300.)]
    fq=face_grid(5);indices=[np.flatnonzero(np.all(q==point,axis=1))[0] for point in fq]
    floor=box_mesh([0,0,-.071],[.45,.4,.002],[.75,.77,.79])
    for i,time_s in enumerate(np.arange(len(times))*.01):
        for camera_id,camera in enumerate(cameras):
            ref=render_frame(positions[i,indices].reshape(6,5,5,3),camera,[floor],time_s=float(time_s))
            for name,result in methods.items():
                pred=result['prediction_positions']
                if i>=len(pred):continue
                image=render_frame(pred[i,indices].reshape(6,5,5,3),camera,[floor],time_s=float(time_s))
                result['render_rows'].append(dict(frame=i,time_s=float(time_s),camera=camera_id,role='new_camera' if camera_id==2 else 'observed_camera_future_time',metrics=compare_renderings(image,ref)))
    fields=['psnr_foreground_union_db','ssim_rgb_foreground_union','silhouette_iou','depth_joint_rmse_m','psnr_global_db','ssim_rgb_global']
    for result in methods.values():
        del result['prediction_positions'];result['render_summary']={}
        for field in fields:
            values=[r['metrics'][field] for r in result['render_rows'] if r['metrics'][field] is not None]
            result['render_summary'][field]=dict(mean=float(np.mean(values)) if values else None,available=len(values),expected=len(times)*len(cameras))
    sources=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/forecast_metrics.py',ROOT/'scripts/compositional_dynamics_v17/surface_physics_metrics.py',ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py',ROOT/'scripts/compositional_dynamics_v17/render_sequence.py',ROOT/'scripts/compositional_dynamics_v17/render_metrics.py',ROOT/'scripts/physical_refinement_v15/observations.py']
    inputs=[PUBLIC,*evidence,TRUTH/'trajectory.npz',TRUTH/'complete.json']
    report=dict(scope='ONE development object; masked-position inferred physics followed by new dropheight. Analytic teacher n3 is not spatially qualified reference. No population/SOTA/real-world accuracy claim.',
                primary_methods=dict(ours='ours_fine',native_phystwin=native_primary),selection_scope='Fine ours chosen from calibration forward diagnosis; native variant chosen only by common calibration replay. All declared variants retained.',methods=methods,expected_frames=len(times),expected_cameras=len(cameras),cameras=[vars(c) for c in cameras],
                rendering_scope='Common known procedural appearance and98point surface mesh isolate motion error; these are NOT native learned-appearance quality scores or rendering-parity proof.',
                aggregation='All recorded frames including initialcondition, uniform per-frame/camera mean image scores; geometry rest-area quadrature. No bootstrapCI from one object.',
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    write(OUT,report);print(json.dumps({name:result['geometry'].get('world_point_rms_m') for name,result in methods.items()}))


if __name__=='__main__':main()
