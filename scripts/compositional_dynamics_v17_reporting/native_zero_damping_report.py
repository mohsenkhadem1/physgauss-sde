"""Post-hoc development mechanism scores, no fresh cohort data or model selection."""
import json
import numpy as np
from scripts.compositional_dynamics_v17.prototype import ROOT,sha,write,face_grid
from scripts.compositional_dynamics_v17.forecast_metrics import compare_surface_forecast
REPORT=ROOT/'reports/compositional-dynamics-v17';BASE=ROOT/'runs/compositional-dynamics-v17'

def main():
    planpath=REPORT/'native-zero-damping-plan.json';plan=json.loads(planpath.read_text());evidence={str(planpath.relative_to(ROOT)):sha(planpath)};states={}
    for key,h in plan['source_input_sha256'].items():
        if sha(ROOT/key)!=h:raise ValueError('Diagnostic input changed')
    for child in plan['children']:
        cp=ROOT/child['path']
        if sha(cp)!=child['sha256']:raise ValueError('Child plan changed')
        folder=BASE/'native-zero-damping'/child['name'];status=json.loads((folder/'complete.json').read_text())
        if status['plan_sha256']!=sha(cp):raise ValueError('Replay identity changed')
        states[child['name']]=status
        for f in [cp,folder/'complete.json']:evidence[str(f.relative_to(ROOT))]=sha(f)
        if status['completed']:
            if sha(folder/'prediction.npz')!=status['prediction_sha256']:raise ValueError('Prediction changed')
            evidence[str((folder/'prediction.npz').relative_to(ROOT))]=sha(folder/'prediction.npz')
    public=BASE/'calibration-public';metadata=json.loads((public/'metadata.json').read_text());q=np.array(metadata['material_queries']);dims=np.array([.12,.06,.06]);rest=q*dims+[0,-.03,-.03]
    with np.load(public/'observations.npz')as f:obs=f['object_points'];visible=f['object_visibilities']
    target=BASE/'calibration-development/forecast';status=json.loads((target/'complete.json').read_text())
    if not status['completed']or sha(target/'trajectory.npz')!=status['trajectory_sha256']:raise ValueError('Development reference changed')
    dense=face_grid(17);ids=[np.flatnonzero(np.all(dense==point,axis=1))[0]for point in q]
    with np.load(target/'trajectory.npz')as f:
        sample=np.arange(0,len(f['times_s']),10);times=f['times_s'][sample];truth=f['surface_frames'][sample].reshape(len(sample),-1,3)[:,ids]
    methods={}
    for method,cal,forecast in [('primary','native-phystwin-budget-calibration','native-phystwin-budget-forecast'),('zero_drag_and_dashpot','native-zero-damping/calibration','native-zero-damping/forecast')]:
        row={}
        for label,folder in [('calibration',BASE/cal),('forecast',BASE/forecast)]:
            status=json.loads((folder/'complete.json').read_text());row[label]=dict(completed=status['completed'],error=status.get('error'))
            evidence[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json')
            if not status['completed']:continue
            if sha(folder/'prediction.npz')!=status['prediction_sha256']:raise ValueError('Prediction seal changed')
            evidence[str((folder/'prediction.npz').relative_to(ROOT))]=sha(folder/'prediction.npz')
            with np.load(folder/'prediction.npz')as f:pt=f['times_s'];x=f['tracked_positions_m'].astype(float)
            if label=='calibration':
                if x.shape!=obs.shape or not np.allclose(pt,np.arange(21)*.01,atol=1e-12,rtol=0):raise ValueError('Calibration clock/shape')
                e=(x[1:]-obs[1:])[visible[1:]];row[label].update(visible_points=len(e),coordinate_mse_m2=float(np.mean(e**2)),coordinate_rmse_mm=float(np.sqrt(np.mean(e**2))*1000))
            else:
                row[label]['geometry']=compare_surface_forecast(rest,q,dims,times,truth,pt,x)
                row[label]['precontact_80ms']=dict(surface_centroid_drop_m=float(x[0,:,2].mean()-x[8,:,2].mean()),minimum_z_m=float(x[8,:,2].min()),ballistic_drop_m=.5*9.8*.08**2,reference_surface_centroid_drop_m=float(truth[0,:,2].mean()-truth[8,:,2].mean()))
        methods[method]=row
    for f in [public/'metadata.json',public/'observations.npz',target/'complete.json',target/'trajectory.npz']:evidence[str(f.relative_to(ROOT))]=sha(f)
    write(REPORT/'native-zero-damping-results.json',dict(scope=plan['scope'],methods=methods,selected_parameters=plan['selected_parameters'],intervention=plan['intervention'],interpretation='Post-hoc mechanism diagnostic only; jointly removing two damping terms does not isolate each separately. No refit, reselection, fresh-cohort change or primary-baseline replacement. Remaining error may include spring law, discretization, contact, optimization and mass effects.',evidence_sha256=evidence,source_sha256={str(__import__('pathlib').Path(__file__).relative_to(ROOT)):sha(__file__)}))
if __name__=='__main__':main()
