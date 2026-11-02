"""Common visibility-based metric on permitted calibration only, without refitting."""
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,sha,write
import json

def main():
    base=ROOT/'runs/compositional-dynamics-v17/native-phystwin-calibration-replay';public=ROOT/'runs/compositional-dynamics-v17/calibration-public';status=json.loads((base/'complete.json').read_text());plan=ROOT/'reports/compositional-dynamics-v17/native-phystwin-calibration-replay-plan.json'
    if not status['completed']or status['plan_sha256']!=sha(plan)or status['prediction_sha256']!=sha(base/'prediction.npz'):raise ValueError('Complete sealed calibration replay required')
    with np.load(base/'prediction.npz')as f:pred=f['tracked_positions_m'].astype(float);times=f['times_s']
    with np.load(public/'observations.npz')as f:obs=f['object_points'].astype(float);visible=f['object_visibilities']
    if pred.shape!=(21,98,3)or obs.shape!=pred.shape or not np.allclose(times,np.arange(21)*.01,atol=1e-12,rtol=0):raise ValueError('Same public calibration correspondence and clock')
    errors=(pred[1:]-obs[1:])[visible[1:]]
    if not np.isfinite(errors).all()or len(errors)==0:raise ValueError('Finite visible prediction errors')
    result=dict(scope='Selected native checkpoint calibration replay, not held performance. Same frames1..20/visible scalar-coordinate objective as shared-law fit, no alignment or refit.',visible_point_count=len(errors),coordinate_mse_m2=float(np.mean(errors**2)),coordinate_rmse_mm=float(1000*np.sqrt(np.mean(errors**2))),visible_vector_rms_mm=float(1000*np.sqrt(np.mean(np.sum(errors**2,axis=-1)))),sha256={str(p.relative_to(ROOT)):sha(p)for p in [plan,base/'complete.json',base/'prediction.npz',public/'observations.npz',__import__('pathlib').Path(__file__)]})
    write(ROOT/'reports/compositional-dynamics-v17/native-calibration-common-metric.json',result);print(json.dumps(result))
if __name__=='__main__':main()
