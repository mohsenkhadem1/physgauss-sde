"""Calibration-only native capacity comparison; never opens held forecast responses."""
from pathlib import Path
import json
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,sha,write
PLAN=ROOT/'reports/compositional-dynamics-v17/native-capacity-selection-plan.json'

def main():
    plan=json.loads(PLAN.read_text())
    if plan['source_sha256']!=sha(__file__):raise ValueError('Calibration reporting rule changed')
    expected_plans={}
    for particles,record in plan['calibration_plans'].items():
        if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Calibration recipe changed')
        expected_plans[int(particles)]=record['sha256']
    observed=ROOT/'runs/compositional-dynamics-v17/calibration-public/observations.npz'
    if sha(observed)!=plan['public_observations_sha256']:raise ValueError('Public observation seal changed')
    with np.load(observed)as f:truth=f['object_points'].astype(float);visible=f['object_visibilities'][1:]
    rows=[];evidence={str(observed.relative_to(ROOT)):sha(observed)}
    for particles,folder in [(125,'native-phystwin-calibration-replay'),(729,'native-phystwin-capacity-calibration')]:
        base=ROOT/'runs/compositional-dynamics-v17'/folder;status_path=base/'complete.json';row=dict(particles=particles,complete=False)
        if status_path.exists():
            status=json.loads(status_path.read_text());evidence[str(status_path.relative_to(ROOT))]=sha(status_path)
            if status['plan_sha256']!=expected_plans[particles]:raise ValueError('Replay plan identity mismatch')
            if status['completed']:
                path=base/'prediction.npz'
                if sha(path)!=status['prediction_sha256']:raise ValueError('Calibration replay changed')
                with np.load(path)as f:prediction=f['tracked_positions_m'].astype(float);times=f['times_s']
                if prediction.shape!=(21,98,3)or not np.allclose(times,np.arange(21)*.01,atol=1e-12,rtol=0):raise ValueError('Same calibration clock/correspondence')
                delta=(prediction[1:]-truth[1:])[visible]
                if not np.isfinite(delta).all():raise ValueError('Nonfinite complete calibration')
                mse=float(np.mean(delta**2));row.update(complete=True,coordinate_mse_m2=mse,coordinate_rmse_mm=float(1000*np.sqrt(mse)),vector_rms_mm=float(1000*np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),visible_points=len(delta));evidence[str(path.relative_to(ROOT))]=sha(path)
            else:row['error']=status.get('error','Replay failed')
        else:
            terminal=ROOT/'runs/compositional-dynamics-v17/native-phystwin-capacity-public/stage-complete.json'
            if not terminal.exists():raise ValueError('Capacity stage not terminal; do not choose from missing live results')
            row['error']='No completed calibration replay at capacity terminal';evidence[str(terminal.relative_to(ROOT))]=sha(terminal)
        rows.append(row)
    valid=[r for r in rows if r['complete']];selected=min(valid,key=lambda r:(r['coordinate_mse_m2'],r['particles']))['particles']if valid else None
    write(ROOT/'reports/compositional-dynamics-v17/native-capacity-calibration-comparison.json',dict(plan_sha256=sha(PLAN),rows=rows,selected_capacity=selected,rule=plan['rule'],scope='Development calibration-only capacity/configuration choice; no held outcome read. Not a proof of better held prediction.125result and all failures retained.',evidence_sha256=evidence))
if __name__=='__main__':main()
