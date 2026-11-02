"""Final common calibration-only selection across fixed native control recipes."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,sha,write
PLAN=ROOT/'reports/compositional-dynamics-v17/native-phystwin-budget-plan.json'
CASES=[('original125','native-phystwin-calibration-replay','native-phystwin-calibration-replay-plan.json'),('capacity729','native-phystwin-capacity-calibration','native-phystwin-capacity-plan.json'),('extended125','native-phystwin-budget-calibration','native-phystwin-budget-plan.json')]

def main():
    p=json.loads(PLAN.read_text())
    if p['source_sha256'][str(Path(__file__).relative_to(ROOT))]!=sha(__file__):raise ValueError('Predeclared calibration selection changed')
    observations=ROOT/'runs/compositional-dynamics-v17/calibration-public/observations.npz'
    if p['input_sha256'][str(observations.relative_to(ROOT))]!=sha(observations):raise ValueError('Common observations changed')
    with np.load(observations)as f:truth=f['object_points'].astype(float);visible=f['object_visibilities'][1:]
    rows=[];evidence={str(observations.relative_to(ROOT)):sha(observations)}
    for name,folder,planfile in CASES:
        base=ROOT/'runs/compositional-dynamics-v17'/folder;status_path=base/'complete.json';expected=ROOT/'reports/compositional-dynamics-v17'/planfile;row=dict(variant=name,complete=False)
        if not status_path.exists():
            terminal=ROOT/'runs/compositional-dynamics-v17/native-phystwin-budget-stage/complete.json'
            if name!='extended125' or not terminal.exists():raise ValueError('Wait for alldeclared calibration attempts to be terminal; no live omissions')
            stage=json.loads(terminal.read_text())
            if stage['completed'] or stage['plan_sha256']!=sha(PLAN):raise ValueError('Missing replay inconsistent with terminal stage')
            row['error']=stage.get('error','Budget stage failed before calibration replay');rows.append(row);evidence[str(terminal.relative_to(ROOT))]=sha(terminal);continue
        status=json.loads(status_path.read_text());evidence[str(status_path.relative_to(ROOT))]=sha(status_path);evidence[str(expected.relative_to(ROOT))]=sha(expected)
        if status['plan_sha256']!=sha(expected):raise ValueError('Replay recipe identity mismatch')
        if status['completed']:
            path=base/'prediction.npz'
            if sha(path)!=status['prediction_sha256']:raise ValueError('Replay prediction seal changed')
            with np.load(path)as f:pred=f['tracked_positions_m'].astype(float);times=f['times_s']
            if pred.shape!=(21,98,3)or not np.allclose(times,np.arange(21)*.01,atol=1e-12,rtol=0):raise ValueError('Common calibration clock/points')
            residual=(pred[1:]-truth[1:])[visible]
            if not np.isfinite(residual).all():raise ValueError('Finite complete replay')
            mse=float(np.mean(residual**2));row.update(complete=True,coordinate_mse_m2=mse,coordinate_rmse_mm=float(1000*np.sqrt(mse)),vector_rms_mm=float(1000*np.sqrt(np.mean(np.sum(residual**2,axis=-1)))),visible_points=len(residual));evidence[str(path.relative_to(ROOT))]=sha(path)
        else:row['error']=status.get('error','Declared calibration failure')
        rows.append(row)
    valid=[row for row in rows if row['complete']];selected=min(valid,key=lambda row:row['coordinate_mse_m2'])['variant']if valid else None
    write(ROOT/'reports/compositional-dynamics-v17/native-budget-calibration-comparison.json',dict(plan_sha256=sha(PLAN),rows=rows,selected_recipe=selected,rule=p['capacity_comparison'],tie=p['selection_tie'],scope='Final development training-budget choice from fixed-checkpoint calibration replay only, before held scoring. All three recipes retained. No more optimization-budget tuning.',evidence_sha256=evidence))
if __name__=='__main__':main()
