"""Explicit oracle CALIBRATION-only diagnostic, never an estimator input.
Separates frozen-law/mesh error from parameter-identification error; no forecast read.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import concurrent.futures,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.matched_material_fit import simulate,permitted_inputs,observed_loss,PLAN,verify
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
RAW=ROOT/'runs/compositional-dynamics-v17/calibration-oracle-diagnostic'
DPLAN=REPORT/'calibration-oracle-diagnostic-plan.json'


def job(n):
    torch.set_num_threads(1);plan=json.loads(PLAN.read_text());verify(plan);plan=dict(plan,n=n)
    metadata,obs,visible=permitted_inputs();oracle=json.loads(DPLAN.read_text());start=time.monotonic()
    positions,diagnostics=simulate(oracle['oracle_parameters'],oracle['oracle_family_flag'],plan,metadata,time.monotonic()+540)
    np.savez_compressed(RAW/f'n{n}.npz',positions_m=positions,times_s=diagnostics['saved_times_s'])
    result=dict(n=n,dt_s=plan['dt_s'],diagnostics=diagnostics,seconds=time.monotonic()-start,prediction_sha256=sha(RAW/f'n{n}.npz'))
    if diagnostics['completed']:
        loss,rms,count=observed_loss(positions,obs,visible);result.update(loss_m2=loss,visible_position_rms_mm=rms*1000,observed_point_count=count)
    return result


def main():
    if RAW.exists() or DPLAN.exists():raise ValueError('Preserve diagnostic')
    training=json.loads((REPORT/'calibration-development-plan.json').read_text())
    flag={'mooney_rivlin':-1.,'yeoh':1.}[training['private_teacher_family']]
    oracle=dict(scope=__doc__,oracle_parameters=[*training['private_teacher_latent'],float(np.log(training['penalty_N_m3']))],oracle_family_flag=flag,meshes=[2,3],workers=2,seconds=600,
                source_sha256=sha(__file__),private_calibration_plan_sha256=sha(REPORT/'calibration-development-plan.json'),fitting_plan_sha256=sha(PLAN),rule='This report/raw directory must not be read by parameter estimators. Labels intentionally used only for diagnostic forward evaluation on calibration motion.')
    write(DPLAN,oracle);RAW.mkdir(parents=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(job,[2,3]))
    write(REPORT/'calibration-oracle-diagnostic-results.json',dict(plan_sha256=sha(DPLAN),rows=rows,scope=oracle['scope']))
    print(json.dumps(rows))


if __name__=='__main__':main()
