"""Development-only oracle forward discrepancy; no unknown-study data access.
Prediction-only scalar MSE inflation is supplementary, not calibrated coverage.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,json,signal,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS,BULK,INTERFACE,predict,sha,write
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/model-discrepancy-development';PLAN=R/'model-discrepancy-development-plan.json'
METHODS=('analytic_reference_n4','analytic_fitted','shared_laws_fitted')


def held_commands(g):
    # Exact public shapes from frozen unknown_action_campaign.held_commands.
    a=np.zeros((2,6));a[1,4]=.30;b=np.zeros((2,6));b[0,0]=-.07*g.dimensions[0];b[1,0]=.07*g.dimensions[0];b[0,3]=-.30;b[1,3]=.30
    return [a,b]


def summarize_errors(errors):
    e=np.asarray(errors,float)
    if e.shape!=(4,2,26,3) or not np.isfinite(e).all():raise ValueError('All four objects/two commands/26points/3coordinates required; no dropping')
    case_mse=np.mean(e**2,axis=(2,3));mse=float(np.mean(np.mean(case_mse,axis=1)))
    return dict(discrepancy_variance_m2=mse,coordinate_rms_m=float(np.sqrt(mse)),signed_mean_m=float(e.mean()),signed_axis_mean_m=e.mean(axis=(0,1,2)).tolist(),
                centered_coordinate_variance_m2=float(np.mean((e-e.mean())**2)),signed_coordinate_quantiles_m={str(q):float(np.quantile(e,q))for q in (0,.01,.05,.25,.5,.75,.95,.99,1)},
                case_coordinate_rms_m=np.sqrt(case_mse).tolist(),signed_per_point_mean_m=e.mean(axis=(0,1)).tolist(),
                definition='Equal four-object and equal two-command average coordinate squared error (MSE includes squared bias); no bias correction applied')


def prepare():
    if PLAN.exists() or OUT.exists():raise ValueError('Preserve previous diagnostic')
    training=R/'training-data-plan.json';raw=json.loads(training.read_text());scenes=[copy.deepcopy(x['scene'])for x in raw['scenes']if x['role']=='validation']
    if len(scenes)!=4 or [s['name']for s in scenes]!=[f'validation-{i:02d}'for i in range(4)]:raise ValueError('Exact existing four development objects required')
    for s in scenes:s['gravity']=[0.,0.,-9.81];s['density_kg_m3']=125.
    files=closure([Path(__file__)])+[ROOT/'tests/test_model_discrepancy.py',ROOT/'scripts/compositional_neural_mechanics/unknown_action_campaign.py']
    plan=dict(scope='SUPPLEMENTARY development oracle forward-discrepancy calibration only, BEFORE accessing any unknown6 truth/results',scenes=scenes,
              scene_change='Existing four validation descriptors unchanged; known gravity changed fromzero to−9.81m/s², density125kg/m³ retained, matching prospective study. No new objectdraws.',
              methods=list(METHODS),commands='public held_commands: rightbend_y.30; opposedstretch.07L+opposedtwist_x.30',material_queries=POINTS.tolist(),
              n={'analytic_reference_n4':4,'analytic_fitted':2,'shared_laws_fitted':2},parameters='Exact oracle physical descriptors for all methods; no inferredparameters or optimization',
              aggregation='Per-method equal-object/equal-command average coordinate MSE; signedbias/distribution retained; all24solves required',
              supplemental_band='sqrt(original local predictive variance + fixed method MSE); original mean/likelihood/fits/actions and zero-discrepancy primary remain unchanged',
              caveats=['Oracle forward error may not represent parameter-inference bias or extrapolation error.','Four development objects do not establish calibratedcoverage.','Scalar variance inflation ignores spatial/output correlations and signed bias.','n4 reference is a finite-resolution comparator, not exactcontinuum truth.'],
              workers=1,device='cpu',per_solve_seconds=180,stage_seconds=1200,
              source_sha256={str(p.relative_to(ROOT)):sha(p)for p in files},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in (training,BULK,INTERFACE)})
    write(PLAN,plan);print(PLAN)


def verify(plan):
    for table in ('source_sha256','input_sha256'):
        if any(sha(ROOT/p)!=h for p,h in plan[table].items()):raise ValueError('Pinned discrepancy source/input changed')


def run():
    torch.set_num_threads(1);plan=json.loads(PLAN.read_text());verify(plan);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+plan['stage_seconds'];rows=[]
    def alarm(*args):raise TimeoutError('Bounded discrepancy solve timeout')
    signal.signal(signal.SIGALRM,alarm)
    for index,scene in enumerate(plan['scenes']):
        g=GeometrySpec(**scene['geometry']);categories=dict(bulk_family=scene['bulk_family'],interface_family=int(scene['interface_latents'][0][2]),heterogeneity=scene['factors']['heterogeneity']);commands=held_commands(g)
        for method in METHODS:
            estimator=build(g,categories,[dict(pose=commands[0],points=POINTS,displacement=np.zeros_like(POINTS))],method=='shared_laws_fitted');estimator.kw['n']=plan['n'][method]
            for action,pose in enumerate(commands):
                verify(plan);row=dict(world=index,name=scene['name'],method=method,action=action,pose=pose.tolist(),plan_sha256=sha(PLAN));t=time.monotonic();path=OUT/f'{index}-{method}-{action}'
                try:
                    if deadline-time.monotonic()<=0:raise TimeoutError('Stage exhausted')
                    signal.setitimer(signal.ITIMER_REAL,min(plan['per_solve_seconds'],deadline-time.monotonic()));values=predict(estimator,pack(scene),pose)
                    if any(not np.isfinite(v).all()for v in values.values()):raise ValueError('Nonfinite oracle output')
                    np.savez_compressed(path.with_suffix('.npz'),**values);row.update(completed=True,response_sha256=sha(path.with_suffix('.npz')))
                except Exception as error:row.update(completed=False,error=f'{type(error).__name__}: {error}')
                finally:signal.setitimer(signal.ITIMER_REAL,0)
                row['seconds']=time.monotonic()-t;write(path.with_suffix('.json'),row);rows.append(row)
                write(R/'model-discrepancy-development-progress.json',dict(plan_sha256=sha(PLAN),rows=rows));print(index,method,action,row['completed'],flush=True)
    report=dict(plan_sha256=sha(PLAN),rows=rows,all24_complete=len(rows)==24 and all(x['completed']for x in rows),calibration={},elapsed_seconds=time.monotonic()-start,
                no_unknown_truth_access='This source has no reader for unknown-action-reserved or its metrics; only public command code is source-pinned.',scope=plan['scope'],caveats=plan['caveats'])
    if report['all24_complete']:
        for method in METHODS[1:]:
            errors=np.stack([np.stack([np.load(OUT/f'{i}-{method}-{j}.npz')['displacement']-np.load(OUT/f'{i}-analytic_reference_n4-{j}.npz')['displacement']for j in (0,1)])for i in range(4)])
            np.savez_compressed(OUT/f'{method}-signed-errors.npz',errors=errors)
            report['calibration'][method]=summarize_errors(errors);report['calibration'][method]['error_tensor_sha256']=sha(OUT/f'{method}-signed-errors.npz')
    verify(plan);write(R/'model-discrepancy-development.json',report)
    # This artifact fixes variance before any later supplementary scorer is authorized.
    write(R/'model-discrepancy-variance-freeze.json',dict(report_sha256=sha(R/'model-discrepancy-development.json'),plan_sha256=sha(PLAN),created_unix=time.time(),available=report['all24_complete'],
          method_variance_m2={k:v['discrepancy_variance_m2']for k,v in report['calibration'].items()},scope='Prediction-only supplementary variance; primary zero-discrepancy outputs unchanged; no coverage claim'))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();prepare()if a.prepare else run()
