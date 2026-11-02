"""Prospective optimizer-only comparison on the existing two DEVELOPMENT cases.
Prepare does not run fits. Run refuses until original pilot is terminal.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK,INTERFACE,sha,write,predict
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16'
BASE=ROOT/'runs/compositional-neural-mechanics-v16';OUT=BASE/'gauss-newton-development-comparison';PLAN=R/'gauss-newton-development-comparison-plan.json'

def prepare():
    if PLAN.exists() or OUT.exists():raise ValueError('Preserve previous plan/output')
    parent=R/'full-continuous-inference-pilot-plan.json'
    artifacts=[parent]+[BASE/f'full-continuous-inference-pilot/world{i}/observations-and-development-truth.npz' for i in (0,1)]+[BULK,INTERFACE]
    sources=closure([Path(__file__)])+[ROOT/'tests/test_compositional_gauss_newton.py']
    plan=dict(scope='Optimizer-only DEVELOPMENT comparison; same two worlds/observations/models/priors/bounds as completed original pilot; no new simulation datasets or tuning',
              worlds=[0,1],models=['analytic','learned'],active=list(range(15)),initial='neutral_parameters',method='trf',x_scale='jac',max_evaluations=40,fit_seconds=900,stage_seconds=2000,workers=2,device='cpu',
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts})
    write(PLAN,plan);print(PLAN)

def verify(require_snapshot=False):
    plan=json.loads(PLAN.read_text())
    for table in ('source_sha256','artifact_sha256'):
        if any(sha(ROOT/p)!=h for p,h in plan[table].items()):raise ValueError('Prospective comparison hash mismatch')
    status=json.loads((R/'full-continuous-inference-pilot-status.json').read_text())
    if len(status['exit_codes'])!=2 or any(x!=0 for x in status['exit_codes']):raise ValueError('Original pilot must complete before comparison')
    for i in (0,1):
        prior=json.loads((R/f'full-continuous-inference-pilot-world{i}.json').read_text())
        if len(prior.get('fits',[]))!=2 or 'elapsed_seconds' not in prior:raise ValueError('Original world report incomplete')
    if require_snapshot:
        snapshot=json.loads((OUT/'runtime-snapshot.json').read_text())
        if snapshot['plan_sha256']!=sha(PLAN):raise ValueError('Runtime snapshot plan mismatch')
        for table in ('original_sha256','snapshot_sha256'):
            if any(sha(ROOT/p)!=h for p,h in snapshot[table].items()):raise ValueError('Completed original report snapshot changed')
    return plan

def worker(index):
    torch.set_num_threads(1);plan=verify(require_snapshot=True);priorpath=OUT/f'original-world{index}.json';prior=json.loads(priorpath.read_text())
    raw=np.load(BASE/f'full-continuous-inference-pilot/world{index}/observations-and-development-truth.npz')
    obs=[dict(pose=p,points=raw['points'],displacement=raw[f'obs_{j}']) for j,p in enumerate(raw['observed_poses'])]
    geometry=GeometrySpec(**prior['scene']['geometry']);records=[];path=R/f'gauss-newton-development-world{index}.json';local=OUT/f'world{index}';local.mkdir(parents=True,exist_ok=False)
    def alarm(*args):raise TimeoutError('External900s fit deadline')
    signal.signal(signal.SIGALRM,alarm)
    for learned in (False,True):
        estimator=build(geometry,prior['categories_conditioned'],obs,learned);signal.setitimer(signal.ITIMER_REAL,plan['fit_seconds'])
        try:result=GaussNewtonMAP(estimator).fit(active=plan['active'],max_evaluations=plan['max_evaluations'],seconds=plan['fit_seconds']-1)
        except Exception as error:result=dict(parameters=estimator.mean.copy(),loss=None,optimizer_success=False,incomplete=True,message=f'{type(error).__name__}: {error}',evaluations=[])
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()};row['model']='learned' if learned else 'analytic';row['heldout_errors']=[]
        # Only after optimizer has returned: inspect the already-declared development response targets.
        for j,p in enumerate(raw['held_poses']):
            try:
                signal.setitimer(signal.ITIMER_REAL,30);pred=predict(estimator,result['parameters'],p)
                row['heldout_errors'].append({key:float(np.sqrt(np.mean((value-raw[f'future_{j}_{key}'])**2))) for key,value in pred.items()})
                np.savez_compressed(local/f"forecast-{row['model']}-{j}.npz",**pred)
            except Exception as error:row['heldout_errors'].append(dict(failure=f'{type(error).__name__}: {error}'))
            finally:signal.setitimer(signal.ITIMER_REAL,0)
        row['packed_latent_error']=(result['parameters']-raw['truth']).tolist();records.append(row)
        write(path,dict(plan_sha256=sha(PLAN),original_report_sha256=sha(priorpath),runtime_snapshot_sha256=sha(OUT/'runtime-snapshot.json'),world=index,fits=records));print(index,row['model'],row['loss'],row['incomplete'],flush=True)

def run():
    plan=verify();OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic();jobs=[]
    originals={};copies={}
    for source,name in [(R/'full-continuous-inference-pilot-status.json','original-status.json')]+[(R/f'full-continuous-inference-pilot-world{i}.json',f'original-world{i}.json')for i in (0,1)]:
        raw=source.read_bytes();target=OUT/name;target.write_bytes(raw)
        originals[str(source.relative_to(ROOT))]=sha(source);copies[str(target.relative_to(ROOT))]=sha(target)
        if originals[str(source.relative_to(ROOT))]!=copies[str(target.relative_to(ROOT))]:raise ValueError('Source changed during snapshot')
    write(OUT/'runtime-snapshot.json',dict(plan_sha256=sha(PLAN),original_sha256=originals,snapshot_sha256=copies,created_unix=time.time()))
    verify(require_snapshot=True)
    for i in plan['worlds']:
        log=(OUT/f'world{i}.log').open('w');p=subprocess.Popen([sys.executable,'-m','scripts.compositional_neural_mechanics.gauss_newton_comparison','--worker',str(i)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((p,log))
    try:
        for p,_ in jobs:p.wait(timeout=max(.01,plan['stage_seconds']-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:pass
    finally:
        for p,log in jobs:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
    write(R/'gauss-newton-development-status.json',dict(plan_sha256=sha(PLAN),exit_codes=[p.returncode for p,_ in jobs],elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',type=int,choices=[0,1]);a=parser.parse_args()
    if a.prepare:prepare()
    elif a.run:run()
    else:worker(a.worker)
