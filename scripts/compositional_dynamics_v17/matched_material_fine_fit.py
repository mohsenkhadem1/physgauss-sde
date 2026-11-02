"""Predeclared finer-mesh inference from the SAME permitted calibration only.
Chosen from calibration forward-error diagnosis, before any forecast scoring.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,importlib.util,json,multiprocessing,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
BASEPLAN=REPORT/'matched-material-fit-plan.json';BASERESULT=REPORT/'matched-material-fit-results.json'
BASECOMMIT=ROOT/'runs/compositional-dynamics-v17/matched-material-fit/fitted-parameter-commit.json'
PLAN=REPORT/'matched-material-fine-fit-plan.json';RESULT=REPORT/'matched-material-fine-fit-results.json'
RAW=ROOT/'runs/compositional-dynamics-v17/matched-material-fine-fit'


def prepare():
    if PLAN.exists() or RAW.exists():raise ValueError('Preserve fine fit')
    base=json.loads(BASEPLAN.read_text());results=json.loads(BASERESULT.read_text());commit=json.loads(BASECOMMIT.read_text())
    if not results['terminal'] or not results['search_complete'] or commit['results_sha256']!=sha(BASERESULT):raise ValueError('Completed coarse search required')
    starts=[]
    for flag in [-1.,1.]:
        eligible=[r for r in results['rows'] if r['completed'] and r['task']['flag']==flag]
        if not eligible:raise ValueError('Both family starts required')
        best=min(eligible,key=lambda r:r['loss_m2']);starts.append(dict(flag=flag,parameters=best['task']['parameters'],coarse_record=best['task']['id']))
    plan=dict(base,n=3,workers=2,seconds=5400,starts=starts,refinement=dict(method='Powell',maxfev=32,xtol=.02,ftol=.001,direction_order=['scale','log_contact_stiffness','shape']),
              scope='Development finer-discretization inference on identical permitted masked calibration observations. Unknownfamily/scale/shape/contact law stiffness; no forecast read. Shared law/checkpoint unchanged. Coarse fit remains ablation; fine variant chosen before forecast scoring.',
              source_sha256=dict(base['source_sha256']),input_sha256=dict(base['input_sha256']))
    plan['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    for p in [BASEPLAN,BASERESULT,BASECOMMIT]:plan['input_sha256'][str(p.relative_to(ROOT))]=sha(p)
    # This phase has two optimizations, not a second54-candidate grid.
    plan.pop('tasks');write(PLAN,plan)


def implementation():
    spec=importlib.util.spec_from_file_location('private_fine_fit',ROOT/'scripts/compositional_dynamics_v17/matched_material_fit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.PLAN=PLAN;module.RAW=RAW;return module


def worker(arguments):
    start,deadline=arguments;module=implementation();plan=json.loads(PLAN.read_text());module.verify(plan);count=0;ids=[]
    def objective(parameters):
        nonlocal count
        if time.monotonic()>=deadline:raise TimeoutError('Fine fit deadline')
        task=dict(id=f"fine-{int(start['flag']):+d}-{count:03d}",flag=start['flag'],parameters=np.asarray(parameters).tolist());count+=1
        row=module.evaluate(task,deadline);ids.append(task['id']);return row['loss_m2']
    summary=dict(flag=start['flag'],converged=False,initial_parameters=start['parameters'])
    try:
        result=minimize(objective,start['parameters'],method='Powell',bounds=plan['bounds'],options=dict(maxfev=plan['refinement']['maxfev'],xtol=plan['refinement']['xtol'],ftol=plan['refinement']['ftol'],direc=np.eye(3)[[0,2,1]]))
        summary.update(converged=bool(result.success),message=str(result.message),evaluations=int(result.nfev),optimizer_parameters=result.x.tolist())
    except Exception as error:summary['error']=f'{type(error).__name__}: {error}'
    summary['records']=ids;write(RAW/f"refinement-{int(start['flag']):+d}.json",summary);return summary


def run():
    plan=json.loads(PLAN.read_text());implementation().verify(plan)
    if RAW.exists():raise ValueError('Preserve fine fit')
    RAW.mkdir(parents=True);start=time.monotonic();deadline=start+plan['seconds']-30;summaries=[];error=None
    pool=multiprocessing.get_context('spawn').Pool(plan['workers'])
    try:
        results=pool.imap_unordered(worker,[(s,deadline-15) for s in plan['starts']])
        for _ in plan['starts']:summaries.append(results.next(timeout=max(.01,deadline-time.monotonic())))
    except Exception as exception:error=f'{type(exception).__name__}: {exception}'
    finally:pool.terminate();pool.join()
    rows=[]
    for p in sorted(RAW.glob('*/complete.json')):
        row=json.loads(p.read_text())
        if row['plan_sha256']!=sha(PLAN):raise ValueError('Foreign fine result')
        rows.append(row)
    valid=[r for r in rows if r['completed']];best=min(valid,key=lambda r:r['loss_m2']) if valid else None
    coverage={str(int(flag)):dict(attempted=sum(r['task']['flag']==flag for r in rows),completed=sum(r['task']['flag']==flag and r['completed'] for r in rows)) for flag in [-1.,1.]}
    search_complete=len(summaries)==2 and all('error' not in s for s in summaries)
    implementation().verify(plan);report=dict(plan_sha256=sha(PLAN),terminal=True,search_complete=search_complete,stage_error=error,rows=rows,refinements=summaries,family_coverage=coverage,both_families_have_completed_candidates=all(v['completed']>0 for v in coverage.values()),best_training_candidate=best,seconds=time.monotonic()-start,forecast_read=False)
    write(RESULT,report)
    if best and search_complete:
        write(RAW/'fitted-parameter-commit.json',dict(plan_sha256=sha(PLAN),results_sha256=sha(RESULT),search_complete=True,family_coverage=coverage,both_families_have_completed_candidates=report['both_families_have_completed_candidates'],task=best['task'],loss_m2=best['loss_m2'],training_inputs_sha256=plan['input_sha256'],scope='Best completed n3 calibrationcandidate; no forecast response used; optimization convergence separately reported.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else run()
