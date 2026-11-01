"""Additive orchestration for exactly16 declared g9 refinement geometries."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from scripts.physical_refinement_v15 import render_bank as base
from scripts.physical_refinement_v15.quadrature import geometry_cases
from scripts.physical_refinement_v15.study import base_cases,development_actions
ROOT=base.ROOT;BASE=base.BASE;REPORT=base.REPORT
EXTRA_SOURCES=['scripts/physical_refinement_v15/render_extra_bank.py','scripts/physical_refinement_v15/quadrature.py']
def extras():return {p:base.sha(ROOT/p)for p in EXTRA_SOURCES}
def names_from_plan(path):
    plan=json.loads(path.read_text());names=plan['case_names'];old={c['name']for c in base_cases()};expected=[c['name']for c in geometry_cases(9)if c['name']not in old]
    if names!=expected or len(names)!=16 or old.intersection(names):raise ValueError('Exactly declared16newg9cases required; old20 outputs forbidden')
    if plan['quadrature_sha256']!=base.sha(ROOT/'scripts/physical_refinement_v15/quadrature.py'):raise ValueError('Quadrature source changed')
    return names

def worker(job,original_sources,extra_sources,deadline,plan_sha):
    if extras()!=extra_sources:raise RuntimeError('Extra renderer sources changed')
    result=base.render_job(job,original_sources,deadline)
    if extras()!=extra_sources:raise RuntimeError('Extra renderer sources changed')
    path=BASE/'render-bank'/job[0]/job[1]/'complete.json';record=json.loads(path.read_text());extension=dict(source_sha256=extra_sources,case_plan_sha256=plan_sha,scope='Declared g9 additionalgeometry only; original20 preserved')
    if 'extra_geometry_declaration'in record and record['extra_geometry_declaration']!=extension:raise ValueError('Changed extra declaration cache')
    record['extra_geometry_declaration']=extension;base.write(path,record);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case-plan',type=Path,required=True);parser.add_argument('--deadline-utc',required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--execute',action='store_true');parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Maximum4workers')
    names=names_from_plan(args.case_plan);pin=base.sha(args.case_plan);source=base.sources();extra=extras();actions=development_actions();jobs=[(name,a,i,p.tolist())for name in names for i,(a,p)in enumerate(actions.items())]
    plan=dict(case_plan=str(args.case_plan),case_plan_sha256=pin,source_sha256=source,extra_source_sha256=extra,expected_jobs=len(jobs),deadline_utc=args.deadline_utc,workers=args.workers)
    pp=REPORT/'extra-render-plan.json'
    if pp.exists()and json.loads(pp.read_text())!=plan:raise ValueError('Changed existing extra render plan')
    base.write(pp,plan)
    if not args.execute:return
    pending=list(jobs);running={};results=[]
    with ProcessPoolExecutor(max_workers=args.workers)as pool:
        while pending or running:
            base.check_sources(source)
            if extras()!=extra or base.sha(args.case_plan)!=pin:raise ValueError('Source/declaration changed')
            if base.seconds(args.deadline_utc)<=0:raise TimeoutError('Extra bank deadline reached')
            for job in list(pending):
                if len(running)>=args.workers:break
                if (BASE/'hypotheses'/job[0]/'n4'/f'pose-{job[2]:03d}'/'complete.json').exists():
                    running[pool.submit(worker,job,source,extra,args.deadline_utc,pin)]=job;pending.remove(job)
            if running:
                done,_=wait(running,timeout=min(10,max(.1,base.seconds(args.deadline_utc))),return_when=FIRST_COMPLETED)
                for f in done:
                    job=running.pop(f)
                    try:results.append(f.result())
                    except Exception as error:results.append(dict(case=job[0],action=job[1],state='failed',error=repr(error)))
            elif pending:
                if not args.wait:break
                time.sleep(min(10,max(.1,base.seconds(args.deadline_utc))))
            base.write(BASE/'extra-render-bank-status.json',dict(expected=len(jobs),completed=sum(r['state']=='complete'for r in results),failed=sum(r['state']=='failed'for r in results),pending=len(pending),running=len(running),results=results))
    print('extra render terminal',len(results),'/',len(jobs),flush=True)
if __name__=='__main__':main()
