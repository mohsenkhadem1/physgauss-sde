"""Durable exposed-case downstream supervisor; no fresh-case discovery/access."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha

CASES=('object-soft-task2-demo0','spatial-soft-task1-demo0','spatial-soft-task2-demo0',
       'single_lift_sloth','single_lift_zebra','double_lift_zebra')
BASE=Path('runs/gaussian-boundary-flow-v03-expansion')
OUT=Path('reports/gaussian-boundary-flow-v02/expansion-downstream')


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def live_workers():
    result=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            words=[x.decode(errors='replace')for x in args]
            if 'scripts.gaussian_boundary_flow_v02.expansion_runner'in words:
                result.append({'pid':int(proc.name),'cases':[c for c in CASES if c in words]})
        except (OSError,ProcessLookupError):pass
    return result


def source_state(case,workers):
    source=BASE/case
    if(source/'complete.json').exists():
        try:load_sealed(source);return 'ready',None
        except Exception as error:return 'failed',{'reason':'source_seal_invalid','error':str(error)}
    for folder in BASE.glob('worker-*'):
        for name in ['complete.json','status.json']:
            path=folder/name
            if not path.exists():continue
            d=json.loads(path.read_text());rows=d.get('jobs',d.get('completed_jobs',[]))
            for row in rows:
                if row['case']==case and row['state']=='failed':return 'failed',row
                if row['case']==case and row['state']in ('complete','verified_completed'):
                    return 'failed',{'reason':'worker_complete_but_source_seal_absent','worker':str(folder)}
    return ('waiting',None)if any(case in w['cases']for w in workers)else('orphaned',None)


def process_case(case):
    source=BASE/case;folder=OUT/case;folder.mkdir(exist_ok=True,parents=True)
    scorer='softvt_score'if '-soft-'in case else'object_score'
    derived=Path('runs/gaussian-boundary-flow-v03-expansion-contact')/case
    retrieval=Path('runs/gaussian-boundary-flow-v03-expansion-retrieval')/case
    contact=Path('runs/gaussian-boundary-flow-v03-expansion-retrieval-contact')/case
    py='.venv/bin/python';evalpy='runs/phystwin-env/bin/python';module='scripts.gaussian_boundary_flow_v02.'
    stages=[('learned_raw_score',[evalpy,'-m',module+scorer,'--run',str(source),'--output',str(folder/'learned-raw-scores.json')],None),
            ('learned_contact',[py,'-m',module+'pose_shape_contact_campaign','--source',str(source),'--output',str(derived)],derived),
            ('learned_contact_score',[evalpy,'-m',module+scorer,'--run',str(derived),'--output',str(folder/'learned-contact-scores.json')],None),
            ('retrieval',[py,'-m',module+'bank_retrieval','--source',str(source),'--output',str(retrieval),'--top8','--device','cuda:2'],retrieval),
            ('retrieval_raw_score',[evalpy,'-m',module+scorer,'--run',str(retrieval),'--output',str(folder/'retrieval-raw-scores.json')],None),
            ('retrieval_contact',[py,'-m',module+'retrieval_contact_campaign','--source',str(retrieval),'--output',str(contact)],contact),
            ('retrieval_contact_score',[evalpy,'-m',module+scorer,'--run',str(contact),'--output',str(folder/'retrieval-contact-scores.json')],None)]
    status={'case':case,'state':'running','source_prediction_sha256':sha(source/'prediction.npz'),'stages':[]}
    prior=folder/'status.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        if old['source_prediction_sha256']!=status['source_prediction_sha256']:raise ValueError('Changed source prediction across supervisor resume')
        if old['state']in ('failed','complete'):return old
        status['stages']=old['stages']
    for name,command,run in stages:
        if any(r['stage']==name and r['state']=='complete'for r in status['stages']):continue
        write(folder/'active-stage.json',{'stage':name,'command':command,'started_unix':time.time()})
        start=time.monotonic();error=None;code=0
        try:
            if run is not None and(run/'complete.json').exists():load_sealed(run)
            else:
                with(folder/(name+'.log')).open('a')as log:
                    code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            if code:raise RuntimeError('Stage returned '+str(code))
            if run is not None:load_sealed(run)
        except Exception as exc:error=str(exc)
        row={'stage':name,'state':'failed'if error else'complete','seconds':time.monotonic()-start,'returncode':code,'error':error}
        status['stages'].append(row);write(prior,status)
        if error:
            status['state']='failed';write(prior,status);return status
    status['state']='complete';write(prior,status);return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-hours',type=float,default=3);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path('scripts/gaussian_boundary_flow_v02/bank_retrieval.py'),Path('scripts/gaussian_boundary_flow_v02/retrieval_contact_campaign.py'),Path('scripts/gaussian_boundary_flow_v02/pose_shape_contact_campaign.py')]
    plan={'cases':list(CASES),'source_root':str(BASE),'gpu':'cuda:2','max_hours':args.max_hours,'frozen_expansion_plan_sha256':sha('reports/gaussian-boundary-flow-v02/compactexpansion-plan.json'),'sources':{str(p):sha(p)for p in sources},'policy':'No reruns or case substitution after failure; no fresh cases. Source seals verified before any target scoring.'}
    path=OUT/'plan.json'
    if path.exists()and json.loads(path.read_text())!=plan:raise ValueError('Changed downstream plan')
    if not path.exists():write(path,plan)
    done={};orphan_since={};deadline=time.monotonic()+args.max_hours*3600
    while len(done)<len(CASES):
        workers=live_workers();write(OUT/'worker-monitor.json',{'unix':time.time(),'live_workers':workers,'done':list(done)})
        for case in CASES:
            if case in done:continue
            state,detail=source_state(case,workers)
            if state=='ready':
                try:done[case]=process_case(case)
                except Exception as error:done[case]={'state':'downstream_exception','error':str(error)}
                print(case,done[case]['state'],flush=True)
            elif state=='failed':done[case]={'state':'source_failed','details':detail}
            elif state=='orphaned':
                orphan_since.setdefault(case,time.monotonic())
                if time.monotonic()-orphan_since[case]>120:done[case]={'state':'source_worker_missing','details':'No matching live worker and no completed source for120seconds'}
            else:orphan_since.pop(case,None)
        if time.monotonic()>deadline:
            for case in CASES:
                if case not in done:done[case]={'state':'timeout_waiting_for_source'}
        write(OUT/'summary.json',{'cases':done,'complete_cases':len(done),'expected_cases':len(CASES),'all_terminal':len(done)==len(CASES)})
        if len(done)<len(CASES):time.sleep(15)

if __name__=='__main__':main()
