"""Bounded, resumable three-GPU queue with explicit job failures."""
import concurrent.futures,json,os,subprocess,threading,time
from pathlib import Path
from scripts.real_transfer_v19.phystwin_run import ROOT,BASE,write,sha,export

CASES=['single_lift_sloth','single_lift_dinosor','single_push_sloth','double_lift_sloth',
       'double_stretch_sloth','single_lift_zebra','double_lift_zebra','double_stretch_zebra']
LABEL='exploratory-01'

def main():
    report=ROOT/'reports/real-transfer-v19';queue=BASE/'queue'/LABEL;queue.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'scripts/real_transfer_v19').glob('*.py'))
    plan=dict(cases=CASES,methods=['neural','analytic'],nodes=120,fit_frames=18,iterations=50,
              fit_seconds=600,total_seconds=1800,devices=['cuda:0','cuda:1','cuda:2'],
              prior_development='single_lift_sloth, first 70 percent of official training prefix',
              split='Official train/test suffixes; all eight cases historically exposed, exploratory not fresh test.',
              model_selection='Single fixed neural configuration; no per-case future-error selection.',
              baseline_scope='Neural constitutive-law adapter vs matched analytic FEM plus official released PhysTwin, persistence and prefix-fit ridge.',
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    planpath=report/(LABEL+'-protocol.json')
    if planpath.exists():
        if json.loads(planpath.read_text())!=plan:raise ValueError('Campaign source/protocol changed')
    else:write(planpath,plan)
    for case in CASES:
        export(case,120)
        for method in plan['methods']:
            path=queue/(case+'-'+method+'.json')
            if not path.exists():write(path,dict(case=case,method=method,state='pending'))
    lock=threading.Lock()
    def worker(device):
        while True:
            with lock:
                jobs=[(p,json.loads(p.read_text())) for p in sorted(queue.glob('*.json'))]
                pending=[(p,j) for p,j in jobs if j['state']=='pending']
                if not pending:return
                path,job=pending[0];job.update(state='running',device=device,started=time.time());write(path,job)
            command=[str(ROOT/'runs/phystwin-env/bin/python'),'-m','scripts.real_transfer_v19.phystwin_run',
                     '--case',job['case'],'--method',job['method'],'--device',device,'--label',LABEL,
                     '--nodes','120','--frames','18','--iterations','50','--fit-seconds','600','--total-seconds','1800']
            with path.with_suffix('.log').open('w') as log:
                try:code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1810).returncode
                except subprocess.TimeoutExpired:code=124
            job.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());write(path,job)
            print(device,job['case'],job['method'],job['state'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(worker,device) for device in plan['devices']]:future.result()
    write(report/(LABEL+'-jobs.json'),[json.loads(p.read_text()) for p in sorted(queue.glob('*.json'))])

if __name__=='__main__':main()
