"""Finish the independently audited extension once every declared case is ready."""
import sys,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,check_deadline,write_json,datetime,timezone
PYTHON=str(ROOT/'runs/phystwin-env/bin/python')

def ready():
    jobs=[json.loads(p.read_text()) for p in (BASE/'feasibility-jobs').glob('*.json')]
    if len(jobs)!=124 or any(j['state'] not in ['complete','failed'] for j in jobs):return False
    if not (BASE/'postprocessing-complete.json').exists():return False
    return all(len(list((BASE/'feasibility-evaluation'/cohort).glob('*/metrics.json')))==count
        for cohort,count in [('development',21),('regression',43),('untouched_confirmation',42),('fresh_confirmation',42)])

def main():
    check_deadline()
    while not ready():check_deadline();time.sleep(20)
    commands=[([PYTHON,'-m','pytest','tests/test_external_fem.py']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_contact_*.py'))]+['-q'],REPORT/'feasibility-numerical-tests.txt',240),
        ([PYTHON,'scripts/contact_benchmark_v12/feasibility_report.py'],BASE/'feasibility-final-report.log',180),
        ([PYTHON,'scripts/contact_benchmark_v12/render_guarded.py','--all'],BASE/'feasibility-final-videos.log',600),
        ([PYTHON,'scripts/contact_benchmark_v12/feasibility_verify.py'],BASE/'feasibility-final-verification.log',600)]
    for command,log,timeout in commands:
        check_deadline()
        with open(log,'w') as output:subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    write_json(BASE/'feasibility-postprocessing-complete.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Verified; final independent review, visual review and publication pending'))
    print('FEASIBILITY_POSTPROCESSING_COMPLETE',flush=True)

if __name__=='__main__':main()
