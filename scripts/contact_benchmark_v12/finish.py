"""Persistent final checks after fixed cohorts and fairness supplement finish."""
import sys,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import *
PYTHON=str(ROOT/'runs/phystwin-env/bin/python')

def ready():
    jobs=[json.loads(p.read_text()) for p in (BASE/'jobs').glob('*.json')]
    if len(jobs)!=209 or any(j['state'] not in ['complete','failed'] for j in jobs):return False
    for cohort,count in [('development',21),('regression',43),('untouched_confirmation',42)]:
        root=BASE/'evaluation'/cohort
        if len(list(root.glob('*/metrics.json')))!=count:return False
        if cohort!='development' and len(list(root.glob('*/native-scene-supplement.json')))!=40:return False
    return True

def main():
    check_deadline()
    while not ready():check_deadline();time.sleep(20)
    commands=[([PYTHON,'-m','pytest','tests/test_external_fem.py']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_contact_*.py'))]+['-q'],REPORT/'numerical-tests.txt',180),
        ([PYTHON,'scripts/contact_benchmark_v12/report.py','--cohorts','development','regression','untouched_confirmation','--paper'],BASE/'final-report.log',180),
        ([PYTHON,'scripts/contact_benchmark_v12/render.py','--all'],BASE/'final-videos.log',600),
        ([PYTHON,'scripts/contact_benchmark_v12/verify.py'],BASE/'final-verification.log',300)]
    for command,log,timeout in commands:
        check_deadline()
        with open(log,'w') as output:subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    write_json(BASE/'postprocessing-complete.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Verified; final human visual review and repository publication pending'))
    print('POSTPROCESSING_COMPLETE',flush=True)

if __name__=='__main__':main()
