"""Finish the fixed exploratory cohort once all predictions are committed."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,check_deadline,write_json,datetime,timezone
PYTHON=str(ROOT/'runs/phystwin-env/bin/python')

def ready():
    jobs=[json.loads(p.read_text()) for p in (BASE/'model-selection-jobs').glob('*.json')]
    return len(jobs)==296 and all(j['state'] in ['complete','failed'] for j in jobs) and all(
        len(list((BASE/'model-selection-evaluation'/cohort).glob('*/metrics.json')))==n
        for cohort,n in [('development',21),('regression',43),('untouched_confirmation',42),('fresh_confirmation',42)])

def main():
    while not ready():check_deadline();time.sleep(20)
    commands=[([PYTHON,'-m','pytest','tests/test_model_selection_v12.py','tests/test_contact_fieldfree_runner.py','-q'],REPORT/'model-selection-tests.txt',240),
              ([PYTHON,'scripts/contact_benchmark_v12/model_selection_report.py'],BASE/'model-selection-report.log',180),
              ([PYTHON,'scripts/contact_benchmark_v12/model_selection_verify.py'],BASE/'model-selection-verification.log',900)]
    for command,log,timeout in commands:
        check_deadline()
        with open(log,'w') as output:subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    write_json(BASE/'model-selection-postprocessing-complete.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='All148 predictions and scores audited; independent review, videos and publication pending'))

if __name__=='__main__':main()
