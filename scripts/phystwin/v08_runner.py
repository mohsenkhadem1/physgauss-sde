"""Resume-safe v08 orchestrator; independent fits share per-output locks."""
from pathlib import Path
import argparse,json,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/phystwin-informative-v08';PY=ROOT/'.venv/bin/python';PHYS=ROOT/'runs/phystwin-env/bin/python'
def run(script,*args,python=PY):subprocess.run([str(python),str(ROOT/'scripts/phystwin'/script),*map(str,args)],cwd=ROOT,check=True)
def wait_files(pattern,count):
 while len(list(BASE.glob(pattern)))<count:
  for p in [*BASE.glob('reference-*-status.json'),*BASE.glob('bank-*-status.json')]:
   if json.loads(p.read_text())['state']=='failed':raise RuntimeError(f'Failed dependency {p.name}')
  time.sleep(10)
def fit(split,obj,budget,refine=True):
 for step in ([1,2] if refine else [1]):
  path=BASE/'phystwin'/f'{split}-{obj}-{budget}'/f'summary-dt{step}.json'
  if path.exists():continue
  args=['--split',split,'--object',obj,'--budget',budget,'--step-refinement',step]
  if step==2:args+=['--evaluate-only']
  run('run_v08_phystwin.py',*args,python=PHYS)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['analysis','bridge','audit','finalize']);p.add_argument('--worker',type=int,default=0);a=p.parse_args();status=BASE/f'{a.mode}-{a.worker}-status.json'
 def mark(state,**kw):status.write_text(json.dumps(dict(state=state,time=time.time(),**kw),indent=2)+'\n')
 try:
  mark('running')
  if a.mode=='analysis':
   wait_files('fields/development-*-reference.npz',12);run('v08_study.py','observations','--split','development')
   wait_files('fields/bank*-n4-q3.npz',192);run('v08_diagnose.py')
   for split in ['calibration','test']:
    wait_files(f'fields/{split}-*-reference.npz',24);run('v08_study.py','observations','--split',split)
    run('v08_study.py','plan','--split',split);run('v08_study.py','predict','--split',split)
    if split=='calibration':run('v08_study.py','seal')
  elif a.mode=='audit':
   wait_files('test-cost_aware-65.npz',1)
   # The 97^3 audit can fill an 11 GB GPU; run after fitting releases it.
   for worker in range(3):
    path=BASE/f'bridge-{worker}-status.json'
    while not path.exists() or json.loads(path.read_text())['state']!='complete':
     if path.exists() and json.loads(path.read_text())['state']=='failed':raise RuntimeError(f'bridge-{worker} failed')
     time.sleep(10)
   run('v08_prediction_audit.py')
  elif a.mode=='bridge':
   if a.worker<3:
    wait_files('development-observations.npz',1)
    for member in range(3):fit('development',a.worker,f'member{member}',False)
    run('v08_ensemble.py','--split','development','--object',a.worker)
    for budget in ['dev_repeat',*[f'dev_action{i}' for i in range(6)]]:fit('development',a.worker,budget)
   wait_files('seal.json',1);wait_files('plans/test-adaptive_pokes.json',1)
   for obj in range(a.worker,6,3):
    for member in range(3):fit('test',obj,f'member{member}',False)
    run('v08_ensemble.py','--split','test','--object',obj)
    for policy in ['fixed','random','adaptive_pokes','ensemble']:fit('test',obj,policy)
  else:
   for name in ['analysis-0',*[f'bridge-{i}' for i in range(3)]]:
    path=BASE/f'{name}-status.json'
    while not path.exists() or json.loads(path.read_text())['state']!='complete':
     if path.exists() and json.loads(path.read_text())['state']=='failed':raise RuntimeError(f'{name} failed')
     time.sleep(10)
   run('v08_ensemble.py','--diagnose')
   if not (ROOT/'reports/phystwin-informative-v08/prediction-refinement.json').exists():run('v08_prediction_audit.py')
   run('report_v08.py');run('render_v08.py')
  mark('complete')
 except BaseException as e:mark('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
