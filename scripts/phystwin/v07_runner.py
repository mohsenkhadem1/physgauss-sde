"""Durable v07 pipeline; explicit failure and completion markers, resumable stages."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/phystwin-cost-aware-v07';PY=ROOT/'.venv/bin/python'
def run(script,*args,python=PY):subprocess.run([str(python),str(ROOT/'scripts/phystwin'/script),*map(str,args)],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['analysis','bridge','finalize']);p.add_argument('--worker',type=int,default=0);a=p.parse_args();status=BASE/f'{a.mode}-{a.worker}-status.json'
 def mark(state,**kw):status.write_text(json.dumps(dict(state=state,time=time.time(),**kw),indent=2)+'\n')
 try:
  mark('running')
  if a.mode=='analysis':
   for i in range(3):
    while subprocess.run(['systemctl','--user','is-active','--quiet',f'phystwin-v07-ref{i}']).returncode==0:time.sleep(10)
   assert len(list((BASE/'fields').glob('*-reference.npz')))==60,'Missing reference solutions; inspect reference units'
   run('v07_study.py','observations')
   for split in ['calibration','test']:
    run('v07_study.py','plan','--split',split)
    run('v07_study.py','predict','--split',split,'--size',65)
    if split=='calibration':run('v07_study.py','seal')
   run('v07_study.py','predict','--split','test','--size',33)
  elif a.mode=='bridge':
   while not (BASE/'test-cost_aware-65.npz').exists() or (a.worker==0 and json.loads((BASE/'analysis-0-status.json').read_text())['state']!='complete'):
    path=BASE/'analysis-0-status.json'
    if path.exists() and json.loads(path.read_text())['state']=='failed':raise RuntimeError('Analysis failed')
    time.sleep(10)
   for obj in range(a.worker,6,3):
    for policy in ['repeat_initial','fixed','random','adaptive_pokes']:
     for step in [1,2]:
      path=BASE/'phystwin'/f'object-{obj}-{policy}'/f'summary-dt{step}.json'
      if path.exists():continue
      args=['--object',obj,'--budget',policy,'--step-refinement',step]
      if step==2:args+=['--evaluate-only']
      run('run_v07_phystwin.py',*args,python=ROOT/'runs/phystwin-env/bin/python')
  else:
   for name in ['analysis-0', 'bridge-0', 'bridge-1', 'bridge-2']:
    path=BASE/f'{name}-status.json'
    while not path.exists() or json.loads(path.read_text())['state']!='complete':
     if path.exists() and json.loads(path.read_text())['state']=='failed':raise RuntimeError(f'{name} failed')
     time.sleep(10)
   if not (BASE/'test-legacy-initial.npz').exists():run('v07_legacy_check.py')
   if not (ROOT/'reports/phystwin-cost-aware-v07/action-refinement.json').exists():run('v07_action_audit.py')
   if not (ROOT/'reports/phystwin-cost-aware-v07/prediction-refinement.json').exists():run('v07_prediction_audit.py')
   run('report_v07.py')
   run('plot_v07.py')
   if not (ROOT/'reports/phystwin-cost-aware-v07/actual-vs-predicted.mp4').exists():run('render_v07.py')
  mark('complete')
 except BaseException as e:mark('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
