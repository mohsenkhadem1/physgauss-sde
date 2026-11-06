"""Use newly free GPUs for independent refinements; preserve all training budgets."""
import argparse,os,signal,subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-solver-forecast-v01';PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python'
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--resume-pid',type=int,required=True);a=p.parse_args();out=BASE/a.scene;cmd=Path(f'/proc/{a.resume_pid}/cmdline').read_bytes();assert b'train_prefix.py' in cmd and a.scene.encode() in cmd;children=[];logs=[]
try:
 os.kill(a.resume_pid,signal.SIGSTOP);print('Paused waiting supervisor only; base GPU training continues',a.resume_pid,flush=True)
 while True:
  meta=json.loads((out/'native-base/run.json').read_text())
  if meta['status']=='training_complete':break
  if meta['status']=='failed':raise RuntimeError('Base training failed')
  time.sleep(15)
 for arm,gpu,weight in [('native-refined',0,0),('hybrid',1,1e-5)]:
  path=out/arm/'run.json'
  if path.exists() and json.loads(path.read_text()).get('status')=='training_complete':continue
  log=(out/(arm+'.log')).open('w');logs.append(log);args=[str(PY),'scripts/pidg_comparison/run.py','--scene',a.scene,'--gpu',str(gpu),'--data-root','runs/pidg-solver-forecast-v01/data','--steps','5000','--output',str(out/arm),'--dense-weight',str(weight),'--warm-from',str(out/'native-base'),'--flow'];children.append(subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT));print('Started',arm,'GPU',gpu,flush=True)
 for script,done in [('build_adapter','adapter.json'),('repair_projection','projection-fit.json'),('refine_appearance','appearance-fit.json'),('fit_solver','fit.json')]:
  if (out/'solver'/done).exists():continue
  print('Preparing',script,'GPU2',flush=True)
  with (out/(script+'.py.log')).open('w') as log:subprocess.run([str(PY),f'scripts/pidg_solver_forecast/{script}.py','--scene',a.scene,'--gpu','2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
finally:
 for child in children:child.wait()
 for log in logs:log.close()
 try:os.kill(a.resume_pid,signal.SIGCONT);print('Resumed original supervisor',flush=True)
 except ProcessLookupError:pass
