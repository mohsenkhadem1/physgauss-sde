"""Resume-safe worker: wait for prefix training, then fit and score physics."""
import subprocess,time,json,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-solver-forecast-v01';PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();out=BASE/a.scene;report=ROOT/'reports/pidg-solver-forecast-v01'/a.scene;report.mkdir(exist_ok=True)
 while not (out/'training-complete.json').exists():
  for name in ['native-base','native-refined','hybrid']:
   meta=out/name/'run.json'
   if meta.exists() and json.loads(meta.read_text()).get('status')=='failed':raise RuntimeError(f'{name} failed; inspect log')
  time.sleep(20)
 for script,done in [('build_adapter.py','solver/adapter.json'),('repair_projection.py','solver/projection-fit.json'),('refine_appearance.py','solver/appearance-fit.json'),('fit_solver.py','solver/fit.json'),('evaluate.py','evaluation/results.json')]:
  if (out/done).exists() and (script!='fit_solver.py' or json.loads((out/done).read_text()).get('completed')):continue
  with (out/(script+'.log')).open('w') as f:subprocess.run([str(PY),str(ROOT/'scripts/pidg_solver_forecast'/script),'--scene',a.scene,'--gpu',a.gpu],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 if not (report/'motion.json').exists():
  with (out/'motion.log').open('w') as f:subprocess.run([str(ROOT/'runs/deform360-env/bin/python'),'scripts/pidg_solver_forecast/motion_metrics.py','--scene',a.scene],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 for p in (out/'solver').glob('*.json'):shutil.copy2(p,report/p.name)
 (out/'complete.json').write_text(json.dumps(dict(completed=True,finished=time.time())))
if __name__=='__main__':main()
