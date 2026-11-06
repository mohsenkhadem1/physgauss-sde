"""Durable prefix-only PIDG workers. GPU 3 is never touched."""
import subprocess,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python';FLOWPY=ROOT/'runs/deform360-env/bin/python';BASE=ROOT/'runs/pidg-solver-forecast-v01'
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();out=BASE/a.scene;out.mkdir(exist_ok=True)
 def run(args,log,py=PY):
  with (out/log).open('w') as f:subprocess.run([str(py),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 data='runs/pidg-solver-forecast-v01/data'
 if not (BASE/'data'/a.scene/'fit320/preprocessing.json').exists():run(['scripts/pidg_comparison/preprocess.py','--data-root',data,'--scene',a.scene],'preprocess.log',FLOWPY)
 for label,steps,warm,weight in [('native-base',50000,None,0),('native-refined',5000,'native-base',0),('hybrid',5000,'native-base',1e-5)]:
  dest=out/label
  if (dest/'run.json').exists() and json.loads((dest/'run.json').read_text()).get('status')=='training_complete':continue
  args=['scripts/pidg_comparison/run.py','--scene',a.scene,'--gpu',a.gpu,'--data-root',data,'--steps',steps,'--output',dest,'--dense-weight',weight]
  if a.scene!='rubber_duck':args+=['--flow']
  if warm:args+=['--warm-from',out/warm]
  run(args,label+'.log')
 (out/'training-complete.json').write_text(json.dumps(dict(completed=True,time=time.time(),scope='Prefix-only native and fixed-weight dense hybrid; no final forecast scoring yet.')))
if __name__=='__main__':main()
