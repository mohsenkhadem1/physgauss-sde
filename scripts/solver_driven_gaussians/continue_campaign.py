"""Durable sequential continuation after the visual hybrid finishes."""
from pathlib import Path
import subprocess,json,time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/solver-driven-gaussians-v02'
PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python'

def command(script,args,log):
    print(script,args,flush=True)
    with (BASE/log).open('w') as f:subprocess.run([str(PY),str(ROOT/'scripts/solver_driven_gaussians'/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,env={**__import__('os').environ,'PYTHONPATH':str(ROOT)})

while not (BASE/'rgb-hybrid/complete.json').exists():time.sleep(10)
command('project_hybrid.py',['--gpu',0],'project.log')
command('fit_rgb.py',['--gpu',0,'--method','nodes','--steps',2500,'--initialize',BASE/'hybrid-projection/observed-positions.npz'],'rgb-nodes-warm.log')
for kind,name in [('learned','rgb-fit.json'),('analytic','rgb-analytic-fit.json')]:
    command('identify.py',['--observations',BASE/'rgb-nodes/observed-positions.npz','--output',BASE/name,'--gpu',0,'--kind',kind],name+'.log')
children=[]
for fit,output,gpu in [('rgb-fit.json','rgb-refinement',0),('rgb-analytic-fit.json','rgb-analytic-refinement',1)]:
    f=(BASE/(output+'.log')).open('w')
    proc=subprocess.Popen([str(PY),str(ROOT/'scripts/solver_driven_gaussians/refine_physics.py'),'--fit',str(BASE/fit),'--output',str(BASE/output),'--gpu',str(gpu),'--steps','80'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    children.append((proc,f,output))
command('predict.py',['--fit',BASE/'rgb-fit.json','--output',BASE/'rgb-predictions','--gpu',2],'rgb-predict.log')
for proc,f,output in children:
    result=proc.wait();f.close()
    if result:raise RuntimeError(output+' failed; inspect log')
command('predict.py',['--fit',BASE/'rgb-refinement/fit.json','--output',BASE/'rgb-refined-predictions','--gpu',0],'rgb-refined-predict.log')
command('predict.py',['--fit',BASE/'rgb-analytic-refinement/fit.json','--output',BASE/'rgb-analytic-predictions','--gpu',1],'rgb-analytic-predict.log')
print('RGB fitting and forecast stages complete.',flush=True)
