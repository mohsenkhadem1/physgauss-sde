"""Seal calibration-only choices, then execute actual intervention forecasts."""
from pathlib import Path
import json,time,hashlib,subprocess,os,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/solver-driven-gaussians-v02';PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python'
paths=['rgb-refinement','bank-refinement','rgb-analytic-refinement','bank-analytic-refinement']
while not all((BASE/p/'fit.json').exists() for p in paths):time.sleep(10)
selection={}
for kind,candidates in [('learned',paths[:2]),('analytic',paths[2:])]:
 docs=[json.loads((BASE/p/'fit.json').read_text()) for p in candidates];index=max(range(len(docs)),key=lambda i:docs[i]['best']['score']);winner=docs[index]
 selection[kind]=dict(selected=candidates[index],candidates=[dict(path=p,validation_foreground_psnr=d['best']['score'],sha256=hashlib.sha256((BASE/p/'fit.json').read_bytes()).hexdigest()) for p,d in zip(candidates,docs)],criterion='Highest reserved calibration foreground PSNR; no future images or positions.')
 (BASE/('selected-'+kind+'-fit.json')).write_text(json.dumps(winner,indent=2))
(BASE/'selection.json').write_text(json.dumps(selection,indent=2));print(selection,flush=True)
def run(script,args,log):
 with (BASE/log).open('w') as f:subprocess.run([str(PY),str(ROOT/'scripts/solver_driven_gaussians'/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT)},check=True)
def branch(kind,gpu):
 fit=BASE/('selected-'+kind+'-fit.json');out=BASE/('final-'+kind)
 run('predict.py',['--fit',fit,'--output',out,'--gpu',gpu],'final-'+kind+'.log')
 run('pair.py',['--fit',fit,'--output',out,'--gpu',gpu],'final-'+kind+'-pair.log')
with concurrent.futures.ThreadPoolExecutor(2) as executor:list(executor.map(lambda item:branch(*item),[('learned',0),('analytic',2)]))
print('Selected solver forecasts complete.',flush=True)
