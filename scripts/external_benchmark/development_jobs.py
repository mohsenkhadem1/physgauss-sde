"""Deadline-aware development sweep, all on declared development data only."""
from pathlib import Path
import sys,subprocess,argparse,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import ROOT,BASE,check_deadline,write_json
p=argparse.ArgumentParser();p.add_argument('--worker',choices=['native','fem','fem-cpu'],required=True);a=p.parse_args()
python=str(ROOT/'runs/phystwin-env/bin/python');jobs=[]
if a.worker=='native':
 for subset in ['object-soft','spatial-soft']:
  for initial in [10000,30000,100000]:
   tag=f'dev-native-y{initial}'
   jobs.append([python,'scripts/external_benchmark/native_softvt.py','--subset',subset,'--epochs','100','--initial',str(initial),'--tag',tag])
else:
 for subset in ['object-soft','spatial-soft']:
  for episode in ([1,2] if a.worker=='fem' else [3]):
   for materials in [1,6]:
    jobs.append([python,'scripts/external_benchmark/fit.py','--dataset','softvt','--subset',subset,'--episode',str(episode),
      '--iterations','100','--frames','36','--contact','--materials',str(materials),'--tag',f'dev-materials{materials}',
      '--device','cpu' if a.worker=='fem-cpu' else 'cuda:0'])
for i,command in enumerate(jobs):
 check_deadline();log=BASE/f'development-{a.worker}-{i}.log'
 print('START',i,command,flush=True)
 with log.open('w') as f:r=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 write_json(BASE/f'development-{a.worker}-{i}.json',dict(command=command,returncode=r.returncode,log=str(log)))
 print('DONE',i,r.returncode,flush=True)
