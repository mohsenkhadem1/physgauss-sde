"""Persisted postprocessing even if the interactive session disconnects."""
import sys,time,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import *
py=str(ROOT/'runs/phystwin-env/bin/python');count=len(json.loads((REPORT/'frozen-protocol.json').read_text())['jobs'])
while True:
    check_deadline();states=[json.loads(p.read_text()) for p in (BASE/'primary-jobs').glob('*.json')]
    if len(states)==count and all(s['state'] in ['complete','failed'] for s in states) and (BASE/'repairs-complete.json').exists() and (BASE/'adaptive-complete.json').exists() and (BASE/'physical-release-complete.json').exists():break
    time.sleep(30)
commands=[[py,'scripts/external_benchmark/aggregate.py'],[py,'scripts/external_benchmark/build_report.py'],
 [py,'scripts/external_benchmark/render.py','--dataset','phystwin','--case','double_stretch_sloth','--nodes','300','--tag','primary'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','object-soft','--task','1','--episode','0','--tag','primary'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','spatial-soft','--task','2','--episode','0','--tag','primary'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','object-soft','--task','1','--episode','0','--tag','adaptive'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','spatial-soft','--task','2','--episode','0','--tag','adaptive'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','spatial-soft','--task','1','--episode','2','--tag','adaptive','--baseline','original-fem'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','object-soft','--task','1','--episode','0','--tag','physical-release'],
 [py,'scripts/external_benchmark/render.py','--dataset','softvt','--subset','spatial-soft','--task','2','--episode','0','--tag','physical-release'],
 [py,'scripts/external_benchmark/verify_campaign.py']]
records=[]
for command in commands:
    check_deadline();result=subprocess.run(command,cwd=ROOT);records.append(dict(command=command,returncode=result.returncode));write_json(BASE/'postprocessing.json',records)
    if result.returncode and 'aggregate.py' in command[1]:raise RuntimeError('Aggregation failed')
