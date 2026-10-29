"""Resume neural fits once the complete disjoint fine bank has been generated."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.neural import train
from scripts.position_ood.neural_lowrank import train as train_lowrank
from scripts.phystwin.sequential_reference import write
while not all((BASE/f'neural-bank/extrapolated-n3/{i}.npz').exists() for i in range(512)):
    within_deadline();time.sleep(30)
for mode in ['id','all']:
    if not (BASE/f'neural/{mode}-extrapolated-n3/summary.json').exists():train(3,20000,mode,512,'extrapolated')
    if not (BASE/f'neural/lowrank-{mode}-extrapolated-n3/summary.json').exists():train_lowrank('extrapolated-n3',mode,512,20000)
selection={}
for mode in ['id','all']:
    candidates={name:json.loads((BASE/f'neural/{name}/summary.json').read_text())['best_validation_rmse_mm'] for name in [f'{mode}-extrapolated-n3',f'lowrank-{mode}-extrapolated-n3']}
    selection[mode]=dict(selected=min(candidates,key=candidates.get),validation_rmse_mm=candidates)
write(BASE/'neural-selection.json',dict(scope='Architecture selected solely on disjoint training-bank validation objects, before held-out evaluation.',selection=selection))
