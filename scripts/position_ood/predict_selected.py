"""Apply a neural architecture selected on training-bank validation only."""
from pathlib import Path
import argparse,json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE
from scripts.phystwin.sequential_reference import write

def run(split,i,mode):
    selection=json.loads((BASE/'neural-selection.json').read_text())['selection'][mode]['selected']
    if selection.startswith('lowrank-'):
        from scripts.position_ood.neural_lowrank import predict
        source=predict(split,i,'extrapolated-n3',mode)
    else:
        from scripts.position_ood.neural import predict
        source=predict(split,i,mode,3,'extrapolated')
    destination=BASE/f'neural-predictions/{split}-{i}-selected-{mode}.npz';shutil.copy2(source,destination);write(destination.with_suffix('.json'),dict(selected=selection,selection_scope='Disjoint training-bank validation objects only',source=str(source.relative_to(BASE))));return destination
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--object',type=int,required=True);p.add_argument('--mode',choices=['id','all'],required=True);a=p.parse_args();run(a.split,a.object,a.mode)
