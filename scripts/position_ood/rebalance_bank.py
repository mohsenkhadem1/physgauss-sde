from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.neural import bank
from scripts.position_ood.study import BASE
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args();indices=json.loads((BASE/'bank-rebalance.json').read_text())['indices'];bank(mesh=3,forward='extrapolated',indices=indices)
