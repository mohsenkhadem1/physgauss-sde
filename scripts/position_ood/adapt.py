"""One extra positional interaction, with a separate held target action."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE
from scripts.position_ood.run_inverse import run
from scripts.position_ood.uncertainty import run as uncertainty
from scripts.phystwin.sequential_reference import write

def adapt(split,i,policy,iterations=100):
    planpath=BASE/f'inverse/{split}-{i}/joint-n2-p3-uncertainty.json'
    if not planpath.exists():uncertainty(split,i)
    plan=json.loads(planpath.read_text())
    if policy=='adaptive':action=plan['selected']
    elif policy=='fixed':action=4
    elif policy=='random':action=int(np.random.default_rng(292000+i).choice(plan['candidates']))
    else:raise ValueError(policy)
    assert action in [4,5,6,7,8] and plan['target']==9
    write(BASE/f'plans/{split}-{i}-{policy}.json',dict(policy=policy,action=action,target=9,information='Prefix observations and local predictive sensitivities only; probe response acquired after selection.'))
    run(split,i,'joint',2,iterations,prefix=(0,1,2,action))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=1);p.add_argument('--policy',choices=['adaptive','fixed','random'],default='adaptive');a=p.parse_args();adapt(a.split,a.object,a.policy)
