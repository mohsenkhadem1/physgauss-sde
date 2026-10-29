"""One-probe spatial-transfer study, with fixed and random matched controls."""
from pathlib import Path
import argparse,fcntl,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.predict_transfer import run as transfer
from scripts.position_ood.uncertainty import run as uncertainty
from scripts.position_ood.run_inverse import run as fit
from scripts.phystwin.sequential_reference import write

POLICIES=['fixed','random','adaptive'];CANDIDATES=[3,4,5,6,8];TARGET=7

def actions(i,plan,fixed=4):
    assert fixed in CANDIDATES
    return dict(fixed=fixed,random=int(np.random.default_rng(292000+i).choice(CANDIDATES)),adaptive=int(plan['selected']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='transfer_development');p.add_argument('--kind',choices=['cpu','gpu'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--iterations',type=int,default=80);p.add_argument('--fixed-action',type=int,default=4);a=p.parse_args()
    original={'transfer_development':'development','transfer_calibration':'calibration','transfer_test':'test_id'}[a.split];count={'development':3,'calibration':12,'test_id':9}[original];directory=BASE/'active-jobs';directory.mkdir(exist_ok=True)
    while True:
        within_deadline()
        if original=='test_id':
            from scripts.position_ood.seal import verify
            verify()
        pending=0;progress=False
        for i in range(a.worker,count,a.workers):
            dest=BASE/f'inverse/{a.split}-{i}';source=BASE/f'inverse/{original}-{i}';reference=BASE/f'fields/{a.split}-{i}.npz';planpath=dest/'extrapolated-n3-p3-uncertainty.json';choicespath=directory/f'{a.split}-{i}-choices.json'
            if not reference.exists() or not (source/'extrapolated-n3-p3.json').exists():pending+=1;continue
            if a.kind=='cpu':
                if not (dest/'joint-n2-p3.json').exists():transfer(a.split,i,'joint',2)
                if not (dest/'extrapolated-n3-p3.json').exists():transfer(a.split,i,'extrapolated',3)
                if not planpath.exists():uncertainty(a.split,i,'extrapolated',3)
                choice=actions(i,json.loads(planpath.read_text()),a.fixed_action);write(choicespath,dict(target=TARGET,candidates=CANDIDATES,actions=choice,scope='One acquired response per policy; fixed +40 mm y, seeded random, or prefix-only predicted target-variance reduction. The -40 mm y target remains unobserved.'))
            elif not choicespath.exists():pending+=1;continue
            choice=json.loads(choicespath.read_text())['actions']
            for action in sorted(set(choice.values())):
                assert action in CANDIDATES and action!=TARGET
                prefix=(0,1,2,action);tag=f'p4-0-1-2-{action}';coarse=dest/f'joint-n2-{tag}.json';fine=dest/f'extrapolated-n3-{tag}.json';cov=dest/f'extrapolated-n3-{tag}-uncertainty.npz'
                if cov.exists():continue
                pending+=1
                with (directory/f'{a.split}-{i}-{action}-{a.kind}.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if a.kind=='cpu':
                        if not coarse.exists():fit(a.split,i,'joint',2,a.iterations,prefix);progress=True
                        if fine.exists():uncertainty(a.split,i,'extrapolated',3,prefix);progress=True
                    elif coarse.exists() and not fine.exists():fit(a.split,i,'extrapolated',3,a.iterations,prefix);progress=True
        if pending==0:break
        if not progress:time.sleep(20)
    write(directory/f'{a.split}-{a.kind}-{a.worker}-done.json',dict(time=time.time()))
if __name__=='__main__':main()
