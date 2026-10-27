"""Finish unlocked reference cases after the assigned GPU worker exits."""
import argparse,fcntl,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.assist_qualified_bank import identity
from scripts.phystwin.qualified_reference import BASE,protocol,jobs,mixed_truth_job


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--split',choices=['calibration','test'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    original=identity(a.wait_pid)
    while original is not None and identity(a.wait_pid)==original:time.sleep(2)
    if a.split=='test':
        from scripts.phystwin.qualified_policy import verify_seal
        verify_seal(json.loads((BASE/'analysis/seal.json').read_text()))
    rule=json.loads((BASE/'tensor-reference-adoption.json').read_text())
    for name,r in reversed(jobs(protocol(),a.split)):
        if (BASE/'fields'/f'{name}-qualified.npz').exists():continue
        with open(BASE/'job-locks'/f'truth-{name}.lock','a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            mixed_truth_job(name,r,a.device,rule)
    print('REFERENCE_ASSIST_COMPLETE',a.split,flush=True)


if __name__=='__main__':main()
