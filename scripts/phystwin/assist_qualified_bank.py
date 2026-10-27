"""Use an idle bank worker's GPU slot to finish unlocked remaining cases."""
import argparse,fcntl,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.qualified_reference import BASE,protocol,jobs,_bank_job


def identity(pid):
    try:fields=Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()
    except FileNotFoundError:return None
    return None if fields[0]=='Z' else fields[19]


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--mesh',type=int,choices=[8,12],default=12);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    original=identity(a.wait_pid)
    while original is not None and identity(a.wait_pid)==original:time.sleep(2)
    spec=protocol();pending=jobs(spec,'bank')+jobs(spec,'development')
    for name,r in reversed(pending):
        if (BASE/'banks'/f'n{a.mesh}'/f'{name}.npz').exists():continue
        with open(BASE/'job-locks'/f'bank-n{a.mesh}-{name}.lock','a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            _bank_job(name,r,a.device,a.mesh)
    print('BANK_ASSIST_COMPLETE',flush=True)


if __name__=='__main__':main()
