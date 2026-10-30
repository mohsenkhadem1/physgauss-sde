"""Durable campaign configuration, provenance and bounded process launch."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/coupled-contact-v13'
REPORT=ROOT/'reports/coupled-contact-v13'
CONFIG=ROOT/'configs/coupled_contact_v13.json'
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def remaining():
    return datetime.fromisoformat(json.loads(CONFIG.read_text())['deadline_utc']).timestamp()-time.time()
def launch(name,command):
    if remaining()<=0:raise RuntimeError('Campaign deadline reached')
    unit='coupled-contact-v13-'+name
    subprocess.run(['systemd-run','--user','--unit',unit,'--collect','--property=RuntimeMaxSec='+str(max(1,int(remaining()))),'--working-directory='+str(ROOT),'--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',*command],check=True)
    write(BASE/'services'/f'{name}.json',dict(unit=unit,command=command,created_utc=datetime.now(timezone.utc).isoformat()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--launch');p.add_argument('command',nargs='*');a=p.parse_args()
    if a.launch:launch(a.launch,a.command)
    else:
        print(json.dumps(dict(config=json.loads(CONFIG.read_text()),remaining_seconds=max(0,remaining()),services=[json.loads(p.read_text()) for p in (BASE/'services').glob('*.json')]),indent=2))
if __name__=='__main__':main()
