"""Bounded sequential supervisor; each stage is durable, logged and resumable."""
from pathlib import Path
import sys,subprocess,json,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.coupled_contact_v13.campaign import BASE,REPORT,ROOT,remaining,write
from scripts.coupled_contact_v13.pipeline import check_freeze

def main():
    check_freeze();python=ROOT/'runs/phystwin-env/bin/python'
    stages=[('library',['library'])]
    for split in ['development','validation','reserved']:
        stages.extend([(f'{split}-reference',['reference','--split',split]),(f'{split}-evaluate',['evaluate','--split',split])])
    for name,args in stages:
        marker=BASE/'stages'/f'{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('success'):continue
        if remaining()<=0:raise RuntimeError('Campaign deadline reached')
        logfile=BASE/f'{name}.log'
        with logfile.open('a') as f:
            r=subprocess.run([str(python),str(ROOT/'scripts/coupled_contact_v13/pipeline.py'),*args,'--workers','4'],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,remaining()),cwd=ROOT)
        write(marker,dict(success=r.returncode==0,returncode=r.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if r.returncode:raise RuntimeError('Stage failed: '+name)
        subprocess.run([str(python),str(ROOT/'scripts/coupled_contact_v13/report.py')],check=True,cwd=ROOT)
    write(BASE/'complete.json',dict(status='all scheduled experiments complete; independent verification/publication still required',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':main()
