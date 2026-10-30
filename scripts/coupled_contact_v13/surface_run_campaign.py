"""Bounded durable supervisor for the separately frozen surface extension."""
from pathlib import Path
import argparse,os,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from scripts.coupled_contact_v13.campaign import remaining,write
from scripts.coupled_contact_v13.surface_pipeline import BASE,REPORT,implementation


def stages():
    out=[('library',['library'])]
    for split in ('development','reserved'):
        out.extend([(split+'-reference',['reference','--split',split]),(split+'-evaluate',['evaluate','--split',split])])
    return out


def worker_count(value):
    count=int(value)
    if count<1 or count>36:raise argparse.ArgumentTypeError('workers must be1–36')
    return count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=worker_count,default=12);options=parser.parse_args()
    implementation().check_freeze()
    BASE.mkdir(parents=True,exist_ok=True)
    python=ROOT/'runs/phystwin-env/bin/python'
    for name,args in stages():
        implementation().check_freeze()
        marker=BASE/'stages'/f'{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('success'):continue
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached')
        with (BASE/f'{name}.log').open('a') as log:
            result=subprocess.run([str(python),str(ROOT/'scripts/coupled_contact_v13/surface_pipeline.py'),*args,'--workers',str(options.workers)],
                                  stdout=log,stderr=subprocess.STDOUT,timeout=max(1,remaining()),cwd=ROOT)
        write(marker,dict(success=result.returncode==0,returncode=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if result.returncode:raise RuntimeError('Surface stage failed: '+name)
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached')
        subprocess.run([str(python),str(ROOT/'scripts/coupled_contact_v13/surface_report.py')],check=True,timeout=max(1,remaining()),cwd=ROOT)
    write(BASE/'complete.json',dict(status='Surface experiments complete; supplements, independent verification, visual audit and publication still required',
                                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__=='__main__':main()
