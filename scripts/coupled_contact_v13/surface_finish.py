"""Bounded surface supplements, report, verification and fixed video rendering."""
from pathlib import Path
import argparse,os,datetime,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from scripts.coupled_contact_v13.campaign import remaining,write
from scripts.coupled_contact_v13.surface_pipeline import BASE,implementation

STAGES=[('surface_ablation',[]),('surface_report',[]),('surface_verify',[]),
        ('verify_surface_ablation',[]),('surface_render',[])]


def worker_count(value):
    count=int(value)
    if count<1 or count>36:raise argparse.ArgumentTypeError('workers must be1–36')
    return count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=worker_count,default=12);options=parser.parse_args()
    implementation().check_freeze()
    while not (BASE/'complete.json').exists():
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached before surface completion')
        time.sleep(min(20,remaining()))
    python=ROOT/'runs/phystwin-env/bin/python'
    for name,args in STAGES:
        args=[*args,'--workers',str(options.workers)] if name=='surface_ablation' else args
        implementation().check_freeze()
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached during surface postprocessing')
        logpath=BASE/f'finish-{name}.log'
        with logpath.open('w') as log:
            result=subprocess.run([str(python),str(ROOT/f'scripts/coupled_contact_v13/{name}.py'),*args],
                                  stdout=log,stderr=subprocess.STDOUT,timeout=max(1,remaining()),cwd=ROOT)
        if result.returncode:
            write(BASE/'postprocessing-failed.json',dict(stage=name,returncode=result.returncode,log=str(logpath)))
            raise RuntimeError('Surface postprocessing failed: '+name)
    write(BASE/'postprocessing-complete.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          scope='Surface numerical verification and fixed videos complete; independent visual audit and publication still required'))


if __name__=='__main__':main()
