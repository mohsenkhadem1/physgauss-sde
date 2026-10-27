"""Stage full-face calibration references on spare CPUs without selecting a model."""
import argparse,concurrent.futures,fcntl,multiprocessing,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.qualified_reference import BASE,protocol,_truth_job


def one(item):
    name,r,maximum_mesh=item
    directory=BASE/'job-locks';directory.mkdir(parents=True,exist_ok=True)
    with open(directory/f'truth-{name}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:_truth_job(name,r,None,maximum_mesh=maximum_mesh)
        except RuntimeError as e:
            if not str(e).startswith(f'{name}: spatial qualification failed'):raise
            return dict(case=name,status='coarse fields staged; not qualified',maximum_mesh=maximum_mesh)
    return dict(case=name,status='qualified')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--maximum-mesh',type=int,choices=[8,12,16],default=12);a=p.parse_args()
    jobs=[(f'calibration-{i}',r,a.maximum_mesh) for i,r in enumerate(protocol()['calibration']) if r['layout'].endswith('full')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(one,jobs):print(result,flush=True)

if __name__=='__main__':main()
