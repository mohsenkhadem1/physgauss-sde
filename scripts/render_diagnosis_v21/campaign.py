"""Bounded three-GPU evaluation-only oracle diagnostic."""
import concurrent.futures,json,os,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/render-diagnosis-v21'
CASES=['single_lift_sloth','single_lift_dinosor','single_push_sloth','double_lift_sloth',
       'double_stretch_sloth','single_lift_zebra','double_lift_zebra','double_stretch_zebra']

def main():
    jobs=list(CASES);lock=threading.Lock();statuses=[]
    def worker(gpu):
        while True:
            with lock:
                if not jobs:return
                case=jobs.pop(0)
            if (BASE/case/'scores.json').exists():continue
            with (BASE/f'{case}.log').open('w') as log:
                try:code=subprocess.run([str(ROOT/'runs/deform360-reconstruction-env/bin/python'),'-u','-m','scripts.render_diagnosis_v21.run_case','--case',case],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),
                    stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
                except subprocess.TimeoutExpired:code=124
            with lock:
                statuses.append(dict(case=case,gpu=gpu,returncode=code,finished=time.time()))
                (BASE/'status.json').write_text(json.dumps(statuses,indent=2)+'\n')
            print(case,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for task in [pool.submit(worker,gpu) for gpu in (0,1,2)]:task.result()

if __name__=='__main__':main()
