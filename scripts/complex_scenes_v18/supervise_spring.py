"""Development author-force adapter cohort; no fitted-result claim."""
import json,os,subprocess,time
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.scenarios import FAMILIES


def main():
    label='spring-adapter-pilot-01';folder=ROOT/'runs/complex-scenes-v18'/label
    folder.mkdir(parents=True,exist_ok=False)
    jobs=[dict(case=f,variant=v)for f in FAMILIES for v in (0,1)];queue=jobs.copy();active={};done=[]
    while queue or active:
        for gpu in (0,2):
            if gpu in active or not queue:continue
            job=queue.pop(0);log=open(folder/f"{job['case']}-v{job['variant']}.log",'ab',buffering=0)
            cmd=[str(ROOT/'runs/complex-scenes-v18/baseline-env/bin/python'),'-m','scripts.complex_scenes_v18.spring_scene','--case',job['case'],'--variant',str(job['variant']),'--label',label]
            process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(process,job,log)
        for gpu,(p,j,log)in list(active.items()):
            if p.poll()is not None:done.append(dict(**j,exit_code=p.returncode));log.close();del active[gpu]
        write(folder/'supervisor.json',dict(pid=os.getpid(),planned=jobs,running=[dict(**j,pid=p.pid,gpu=g)for g,(p,j,_)in active.items()],finished=done,terminal=len(done)==len(jobs),scope='Unfitted development feasibility, author spring internal forces/common scene adapter'))
        time.sleep(2)


if __name__=='__main__':main()
