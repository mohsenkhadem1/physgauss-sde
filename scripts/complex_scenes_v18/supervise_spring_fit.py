"""Bounded all-family development scalar warm-start queue."""
import os,subprocess,time,json
from scripts.complex_scenes_v18.run_development import ROOT,write


if __name__=='__main__':
    folder=ROOT/'runs/complex-scenes-v18/spring-scale-fit-01';folder.mkdir(parents=True,exist_ok=True)
    jobs=json.loads((ROOT/'runs/complex-scenes-v18/development-prefixes/manifest.json').read_text())['cases']
    queue=[];done=[];active={}
    for j in jobs:
        name=f"{j['case']}-v{j['variant']}";out=folder/name
        if (out/'fit-seal.json').exists()and(out/'sealed-rollout'/name/'complete.json').exists():done.append(dict(case=j['case'],variant=j['variant'],recovered=True))
        elif out.exists():raise RuntimeError('Existing incomplete attempt requires explicit review: '+name)
        else:queue.append(j)
    while queue or active:
        for gpu in (0,1):
            if gpu in active or not queue:continue
            j=queue.pop(0);log=open(folder/f"{j['case']}-v{j['variant']}.log",'ab',buffering=0)
            cmd=[str(ROOT/'runs/complex-scenes-v18/baseline-env/bin/python'),'-m','scripts.complex_scenes_v18.spring_scale_fit','--case',j['case'],'--variant',str(j['variant'])]
            p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[gpu]=(p,j,log)
        for gpu,(p,j,log)in list(active.items()):
            if p.poll()is not None:done.append(dict(case=j['case'],variant=j['variant'],exit_code=p.returncode));log.close();del active[gpu]
        write(folder/'supervisor.json',dict(pid=os.getpid(),planned=[dict(case=j['case'],variant=j['variant'])for j in jobs],running=[dict(case=j['case'],variant=j['variant'],pid=p.pid,gpu=g)for g,(p,j,_)in active.items()],finished=done,terminal=len(done)==len(jobs),scope='Scalar stiffness warm starts only; final per-edge fitting outstanding'))
        time.sleep(2)
