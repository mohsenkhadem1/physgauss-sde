"""Run every declared development case as its scalar warm start becomes ready."""
import json,os,subprocess,time
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write


if __name__=='__main__':
    base=ROOT/'runs/complex-scenes-v18';folder=base/'spring-edge-fit-01';folder.mkdir(parents=True,exist_ok=False)
    cases=json.loads((base/'development-prefixes/manifest.json').read_text())['cases'];queue=cases.copy();active={};done=[]
    while queue or active:
        for gpu in (0,1):
            if gpu in active:continue
            selected=None
            for j in queue:
                path=base/'spring-scale-fit-01'/f"{j['case']}-v{j['variant']}"/'fit-seal.json'
                if path.exists():selected=j;break
            if selected is None:continue
            j=selected;queue.remove(j);name=f"{j['case']}-v{j['variant']}";warm=json.loads((base/'spring-scale-fit-01'/name/'fit-seal.json').read_text())
            if not warm['completed']:done.append(dict(case=j['case'],variant=j['variant'],completed=False,error='No valid scalar initialization'));continue
            log=open(folder/(name+'.log'),'ab',buffering=0);cmd=[str(base/'baseline-env/bin/python'),'-m','scripts.complex_scenes_v18.spring_edge_fit','--case',j['case'],'--variant',str(j['variant']),'--epochs','40']
            p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[gpu]=(p,j,log)
        for gpu,(p,j,log)in list(active.items()):
            if p.poll()is not None:done.append(dict(case=j['case'],variant=j['variant'],exit_code=p.returncode));log.close();del active[gpu]
        write(folder/'supervisor.json',dict(pid=os.getpid(),planned=[dict(case=j['case'],variant=j['variant'])for j in cases],running=[dict(case=j['case'],variant=j['variant'],pid=p.pid,gpu=g)for g,(p,j,_)in active.items()],finished=done,waiting_for_initialization=[dict(case=j['case'],variant=j['variant'])for j in queue],terminal=len(done)==len(cases)))
        if queue and not active:
            warmstatus=base/'spring-scale-fit-01/supervisor.json'
            if warmstatus.exists()and json.loads(warmstatus.read_text())['terminal']:raise RuntimeError('Initialization queue ended without every required fit seal; inspect missing cases')
        time.sleep(5)
