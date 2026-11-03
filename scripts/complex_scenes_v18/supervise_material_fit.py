"""Bounded two-worker development calibration queue, all six scene families."""
import json,os,subprocess,time
from scripts.complex_scenes_v18.run_development import ROOT,write


if __name__=='__main__':
    base=ROOT/'runs/complex-scenes-v18';folder=base/'material-scene-fit-01'
    folder.mkdir(parents=True,exist_ok=False)
    cases=json.loads((base/'development-prefixes/manifest.json').read_text())['cases']
    queue=cases.copy();active=[];done=[]
    while queue or active:
        while queue and len(active)<2:
            job=queue.pop(0);name=f"{job['case']}-v{job['variant']}"
            log=open(folder/(name+'.log'),'ab',buffering=0)
            cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.complex_scenes_v18.material_scene_fit','--case',job['case'],'--variant',str(job['variant'])]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((proc,job,log))
        for proc,job,log in active.copy():
            if proc.poll()is not None:
                done.append(dict(case=job['case'],variant=job['variant'],exit_code=proc.returncode));log.close();active.remove((proc,job,log))
        write(folder/'supervisor.json',dict(pid=os.getpid(),planned=[dict(case=j['case'],variant=j['variant'])for j in cases],running=[dict(case=j['case'],variant=j['variant'],pid=p.pid)for p,j,_ in active],finished=done,queued=[dict(case=j['case'],variant=j['variant'])for j in queue],terminal=not queue and not active))
        if queue or active:time.sleep(5)
