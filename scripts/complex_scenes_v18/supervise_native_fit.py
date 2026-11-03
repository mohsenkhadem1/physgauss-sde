"""Resumable bounded sharing queue; adopts owned native fit workers by argv."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.task_worker import alive,start_ticks


def identity(pid,method,label):
    try:
        proc=Path('/proc')/str(pid);args=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        if 'scripts.complex_scenes_v18.spring_edge_fit'not in args or proc.joinpath('cwd').resolve()!=ROOT:return None
        get=lambda key,default:args[args.index(key)+1]if key in args else default
        if get('--method','spring-gaus')!=method or get('--label','spring-edge-fit-01')!=label:return None
        env=proc.joinpath('environ').read_bytes().split(b'\0');gpu=next((v.split(b'=',1)[1].decode()for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
        return (get('--case',None),int(get('--variant','0'))),gpu
    except (OSError,ValueError,IndexError):return None


def run(method,workers):
    if not 1<=workers<=4:raise ValueError('At most four native workers per method')
    base=ROOT/'runs/complex-scenes-v18';label='phystwin-edge-fit-01'if method=='phystwin'else'spring-edge-fit-01';out=base/label
    out.mkdir(parents=True,exist_ok=True);lock=open(out/'sharing-supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=json.loads((base/'development-prefixes/manifest.json').read_text())['cases'];known={(j['case'],j['variant'])for j in cases};active={};done={};owned={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        found=identity(int(proc.name),method,label)
        if found and found[0]in known:
            key,gpu=found
            if key in active:raise ValueError('Duplicate live case worker')
            ticks=start_ticks(int(proc.name))
            if alive(int(proc.name),ticks):active[key]=dict(pid=int(proc.name),ticks=ticks,gpu=gpu)
    for key in known-set(active):
        folder=out/f'{key[0]}-v{key[1]}'
        if folder.exists():done[key]=dict(preserved_existing_attempt=True)
    queue=[(j['case'],j['variant'])for j in cases if (j['case'],j['variant'])not in active and (j['case'],j['variant'])not in done]
    while active or queue:
        for key,value in list(active.items()):
            if not alive(value['pid'],value['ticks']):
                if key in owned:
                    proc,log=owned.pop(key);proc.poll();log.close()
                done[key]=dict(ended=True);del active[key]
        free={int(row.split(',')[0]):int(row.split(',')[1])for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().splitlines()}
        while queue and len(active)<workers:
            eligible=[g for g in (0,1,2)if free.get(g,0)>2048]
            if not eligible:break
            gpu=min(eligible,key=lambda g:sum(v['gpu']==str(g)for v in active.values()))
            key=queue.pop(0);name=f'{key[0]}-v{key[1]}';log=open(out/(name+'.log'),'ab',buffering=0)
            cmd=[str(base/'baseline-env/bin/python'),'-m','scripts.complex_scenes_v18.spring_edge_fit','--case',key[0],'--variant',str(key[1]),'--method',method,'--label',label,'--epochs','40']
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=dict(pid=proc.pid,ticks=start_ticks(proc.pid),gpu=str(gpu));owned[key]=(proc,log);free[gpu]-=1024
        write(out/'sharing-supervisor.json',dict(pid=os.getpid(),worker_limit=workers,running=[dict(case=k[0],variant=k[1],**v)for k,v in active.items()],finished=[dict(case=k[0],variant=k[1],**v)for k,v in done.items()],queued=[dict(case=k[0],variant=k[1])for k in queue],terminal=not active and not queue))
        if active or queue:time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['phystwin','spring-gaus'],required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();run(a.method,a.workers)
