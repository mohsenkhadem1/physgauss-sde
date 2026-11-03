"""Resumable, bounded CPU queue for all six material-calibration families."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write


def identity(pid,label='material-scene-fit-01'):
    try:
        p=Path('/proc')/str(pid)
        if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return None
        argv=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if 'scripts.complex_scenes_v18.material_scene_fit' not in argv or (p/'cwd').resolve()!=ROOT:return None
        actual=argv[argv.index('--label')+1]if '--label'in argv else'material-scene-fit-01'
        if actual!=label:return None
        return argv[argv.index('--case')+1],int(argv[argv.index('--variant')+1])
    except (OSError,ValueError,IndexError):return None


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--workers',type=int,default=8);parser.add_argument('--label',default='material-scene-fit-01');parser.add_argument('--initialize-label');parser.add_argument('--evaluations',type=int,default=64);args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('Bounded worker count required')
    base=ROOT/'runs/complex-scenes-v18';folder=base/args.label;folder.mkdir(parents=True,exist_ok=args.resume)
    lock=open(folder/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=json.loads((base/'development-prefixes/manifest.json').read_text())['cases'];queue=[];active={};done=[]
    known={(j['case'],j['variant']):j for j in cases}
    if args.resume:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            key=identity(int(entry.name),args.label)
            if key in known:
                if key in active:raise RuntimeError('Duplicate live fit for same case')
                active[key]=(int(entry.name),None,None)
    for key,job in known.items():
        casefolder=folder/f'{key[0]}-v{key[1]}'
        if key in active:continue
        if casefolder.exists():
            final=casefolder/'sealed-rollout'/casefolder.name/'complete.json'
            done.append(dict(case=key[0],variant=key[1],completed=bool(final.exists()and json.loads(final.read_text())['completed']),recovered=True))
        else:queue.append(job)
    while queue or active:
        while queue and len(active)<args.workers and available_bytes()>16*1024**3:
            ready=[j for j in queue if not args.initialize_label or (base/args.initialize_label/f"{j['case']}-v{j['variant']}"/'fit-seal.json').exists()]
            if not ready:break
            job=ready[0];queue.remove(job);key=(job['case'],job['variant'])
            initial=base/args.initialize_label/f'{key[0]}-v{key[1]}'if args.initialize_label else None
            if initial and not json.loads((initial/'fit-seal.json').read_text())['completed']:
                done.append(dict(case=key[0],variant=key[1],completed=False,error='Initialization fit failed'));continue
            log=open(folder/(f'{key[0]}-v{key[1]}.log'),'ab',buffering=0)
            cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.complex_scenes_v18.material_scene_fit','--case',key[0],'--variant',str(key[1]),'--label',args.label,'--evaluations',str(args.evaluations)]
            if initial:cmd+=['--initialize-fit',str(initial)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=(proc.pid,proc,log)
        for key,(pid,proc,log)in list(active.items()):
            ended=(proc.poll()is not None)if proc is not None else identity(pid,args.label)!=key
            if ended:
                final=folder/f'{key[0]}-v{key[1]}'/'sealed-rollout'/f'{key[0]}-v{key[1]}'/'complete.json'
                done.append(dict(case=key[0],variant=key[1],completed=bool(final.exists()and json.loads(final.read_text())['completed']),exit_code=proc.returncode if proc else None))
                if log:log.close()
                del active[key]
        write(folder/'supervisor.json',dict(pid=os.getpid(),worker_limit=args.workers,planned=[dict(case=j['case'],variant=j['variant'])for j in cases],running=[dict(case=k[0],variant=k[1],pid=v[0])for k,v in active.items()],finished=done,queued=[dict(case=j['case'],variant=j['variant'])for j in queue],terminal=not queue and not active))
        if queue or active:time.sleep(5)
