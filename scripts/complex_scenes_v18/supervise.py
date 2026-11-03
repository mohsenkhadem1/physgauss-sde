"""Small durable development scheduler; never kills unrelated processes."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--n',type=int,default=1);p.add_argument('--dt',type=float,default=50e-6);p.add_argument('--variants',type=int,default=2);p.add_argument('--families',nargs='+');p.add_argument('--surface-samples',type=int,default=5);args=p.parse_args()
    if not 1<=args.workers<=4 or args.variants not in (1,2):raise ValueError('Explicit worker/variant bound')
    folder=ROOT/'runs/complex-scenes-v18'/args.label;folder.mkdir(parents=True,exist_ok=True)
    families=['clamp_release','curved_push','container_drop','rigid_impact','deformable_pair','heterogeneous']
    if args.families:
        if any(f not in families for f in args.families)or len(set(args.families))!=len(args.families):raise ValueError('Invalid family selection')
        families=args.families
    jobs=[dict(family=f,variant=v)for f in families for v in range(args.variants)];active={};done=[]
    manifest=folder/'supervisor.json'
    def save():
        value=dict(label=args.label,pid=os.getpid(),n=args.n,dt_s=args.dt,workers=args.workers,planned=jobs,running=[dict(**j,pid=proc.pid)for proc,j,log in active.values()],finished=done,terminal=len(done)==len(jobs))
        tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(manifest)
    queue=[]
    for j in jobs:
        name=f"{j['family']}-v{j['variant']}-n{args.n}-dt{args.dt:.8f}-s{args.surface_samples}";existing=folder/name
        if (existing/'complete.json').exists():done.append(dict(**j,status=json.loads((existing/'complete.json').read_text())['state'],recovered=True))
        elif existing.exists():raise RuntimeError('Incomplete existing case requires explicit separate recovery: '+name)
        else:queue.append((name,j))
    while queue or active:
        available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
        while queue and len(active)<args.workers and available>24*2**30:
            name,j=queue.pop(0);log=open(folder/(name+'.log'),'ab',buffering=0)
            cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.complex_scenes_v18.run_development','--case',j['family'],'--variant',str(j['variant']),'--n',str(args.n),'--dt',str(args.dt),'--label',args.label,'--surface-samples',str(args.surface_samples)]
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[name]=(proc,j,log)
        for name,(proc,j,log)in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close();done.append(dict(**j,exit_code=code));del active[name]
        save();time.sleep(2)
    save()


if __name__=='__main__':main()
