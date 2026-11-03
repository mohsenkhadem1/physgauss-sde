"""Bounded, resumable dependency scheduler for the frozen held campaign.

Only launches declared task workers. Existing live worker/child process identities
are adopted, never killed or duplicated. Interrupted tasks need explicit review.
"""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.held_manifest import sha,validate_freeze
from scripts.complex_scenes_v18.held_tasks import build
from scripts.complex_scenes_v18.task_worker import key,alive,start_ticks

TERMINAL={'complete','failed','dependency_failed','interrupted'}


def observe(task,state_dir,plan_hash):
    path=Path(state_dir)/(key(task['id'])+'.json')
    if not path.exists():
        claim=Path(state_dir)/(key(task['id'])+'.claim')
        if claim.exists():
            # A claim with no status is never permission to launch a duplicate.
            try:
                pid=int(claim.read_text());argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
                live='scripts.complex_scenes_v18.task_worker'in argv and task['id']in argv
            except (OSError,ValueError):live=False
            status=dict(state='unresolved_claim'if live else'interrupted',task_id=task['id'],plan_sha256=plan_hash)
            if not live:write(path,status)
            return status
        return dict(state='pending',task_id=task['id'])
    status=json.loads(path.read_text())
    if status['task_id']!=task['id']or status['plan_sha256']!=plan_hash:raise ValueError('Task identity/plan changed')
    if status['state']=='running':
        worker=alive(status['pid'],status['start_ticks'])
        child=alive(status.get('child_pid',-1),status.get('child_start_ticks'))
        if not worker and not child:
            status.update(state='interrupted',error='Worker and child identities no longer live; no automatic retry')
            write(path,status)
    return status


def ready(tasks,states):
    return [t for t in tasks if states[t['id']]['state']=='pending'and all(states[d]['state']=='complete'for d in t['dependencies'])]


def free_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return 0


def gpu_memory():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
    return {int(row.split(',')[0]):int(row.split(',')[1])for row in result.splitlines()}


def run(campaign,freeze_path,workers=8,gpus=(0,1,2)):
    campaign=Path(campaign).resolve();freeze_path=Path(freeze_path).resolve()
    if not 1<=workers<=12 or not gpus or any(g not in (0,1,2)for g in gpus)or len(gpus)!=len(set(gpus)):raise ValueError('Bounded CPUs and authorized GPU 0–2 only')
    frozen=json.loads(freeze_path.read_text());protocol=validate_freeze(frozen,ROOT)
    seal=json.loads((campaign/'manifest-seal.json').read_text())
    if sha(freeze_path)!=seal['freeze_sha256']or sha(campaign/'public-manifest.json')!=seal['public_manifest_sha256']:raise ValueError('Frozen campaign identity mismatch')
    public=json.loads((campaign/'public-manifest.json').read_text());tasks=build(protocol,public,campaign,freeze_path)
    state_dir=campaign/'task-state';state_dir.mkdir(exist_ok=True)
    lock=open(state_dir/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan_path=campaign/'task-plan.json';plan=dict(freeze_sha256=sha(freeze_path),public_manifest_sha256=seal['public_manifest_sha256'],tasks=tasks)
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing task plan differs')
    else:write(plan_path,plan)
    plan_hash=sha(plan_path);children={};launching={}
    while True:
        states={t['id']:observe(t,state_dir,plan_hash)for t in tasks}
        for task_id,(proc,log)in list(children.items()):
            if proc.poll()is not None:log.close();del children[task_id]
        for task_id,proc in list(launching.items()):
            if proc.poll()is not None and states[task_id]['state']in ('pending','unresolved_claim'):
                states[task_id]=dict(task_id=task_id,state='failed',plan_sha256=plan_hash,exit_code=proc.returncode,error='Worker exited before writing status')
                write(state_dir/(key(task_id)+'.json'),states[task_id]);del launching[task_id]
            elif states[task_id]['state']=='running'or proc.poll()is not None:del launching[task_id]
            else:states[task_id]=dict(state='launching')
        for t in tasks:
            if states[t['id']]['state']=='pending'and any(states[d]['state']in TERMINAL-{'complete'}for d in t['dependencies']):
                status=dict(task_id=t['id'],state='dependency_failed',plan_sha256=plan_hash,dependencies=t['dependencies'])
                write(state_dir/(key(t['id'])+'.json'),status);states[t['id']]=status
        running=[t for t in tasks if states[t['id']]['state']in ('running','launching','unresolved_claim')]
        cpu=sum(t['kind']=='cpu'for t in running)
        used={int(states[t['id']]['cuda_visible_devices'])for t in running if t['kind']=='gpu'and states[t['id']].get('cuda_visible_devices','').isdigit()}
        # Fresh launches have not necessarily written their allocation yet.
        for t in running:
            if t['kind']=='gpu'and(t['id']in launching or states[t['id']]['state']=='unresolved_claim'):used.update(gpus)
        available=gpu_memory() if any(t['kind']=='gpu'for t in ready(tasks,states))else{}
        candidates=sorted(ready(tasks,states),key=lambda t:{'prefix':0,'fit':1,'prediction':2,'reference':3,'reference-refinement':4}[t['role']])
        for t in candidates:
            if free_memory()<16*1024**3:break
            gpu=None
            if t['kind']=='cpu':
                if cpu>=workers:continue
            else:
                candidates_gpu=[g for g in gpus if g not in used and available.get(g,0)>=(6144 if t.get('method')=='uniphy'else 2048)]
                if not candidates_gpu:continue
                gpu=candidates_gpu[0]
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''if gpu is None else str(gpu))
            log=open(state_dir/(key(t['id'])+'.log'),'ab',buffering=0)
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','scripts.complex_scenes_v18.task_worker','--plan',str(plan_path),'--task-id',t['id'],'--state-dir',str(state_dir)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[t['id']]=(proc,log);launching[t['id']]=proc
            if gpu is None:cpu+=1
            else:used.add(gpu)
        counts={s:sum(r['state']==s for r in states.values())for s in set(r['state']for r in states.values())}
        terminal=all(r['state']in TERMINAL for r in states.values())
        write(campaign/'supervisor.json',dict(pid=os.getpid(),plan_sha256=plan_hash,counts=counts,terminal=terminal,cpu_workers=workers,gpus=list(gpus),updated_at=time.time()))
        if terminal:
            from scripts.complex_scenes_v18.seal_task_failures import seal as seal_failures
            seal_failures(campaign,plan,states)
            return counts
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--freeze',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2]);a=p.parse_args();print(json.dumps(run(a.campaign,a.freeze,a.workers,tuple(a.gpus))))
