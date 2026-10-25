"""Prospective owned-child pipeline. Never launches, signals or retries the parent.

Crash recovery verifies completed stages; an interrupted stage is retained and
stops the pipeline rather than silently replaying any scientific computation.
"""
import argparse,fcntl,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from experiments import run_fresh64_numerical_continuation_v01 as continued
from experiments import audit_fresh64_numerical_continuation_v01 as auditor
from experiments import analyze_fresh64_continued_predictive_risk_v01 as risk
ROOT=continued.ROOT;LOCAL=ROOT/'runs/operator-dev/fresh64-postprimary-supervisor-v01'
SOURCE='experiments/supervise_fresh64_postprimary_v01.py';TEST='tests/test_supervise_fresh64_postprimary_v01.py'
PARENT_SHA=continued.PARENT_SHA
PARENT_RELEASE='278528c14c1b5bdd1505604aa6605f0bb29bddc5cb896cc5a1b31ed6e5901471'
FINE_SHA='0f616bb0005c9a211623aeeca3d6dd09053c6f2d429f3b1c200ad2891b5d82d6'
PUBLISHED_MAIN='6587f16b2e562ae272b095e35303a07787298f70'
PARENT_PID=440501;PARENT_WORKERS=list(range(440545,440561));GiB=1024**3
GO_NAMES=['continuation-runner-final-source-GO-v01.json','1024-extension-and-threecase-control-final-source-GO-v01.json','pure-fresh64-continuation-helper-source-GO-v01.json','continued-independent-auditor-source-GO-v01.json','extended-linear-and-continued-risk-source-GO-v01.json']
sha=continued.sha

def read(path):return json.loads(Path(path).read_text())
def record(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');os.replace(tmp,path)
def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def exact(binding):
    path=ROOT/binding['path']
    if record(path)!=binding:raise ValueError('Changed pinned bytes: '+binding['path'])
    return path

def identity(pid):
    try:
        base=Path('/proc')/str(pid);s=(base/'stat').read_text();tail=s[s.rfind(')')+2:].split();cmd=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        return dict(pid=pid,PPID=int(tail[1]),PGID=int(tail[2]),state=tail[0],starttime=int(tail[19]),command=cmd)
    except(FileNotFoundError,ProcessLookupError):return None

def alive(item):
    now=identity(item['pid'])
    return now is not None and now['starttime']==item['starttime']and now['state']!='Z'
def absent(pid):
    if identity(pid)is not None:raise ValueError('Prerequisite PID still exists: '+str(pid))
    try:os.killpg(pid,0)
    except ProcessLookupError:return
    raise ValueError('Prerequisite process group still exists: '+str(pid))
def descendant_identities(parent):
    allrows={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=identity(int(path.name))
            if row:allrows[row['pid']]=row
    owned={parent['pid']:parent};changed=True
    while changed:
        changed=False
        for pid,row in allrows.items():
            if pid not in owned and row['PPID']in owned:owned[pid]=row;changed=True
    return list(owned.values())
def cleanup(child,owned):
    """Only identities proven descended from our direct child; never parent study."""
    if any(x['pid']in[PARENT_PID,*PARENT_WORKERS]for x in owned):raise ValueError('Refuse parent cleanup')
    direct=next((x for x in owned if x['pid']==child.pid),None)
    if direct and alive(direct):
        bound={x['pid']:x for x in[ *owned,*descendant_identities(direct)]};owned=list(bound.values())
    if any(x['pid']in[PARENT_PID,*PARENT_WORKERS]for x in owned):raise ValueError('Refuse parent cleanup')
    groups={x['PGID']for x in owned if x['PGID']==x['pid']and alive(x)}
    for pgid in groups:
        if any(x['pid']==pgid and alive(x)for x in owned):os.killpg(pgid,signal.SIGTERM)
    deadline=time.monotonic()+65
    while any(alive(x)for x in owned)and time.monotonic()<deadline:time.sleep(.05)
    for x in owned:
        if x['PGID']==x['pid']and alive(x):os.killpg(x['pid'],signal.SIGKILL)
    child.wait(timeout=10)
    if any(alive(x)for x in owned):raise RuntimeError('Owned stage cleanup incomplete')

def owned_RSS(row):
    deadline=time.monotonic()+.25
    while True:
        if not alive(row):return 0
        try:
            lines=(Path('/proc')/str(row['pid'])/'status').read_text().splitlines()
            rss=next((int(v.split()[1])*1024 for v in lines if v.startswith('VmRSS:')),None)
        except FileNotFoundError:rss=None
        if rss is not None:return rss
        if time.monotonic()>=deadline:
            if not alive(row):return 0
            raise RuntimeError('Live owned process RSS unresolved after bounded exit grace')
        time.sleep(.01)

def resources():
    return dict(workers=8,worker_CPUs=list(range(8)),controller_CPU=16,threads=1,GPU=False,worker_RSS_bytes=4*GiB,controller_RSS_bytes=4*GiB,aggregate_RSS_bytes=36*GiB,RAM_reserve_bytes=30*GiB,disk_reserve_bytes=30*GiB,host_preflight_bytes=66*GiB,maximum_writes_bytes=12*GiB,worker_wall_s=18000,global_wall_s=19800)
def sources():
    names=set(continued.sources())|set(auditor.sources())|set(risk.sources())|{SOURCE,TEST,'experiments/plot_fresh64_calibration_v01.py'}
    return {name:sha(ROOT/name)for name in sorted(names)}
def check_sources(p):
    if p.get('_runtime_release'):exact(p['_runtime_release']);exact(p['_runtime_protocol'])
    if p.get('resources_file'):
        if read(exact(p['resources_file']))!=resources():raise ValueError('Exact fixed continuation resources')
    if p['source_SHA256']!=sources():raise ValueError('Supervisor reviewed source closure changed')
    for item in p['review_receipts']:exact(item)
    for key in('parent_protocol','parent_release','fine_protocol'):exact(p[key])
    if p['parent_protocol']['sha256']!=PARENT_SHA or p['parent_release']['sha256']!=PARENT_RELEASE or p['fine_protocol']['sha256']!=FINE_SHA:raise ValueError('Hardcoded original/fine roots')

def parent_identities():
    controller=identity(PARENT_PID);workers={str(pid):identity(pid)for pid in PARENT_WORKERS}
    for row,action in[(controller,'run'),*[(v,'worker')for v in workers.values()]]:
        if row is None or row['state']=='Z':continue
        if row['PGID']!=row['pid']or 'experiments.run_fresh64_nonlinear_calibration_v01 '+action not in row['command']or PARENT_SHA not in row['command']:raise ValueError('Exact original command/protocol/session identity')
        if action=='worker'and row['PPID']!=PARENT_PID:raise ValueError('Exact original worker parent')
    return controller,workers


def reviewed_receipts():
    receipts=[record(ROOT/'runs/operator-dev/clamped-surface-bridge-review-v01'/n)for n in GO_NAMES]
    for name,receipt in zip(GO_NAMES,receipts):
        value=read(exact(receipt))
        expected_key={'1024-extension-and-threecase-control-final-source-GO-v01.json':'source_math_resource_GO','pure-fresh64-continuation-helper-source-GO-v01.json':'pure_helper_source_GO'}.get(name,'source_GO')
        if value.get(expected_key)is not True:raise ValueError('Explicit named peer source GO')
        for path,digest in value.get('source_SHA256',value.get('sources',{})).items():
            if sha(ROOT/path)!=digest:raise ValueError('Peer GO pinned source changed')
    return receipts


def prepare():
    """Metadata/source freeze only. No parent execution/results/arrays opened."""
    receipts=reviewed_receipts()
    subprocess.run(['git','cat-file','-e',PUBLISHED_MAIN+'^{commit}'],cwd=ROOT,check=True,capture_output=True)
    for name in(continued.SOURCE,continued.TEST,auditor.SOURCE,auditor.TEST,risk.SOURCE):
        data=subprocess.run(['git','show',PUBLISHED_MAIN+':'+name],cwd=ROOT,check=True,capture_output=True).stdout
        if hashlib.sha256(data).hexdigest()!=sha(ROOT/name):raise ValueError('Published reviewed main source differs: '+name)
    parent_identity,parent_workers=parent_identities()
    exclusive(LOCAL/'resources.json',resources())
    p=dict(resources_file=record(LOCAL/'resources.json'),schema='fresh64-postprimary-supervisor-v01',source_SHA256=sources(),published_main=PUBLISHED_MAIN,review_receipts=receipts,parent_protocol=record(continued.parent.LOCAL/'protocol.json'),parent_release=record(continued.parent.LOCAL/'release.json'),fine_protocol=record(continued.fine.LOCAL/'protocol.json'),parent_PID=PARENT_PID,parent_workers=PARENT_WORKERS,parent_identity=parent_identity,parent_worker_identities=parent_workers,resources=resources(),audit_source_SHA256=auditor.audit_predeclared_sources(),wait_cap_s=21000,poll_s=1.,stage_caps=dict(original_audit=[18,360,4*GiB],continuation_prepare=[16,600,4*GiB],continuation_run=[16,20400,36*GiB],continued_audit=[18,660,4*GiB],risk_prepare=[17,120,4*GiB],risk_run=[17,360,4*GiB],plots=[17,120,4*GiB]))
    check_sources(p);exclusive(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')

def supervisor_release(digest):
    p=read(LOCAL/'protocol.json');check_sources(p)
    if sha(LOCAL/'protocol.json')!=digest:raise ValueError('Exact supervisor protocol')
    r=read(LOCAL/'release.json')
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('authorize_exact_continuation_release')is not True:raise ValueError('Explicit root supervisor and continuation authorization')
    p['_runtime_release']=record(LOCAL/'release.json');p['_runtime_protocol']=record(LOCAL/'protocol.json')
    return p

def parent_terminal(p):
    """Read-only: never signal/relaunch the original controller/workers."""
    check_sources(p);path=continued.parent.LOCAL/'execution.json'
    now=identity(PARENT_PID);old=p['parent_identity']
    if now and old and now['starttime']!=old['starttime']:raise ValueError('Parent PID reuse; cannot infer original closure')
    try:e=read(path)
    except json.JSONDecodeError:
        if now and old and now['starttime']==old['starttime']and now['state']!='Z':return False
        raise ValueError('Corrupt parent terminal journal after closure')
    for key,expected in p.get('parent_worker_identities',{}).items():
        actual=identity(int(key))
        if actual and expected and actual['starttime']!=expected['starttime']:raise ValueError('Original worker PID reused')
    if e.get('controller_pid')!=PARENT_PID:raise ValueError('Exact original parent controller')
    rows=e.get('workers',[])
    if any(r.get('pid')not in PARENT_WORKERS for r in rows):raise ValueError('Exact original worker membership')
    if not e.get('terminal'):return False
    if e.get('completed')is not True or'failure'in e or len(rows)!=16 or {r['pid']for r in rows}!=set(PARENT_WORKERS)or any(r.get('reaped')is not True or r.get('exit_code')!=0 for r in rows):raise ValueError('Original execution failed: stop without relaunch')
    for pid in[PARENT_PID,*PARENT_WORKERS]:
        try:absent(pid)
        except ValueError:return False
    return True

def wait_parent(p):
    start=time.monotonic()
    while not parent_terminal(p):
        if time.monotonic()-start>=p['wait_cap_s']:raise TimeoutError('Original parent wait cap; parent untouched')
        time.sleep(min(1.,p['poll_s']))

def stage_command(module,args,cpu):return ['taskset','-c',str(cpu),sys.executable,'-u','-m',module,*args]
def stage(name,command,outputs,p):
    """Completed stages verify and skip. Partial/existing stages never replay."""
    check_sources(p);directory=LOCAL/'stages'/name;statepath=directory/'execution.json'
    if statepath.exists():
        state=read(statepath)
        if state.get('completed')is not True:raise RuntimeError('Interrupted/failed stage retained; no automatic retry: '+name)
        if state.get('command')!=command or state.get('protocol_SHA256')!=sha(LOCAL/'protocol.json'):raise ValueError('Changed completed stage command/protocol')
        if state.get('outputs')!=[record(path)for path in outputs]:raise ValueError('Changed completed stage output membership/bytes')
        for binding in state['outputs']:exact(binding)
        for row in state.get('owned',[]):absent(row['pid'])
        if state.get('child'):absent(state['child']['pid'])
        return state
    if any(Path(path).exists()for path in outputs):raise FileExistsError('Preexisting unowned stage outputs: '+name)
    directory.mkdir(parents=True,exist_ok=True);cpu,cap,rsscap=p['stage_caps'][name]
    state=dict(stage=name,completed=False,command=command,protocol_SHA256=sha(LOCAL/'protocol.json'),supervisor_PID=os.getpid(),outputs=[])
    exclusive(statepath,state);token=directory/'start.json';log=(directory/'stdout.log').open('xb');env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    bootstrap=stage_command('experiments.supervise_fresh64_postprimary_v01',['child','--token',str(token),'--command-json',json.dumps(command)],cpu)
    child=subprocess.Popen(bootstrap,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=[];start=time.monotonic()
    try:
        info=identity(child.pid)
        if not info or info['PGID']!=child.pid:raise RuntimeError('Owned child identity before handshake')
        state['child']=info;atomic(statepath,state);exclusive(token,dict(pid=child.pid,command=command))
        while child.poll()is None:
            rows=descendant_identities(info);owned={x['pid']:x for x in[ *owned,*rows]};owned=list(owned.values());state['owned']=owned;atomic(statepath,state)
            if time.monotonic()-start>cap:raise TimeoutError('Owned stage wall cap: '+name)
            total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for row in rows:
                if alive(row):
                    total+=owned_RSS(row)
            if total>rsscap:raise MemoryError('Owned stage aggregate RSS cap: '+name)
            check_sources(p);time.sleep(.5)
        code=child.wait();state.update(exit_code=code,reaped=True)
        if code!=0:raise RuntimeError('Stage failed without retry: '+name)
        for row in owned:
            if row['pid']!=child.pid and alive(row):raise RuntimeError('Stage left owned descendant alive')
        for path in outputs:
            if not Path(path).is_file():raise ValueError('Missing exact stage output: '+str(path))
        state.update(completed=True,outputs=[record(path)for path in outputs])
    except BaseException as error:
        state['failure']=dict(type=type(error).__name__,message=str(error));cleanup(child,owned or([identity(child.pid)]if identity(child.pid)else[]));state.update(exit_code=child.returncode,reaped=True);raise
    finally:log.close();state['wall_s']=time.monotonic()-start;atomic(statepath,state)
    return state

def original_audit_admission(value):
    if value.get('completed')is not True or value.get('integrity_passed')is not True or value.get('execution_clean')is not True or value.get('comparison_blocked')is not True or value.get('objects')!=64 or value.get('qualified_posteriors',896)>=896:raise ValueError('Original integrity PASS/numerically blocked census required')
def continuation_admission(census,fine_result):
    if census['counts']['blocked']:raise ValueError('Blocked original slots retained; no continuation launch')
    jobs=[e for e in census['entries']if e['decision']=='refine']
    if any(e['plan']['inner'][-1]==1024 and e['plan']['outer'][-1]==1024 for e in jobs):raise ValueError('Both-axis control needed; no automatic release')
    times=[x['seconds']for x in fine_result['cases'].values()]
    if not times or max(times)*((len(jobs)+7)//8)*3>18000:raise ValueError('Same-data fine-control cost admission exceeds fixed wall budget')
    return len(jobs)

def pipeline(digest):
    LOCAL.mkdir(parents=True,exist_ok=True)
    lock=(LOCAL/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another supervisor owns this pipeline')
    statepath=LOCAL/'execution.json'
    previous=None
    if statepath.exists():
        previous=read(statepath)
        if previous.get('protocol_SHA256')!=digest:raise ValueError('Prior supervisor protocol differs')
        if previous.get('identity')and alive(previous['identity']):raise ValueError('Prior supervisor still alive')
        archive=LOCAL/('prior-execution-'+sha(statepath)+'.json')
        if not archive.exists():exclusive(archive,previous)
        previous=record(archive)
    state=dict(completed=False,PID=os.getpid(),identity=identity(os.getpid()),protocol_SHA256=digest,prior_execution=previous)
    atomic(statepath,state)
    try:
        pipeline_locked(digest)
        state['completed']=True
    except BaseException as error:
        state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:atomic(statepath,state);lock.close()

def verify_completed(p,digest):
    value=read(LOCAL/'completed.json')
    if value.get('completed')is not True or value.get('protocol_SHA256')!=digest or set(value.get('stages',{}))!=set(p['stage_caps']):raise ValueError('Exact completed pipeline protocol/stage inventory')
    for name,binding in value['stages'].items():
        path=LOCAL/'stages'/name/'execution.json'
        if record(path)!=binding:raise ValueError('Completed stage journal changed')
        state=read(exact(binding))
        if state.get('completed')is not True or state.get('protocol_SHA256')!=digest:raise ValueError('Completed stage protocol/admission changed')
        for output in state['outputs']:exact(output)
        for row in state.get('owned',[]):absent(row['pid'])
        if state.get('child'):absent(state['child']['pid'])
    return value


def pipeline_locked(digest):
    if set(os.sched_getaffinity(0))!={18}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Supervisor CPU18/one-thread/noGPU')
    p=supervisor_release(digest)
    if(LOCAL/'completed.json').exists():verify_completed(p,digest)
    wait_parent(p)
    old=continued.parent_audit;ep=continued.parent.LOCAL/'execution.json';sp=continued.parent.LOCAL/'summary-LOCAL.json'
    stage('original_audit',stage_command('experiments.audit_fresh64_nonlinear_calibration_v01',['--protocol-sha',PARENT_SHA,'--execution-sha',sha(ep),'--summary-sha',sha(sp)],18),[old.LOCAL/'result.json'],p);original_audit_admission(read(old.LOCAL/'result.json'))
    stage('continuation_prepare',stage_command('experiments.run_fresh64_numerical_continuation_v01',['prepare','--fine-sha',FINE_SHA,'--resources-json',str(LOCAL/'resources.json')],16),[continued.LOCAL/'protocol.json',continued.LOCAL/'census-LOCAL.json',continued.LOCAL/'input-pins.json'],p)
    cp=read(continued.LOCAL/'protocol.json');c=read(continued.LOCAL/'census-LOCAL.json');f=read(ROOT/cp['fine_control']['result']['path']);continuation_admission(c,f);cs=sha(continued.LOCAL/'protocol.json')
    release=dict(approved=True,protocol_SHA256=cs,source_SHA256=cp['source_SHA256'],parent=cp['parent'],fine_control=cp['fine_control'],audit_source_SHA256=p['audit_source_SHA256'],supervisor_authorization=record(LOCAL/'release.json'))
    rp=continued.LOCAL/'release.json'
    if rp.exists():
        if read(rp)!=release:raise ValueError('Existing continuation release differs')
    else:exclusive(rp,release)
    stage('continuation_run',stage_command('experiments.run_fresh64_numerical_continuation_v01',['run','--sha',cs],16),[continued.LOCAL/'execution.json',continued.LOCAL/'summary-LOCAL.json'],p)
    stage('continued_audit',stage_command('experiments.audit_fresh64_numerical_continuation_v01',['--protocol-sha',cs,'--execution-sha',sha(continued.LOCAL/'execution.json'),'--summary-sha',sha(continued.LOCAL/'summary-LOCAL.json')],18),[auditor.LOCAL/'result.json'],p)
    a=read(auditor.LOCAL/'result.json')
    if a.get('integrity_passed')is not True or a.get('comparison_blocked')is not False or a.get('qualified_posteriors')!=896:raise ValueError('Continued independent audit must fully admit all896')
    stage('risk_prepare',stage_command('experiments.analyze_fresh64_continued_predictive_risk_v01',['prepare','--continuation-sha',cs],17),[risk.LOCAL/'recipe.json'],p)
    stage('risk_run',stage_command('experiments.analyze_fresh64_continued_predictive_risk_v01',['run','--recipe-sha',sha(risk.LOCAL/'recipe.json'),'--audit-sha',sha(auditor.LOCAL/'result.json')],17),[risk.LOCAL/'summary.json',risk.LOCAL/'risk-inputs-LOCAL.npz'],p)
    output=ROOT/'runs/operator-dev/fresh64-final-media-v01'
    stage('plots',stage_command('experiments.plot_fresh64_calibration_v01',['--summary',str(continued.LOCAL/'summary-LOCAL.json'),'--audit',str(auditor.LOCAL/'result.json'),'--output',str(output)],17),[output/'provenance.json',output/'stiffness-coverage.png',output/'held-load-prediction.png'],p)
    completion=dict(completed=True,protocol_SHA256=digest,continued_protocol_SHA256=cs,no_parent_signals=True,no_retries=True,no_publication=True,stages={name:record(LOCAL/'stages'/name/'execution.json')for name in p['stage_caps']})
    if(LOCAL/'completed.json').exists():
        if read(LOCAL/'completed.json')!=completion:raise ValueError('Completed pipeline commitments changed')
    else:exclusive(LOCAL/'completed.json',completion)

def child(token,command):
    deadline=time.monotonic()+5
    while not Path(token).exists():
        if time.monotonic()>deadline:raise TimeoutError('No durable owned-stage handshake')
        time.sleep(.02)
    value=read(token)
    if value['pid']!=os.getpid()or value['command']!=command:raise ValueError('Exact child handshake')
    os.execv(command[0]if command[0].startswith('/')else'/usr/bin/'+command[0],command)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','child']);ap.add_argument('--sha');ap.add_argument('--token');ap.add_argument('--command-json');args=ap.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='child':child(args.token,json.loads(args.command_json))
    else:
        def stop(*_):raise InterruptedError('Explicit supervisor stop; only own stage children cleaned')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);pipeline(args.sha)
