"""Root-preauthorized durable operator, outside all frozen scientific graphs.

Source preparation only. Existing scientific gates remain authoritative.
Completed stages verify and skip; a partial/interrupted stage NEVER retries.
Only proven descendants of this operator's own spawned child may be signalled.
"""
import argparse,fcntl,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from experiments import run_fixture_balanced_nh_catalog_v01 as physics
from experiments import run_noisy_poke_benchmark_v01 as noisy
from experiments import audit_noisy_poke_benchmark_v01 as noisy_audit
from experiments import run_matched_finite_poke_control_v01 as finite
ROOT=noisy.ROOT;LOCAL=ROOT/'runs/operator-dev/noisy-poke-operator-v01'
SOURCE='experiments/supervise_noisy_poke_pipeline_v01.py';TEST='tests/test_supervise_noisy_poke_pipeline_v01.py'
PHYSICS_SHA='d915015362736dc04fd2fb24d8bd0c864aa40d0fb592e7f9cb8eb60e2645defc'
DESIGN_SHA=noisy.DESIGN_SHA256;GiB=1024**3
FINITE_AUDIT='experiments/audit_matched_finite_poke_control_v01.py';FINITE_AUDIT_TEST='tests/test_audit_matched_finite_poke_control_v01.py'
CAPS=dict(physics_remaining=[18,21660,76*GiB],physics_finalize=[18,300,4*GiB],physics_audit=[16,3600,4*GiB],noisy_prepare=[18,600,4*GiB],noisy_inputs=[17,300,4*GiB],noisy_choices=[18,6660,28*GiB],finite_choose=[17,1800,4*GiB],noisy_checkpoint=[18,10860,12*GiB],noisy_remaining=[18,10860,28*GiB],noisy_summary=[18,300,4*GiB],noisy_audit=[16,3600,4*GiB],finite_evaluate=[17,1800,4*GiB],finite_audit=[16,1800,4*GiB])
OPERATOR_LIMITS=dict(CPU=19,GPU=False,threads=1,RSS_bytes=4*GiB,RAM_reserve_bytes=30*GiB)
POLICY=dict(noisy_release='exact frozen noisy schema/source/origin/resources; completed independently audited88physical references; approved design',finite_release='exact finite source/resources and same noisy parent protocol; only after32continuous closed seals',noisy_continuation='same32; exact qualified trial0 and existing conservative measured-cost gate; no substitutions')
sha=noisy.sha;read=noisy.read

def record(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');os.replace(temp,path)
def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def exact(value):
    path=ROOT/value['path']
    if record(path)!=value:raise ValueError('Changed exact commitment: '+str(path))
    return path

def identity(pid):
    try:
        path=Path('/proc')/str(pid);text=(path/'stat').read_text();tail=text[text.rfind(')')+2:].split()
        return dict(pid=pid,PPID=int(tail[1]),PGID=int(tail[2]),state=tail[0],starttime=int(tail[19]),command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except(FileNotFoundError,ProcessLookupError):return None

def alive(row):
    now=identity(row['pid']);return now is not None and now['starttime']==row['starttime']and now['state']!='Z'
def absent(pid):
    if identity(pid)is not None:raise ValueError('PID still present: '+str(pid))
    try:os.killpg(pid,0)
    except ProcessLookupError:return
    raise ValueError('Processgroup still present: '+str(pid))
def descendants(parent):
    current=identity(parent['pid'])
    if current is None:return []
    if current['starttime']!=parent['starttime']:raise RuntimeError('Directchild PID reused; refuse descendant ownership')
    table={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=identity(int(path.name))
            if row:table[row['pid']]=row
    owned={parent['pid']:parent};changed=True
    while changed:
        changed=False
        for pid,row in table.items():
            if pid not in owned and row['PPID']in owned:owned[pid]=row;changed=True
    return list(owned.values())
def owned_RSS(row):
    deadline=time.monotonic()+.25
    while True:
        if not alive(row):return 0
        try:lines=(Path('/proc')/str(row['pid'])/'status').read_text().splitlines();value=next((int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:')),None)
        except FileNotFoundError:value=None
        if value is not None:return value
        if time.monotonic()>=deadline:
            if not alive(row):return 0
            raise RuntimeError('Live owned RSS unresolved after bounded exit grace')
        time.sleep(.01)
def cleanup(child,owned):
    """65-second TERM grace; no guessed identities, ancestor or unrelated kills."""
    direct=next((r for r in owned if r['pid']==child.pid),None)
    if direct is None:
        if child.poll()is None:raise RuntimeError('No proven directchild identity; manual recovery required')
        child.wait();return
    if alive(direct):owned=list({r['pid']:r for r in[*owned,*descendants(direct)]}.values())
    def signal_owned(sig):
        leaders={r['pid']for r in owned if r['PGID']==r['pid']and alive(r)}
        for row in owned:
            if not alive(row):continue
            try:
                if row['pid']in leaders:os.killpg(row['pid'],sig)
                elif row['PGID']not in leaders:os.kill(row['pid'],sig) # exact observed identity; never guess a dead leader's group
            except ProcessLookupError:pass
    signal_owned(signal.SIGTERM)
    deadline=time.monotonic()+65
    while any(alive(row)for row in owned)and time.monotonic()<deadline:time.sleep(.05)
    signal_owned(signal.SIGKILL)
    child.wait(timeout=10)
    if any(alive(row)for row in owned):raise RuntimeError('Owned cleanup incomplete; no retry')

def sources():
    names=set(noisy.sources())|set(finite.sources())|{SOURCE,TEST,FINITE_AUDIT,FINITE_AUDIT_TEST}
    return {name:sha(ROOT/name)for name in sorted(names)}
def preapproval(digest):
    path=LOCAL/'root-preapproval.json';p=read(path)
    if sha(path)!=digest or p.get('approved')is not True or p.get('schema')!='noisy-poke-operator-preapproval-v01'or p.get('source_SHA256')!=sources()or p.get('physics_protocol_SHA256')!=PHYSICS_SHA or p.get('prospective_design_SHA256')!=DESIGN_SHA or p.get('stage_caps')!=CAPS or p.get('operator_resources')!=OPERATOR_LIMITS or p.get('dynamic_release_policy')!=POLICY:raise ValueError('Exact ROOT-preapproved operator/sources/policy/resources')
    for key in('physics_release','physics_remaining_release','checkpoint_independent_review'):exact(p[key])
    if p['physics_release']!=record(physics.LOCAL/'release.json')or p['physics_remaining_release']!=record(physics.LOCAL/'continuation-release.json')or p['checkpoint_independent_review']!=record(physics.LOCAL/'checkpoint-independent-review.json'):raise ValueError('Exact manually ROOT-approved physical origin')
    return p

def guard():
    if set(os.sched_getaffinity(0))!={19}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Operator CPU19/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=OPERATOR_LIMITS['RSS_bytes']:raise MemoryError('Own operator4GiB')
    if noisy.available_memory()<OPERATOR_LIMITS['RAM_reserve_bytes']:raise MemoryError('Operator30GiB RAMreserve')

def command(module,args,cpu):return ['taskset','-c',str(cpu),sys.executable,'-u','-m',module,*args]
def verify_stage(value,name,cmd,outputs,digest,parent_sha):
    if not(value.get('completed')is True and value.get('reaped')is True and value.get('exit_code')==0 and value.get('cleanup_exit_code')==0 and value.get('phase')==name and value.get('command')==cmd and value.get('operator_preapproval')==record(LOCAL/'root-preapproval.json')and value.get('source_SHA256')==sources()and value.get('outputs')==[record(path)for path in outputs]and value.get('parent_protocol_SHA256')==parent_sha and value.get('resources')==dict(zip(('CPU','wall_s','RSS_bytes'),CAPS[name]))and value.get('result')==(record(outputs[0])if outputs else None)):raise ValueError('Changed or incomplete exact operator stage: '+name)
    for row in value['observed_owned']:absent(row['pid'])
    absent(value['child_identity']['pid'])
    return value

def stage(name,cmd,outputs,digest,*,parent_sha):
    """A durable launch intent precedes child exec. Partial stages never replay."""
    preapproval(digest);guard();folder=LOCAL/'stages'/name;statepath=folder/'execution.json'
    if statepath.exists():return verify_stage(read(statepath),name,cmd,outputs,digest,parent_sha)
    if any(Path(path).exists()for path in outputs):raise FileExistsError('Unbound existing stage output; no retry: '+name)
    cpu,wall,rss=CAPS[name]
    if noisy.available_memory()<rss+OPERATOR_LIMITS['RSS_bytes']+OPERATOR_LIMITS['RAM_reserve_bytes']:raise MemoryError('Initial full scientific + operator + reserve memory envelope')
    folder.mkdir(parents=True,exist_ok=True);state=dict(phase=name,parent_protocol_SHA256=parent_sha,completed=False,reaped=False,command=cmd,source_SHA256=sources(),operator_preapproval=record(LOCAL/'root-preapproval.json'),resources=dict(CPU=cpu,wall_s=wall,RSS_bytes=rss),outputs=[],observed_owned=[]);exclusive(statepath,state)
    token=folder/'start.json';bootstrap=command('experiments.supervise_noisy_poke_pipeline_v01',['child','--token',str(token),'--command-json',json.dumps(cmd)],cpu);child=None;owned=[];start=time.monotonic()
    try:
        with(folder/'stdout-LOCAL.log').open('xb')as log:
            child=subprocess.Popen(bootstrap,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row=identity(child.pid)
            if not row or row['PGID']!=child.pid:raise RuntimeError('Exact owned child session before handshake')
            owned=[row];state['child_identity']=row;atomic(statepath,state);exclusive(token,dict(pid=child.pid,command=cmd))
            while child.poll()is None:
                fresh=descendants(row);owned=list({v['pid']:v for v in[*owned,*fresh]}.values());state['observed_owned']=owned
                total=sum(owned_RSS(v)for v in fresh if alive(v));state['peak_RSS_bytes']=max(state.get('peak_RSS_bytes',0),total)
                if total>=rss:raise MemoryError('Owned stage RSS aggregate')
                if time.monotonic()-start>wall:raise TimeoutError('Owned stage wall cap')
                preapproval(digest);guard();atomic(statepath,state);time.sleep(.25)
            code=child.wait();state.update(exit_code=code,cleanup_exit_code=code,reaped=True)
            if time.monotonic()-start>wall:raise TimeoutError('Owned stage exceeded wall before final reap')
            if code!=0:raise RuntimeError('Owned stage exit nonzero; no retry')
            for v in owned:absent(v['pid'])
            state.update(completed=True,outputs=[record(path)for path in outputs],result=record(outputs[0])if outputs else None)
    except BaseException as error:
        state['failure']=dict(type=type(error).__name__,message=str(error))
        if child is not None:cleanup(child,owned);state.update(exit_code=child.returncode,cleanup_exit_code=child.returncode,reaped=True)
        raise
    finally:state['wall_s']=time.monotonic()-start;state['observed_owned']=owned;atomic(statepath,state)
    return state

def commit(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('Changed deterministic existing commitment: '+str(path))
    else:exclusive(path,value)
    return record(path)

def controller_closure(base,phase,execution,digest,source,name,approval_sha):
    value=read(execution);receipt=read(LOCAL/'stages'/name/'execution.json');ids=[value['controller_pid'],*[w['pid']for w in value['workers']]]
    if not(value.get('completed')is True and value.get('terminal')is True and 'failure'not in value and value['controller_pid']==receipt['child_identity']['pid']):raise ValueError('Exact successful owned scientific controller')
    for pid in ids:absent(pid)
    out=dict(root_reaped=True,root_exit_code=0,root_owned_supervisor=True,external_subprocess_wait=True,all_owned_PID_and_groups_absent=True,protocol_SHA256=digest,source_SHA256=source,execution_SHA256=sha(execution),owned_pids=ids,operator_preapproval=record(LOCAL/'root-preapproval.json'),operator_stage_execution=record(LOCAL/'stages'/name/'execution.json'))
    path=base/(phase+'-root-closure.json'if base==physics.LOCAL else 'root-closure-'+phase+'.json');return commit(path,out)

def finite_closure(phase,digest,result):
    name='finite_'+phase;path=LOCAL/'stages'/name/'execution.json';v=read(path);ids=[r['pid']for r in v['observed_owned']]
    if v['child_identity']['pid']not in ids:raise ValueError('Own finite writer identity')
    for pid in ids:absent(pid)
    value=dict(root_reaped=True,root_exit_code=0,root_owned_supervisor=True,external_subprocess_wait=True,all_owned_PID_and_groups_absent=True,parent_protocol_SHA256=digest,source_SHA256=finite.sources(),owned_pids=ids,operator_preapproval=record(LOCAL/'root-preapproval.json'),operator_stage_execution=record(path),result=record(result))
    return commit(finite.LOCAL/f'root-closure-{phase}.json',value)

def checkpoint_admission(ns):
    e=noisy.completed_phase('checkpoint',ns);w=read(ROOT/e['workers'][0]['result']['path'])
    if not(w.get('completed')is True and len(w['trials'])==1 and w['trials'][0]['index']==0):raise ValueError('Only fixed trial0 cost checkpoint')
    trial=read(ROOT/w['trials'][0]['record']['path'])
    if trial.get('qualified')is not True or len(trial['posteriors'])!=8 or any(v.get('qualified')is not True for v in trial['posteriors']):raise ValueError('Trial0 all8 numerical qualification before continuation release')
    projection=3*max(read(ROOT/v['record']['path'])['wall_s']for v in trial['posteriors'])*8*11
    if not 0<=projection<=noisy.RESOURCES['evaluation_wall_s']:raise ValueError('Measured checkpoint projection exceeds unchanged worker budget')
    return projection

def pipeline(digest):
    p=preapproval(digest);guard();f=physics.validate(PHYSICS_SHA);physics.continuation_gate(f,PHYSICS_SHA);check=read(exact(p['checkpoint_independent_review']))
    if not(check.get('audit_passed')is True and check.get('all33_retained')is True and check.get('all11_reference_valid')is True):raise ValueError('Manually reaped/audited positive group0 before operator')
    def call(name,module,args,outputs,parent=PHYSICS_SHA):return stage(name,command(module,args,CAPS[name][0]),outputs,digest,parent_sha=parent)
    call('physics_remaining',physics.__name__,['run','--phase','remaining','--sha',PHYSICS_SHA],[physics.LOCAL/'execution-remaining.json'])
    controller_closure(physics.LOCAL,'remaining',physics.LOCAL/'execution-remaining.json',PHYSICS_SHA,f['source_SHA256'],'physics_remaining',digest)
    call('physics_finalize',physics.__name__,['finalize','--sha',PHYSICS_SHA],[physics.LOCAL/'summary-LOCAL.json'])
    call('physics_audit','experiments.audit_fixture_balanced_nh_catalog_v01',[],[physics.LOCAL/'independent-review.json',physics.LOCAL/'compact-summary.json'])
    physics.completed_catalog_gate() # ALL88valid, not merely technical/audit success.
    call('noisy_prepare',noisy.__name__,['prepare'],[noisy.LOCAL/'protocol.json'])
    n=read(noisy.LOCAL/'protocol.json');ns=sha(noisy.LOCAL/'protocol.json')
    if n['source_SHA256']!=noisy.sources()or n['resources']!=noisy.RESOURCES or n['origin']['prospective_design']!=noisy.design_gate():raise ValueError('Exact preauthorized noisy dynamic protocol')
    commit(noisy.LOCAL/'release.json',dict(approved=True,protocol_SHA256=ns,source_SHA256=n['source_SHA256'],origin=n['origin'],root_owned_operator_preapproval=record(LOCAL/'root-preapproval.json')));noisy.validate(ns)
    call('noisy_inputs',noisy.__name__,['inputs','--sha',ns],[noisy.LOCAL/'inputs.json'],ns)
    call('noisy_choices',noisy.__name__,['run','--phase','choices','--sha',ns],[noisy.LOCAL/'execution-choices.json'],ns)
    controller_closure(noisy.LOCAL,'choices',noisy.LOCAL/'execution-choices.json',ns,n['source_SHA256'],'noisy_choices',digest);seals=noisy.all_seals_gate(ns)
    commit(finite.LOCAL/'release.json',dict(approved=True,parent_protocol_SHA256=ns,source_SHA256=finite.sources(),resources=finite.LIMITS,root_owned_operator_preapproval=record(LOCAL/'root-preapproval.json')))
    call('finite_choose',finite.__name__,['choose','--parent-sha',ns],[finite.LOCAL/'finite-choice-seals.json'],ns);finite_closure('choose',ns,finite.LOCAL/'finite-choice-seals.json')
    call('noisy_checkpoint',noisy.__name__,['run','--phase','checkpoint','--sha',ns],[noisy.LOCAL/'execution-checkpoint.json'],ns)
    controller_closure(noisy.LOCAL,'checkpoint',noisy.LOCAL/'execution-checkpoint.json',ns,n['source_SHA256'],'noisy_checkpoint',digest)
    checkpoint_admission(ns)
    commit(noisy.LOCAL/'continuation-release.json',dict(approved=True,protocol_SHA256=ns,source_SHA256=n['source_SHA256'],checkpoint=record(noisy.LOCAL/'execution-checkpoint.json'),all32_choice_seals=seals,root_owned_operator_preapproval=record(LOCAL/'root-preapproval.json')));noisy.continuation_gate(ns)
    call('noisy_remaining',noisy.__name__,['run','--phase','remaining','--sha',ns],[noisy.LOCAL/'execution-remaining.json'],ns)
    controller_closure(noisy.LOCAL,'remaining',noisy.LOCAL/'execution-remaining.json',ns,n['source_SHA256'],'noisy_remaining',digest)
    call('noisy_summary',noisy.__name__,['summarize','--sha',ns],[noisy.LOCAL/'summary-LOCAL.json'],ns)
    call('noisy_audit',noisy_audit.__name__,[],[noisy.LOCAL/'independent-terminal-review.json'],ns)
    if read(noisy.LOCAL/'independent-terminal-review.json').get('audit_passed')is not True:raise ValueError('Independent continuous integrity audit required')
    call('finite_evaluate',finite.__name__,['evaluate','--parent-sha',ns],[finite.LOCAL/'evaluation-results-LOCAL.json',finite.LOCAL/'compact-summary.json'],ns);finite_closure('evaluate',ns,finite.LOCAL/'evaluation-results-LOCAL.json')
    call('finite_audit',FINITE_AUDIT[:-3].replace('/','.'),['--parent-sha',ns],[finite.LOCAL/'independent-terminal-review.json'],ns)
    if read(finite.LOCAL/'independent-terminal-review.json').get('audit_passed')is not True:raise ValueError('Independent finite integrity audit required before completion')
    commit(LOCAL/'completed.json',dict(completed=True,operator_preapproval=record(LOCAL/'root-preapproval.json'),stages={name:record(LOCAL/'stages'/name/'execution.json')for name in CAPS},noisy_audit=record(noisy.LOCAL/'independent-terminal-review.json'),finite_audit=record(finite.LOCAL/'independent-terminal-review.json')))

def run(digest):
    LOCAL.mkdir(parents=True,exist_ok=True)
    with(LOCAL/'operator.lock').open('a+')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);pipeline(digest)
def child(token,cmd):
    deadline=time.monotonic()+5
    while not Path(token).exists():
        if time.monotonic()>deadline:raise TimeoutError('Missing durable child handshake')
        time.sleep(.01)
    v=read(token)
    if v['pid']!=os.getpid()or v['command']!=cmd:raise ValueError('Exact owned launch handshake')
    os.execv(cmd[0]if cmd[0].startswith('/')else'/usr/bin/'+cmd[0],cmd)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','child']);ap.add_argument('--sha');ap.add_argument('--token');ap.add_argument('--command-json');a=ap.parse_args()
    if a.action=='child':child(a.token,json.loads(a.command_json))
    else:
        def stop(*_):raise InterruptedError('Owned operator interrupted; no automatic retry')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);run(a.sha)
