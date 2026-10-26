"""Durable, explicitly versioned resource-only and FP-verifier recovery.

No old source, choices, noise or completed trial is overwritten or replayed.
All scientific gates still apply; this supervisor owns only its descendants.
"""
import argparse,fcntl,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from experiments import run_noisy_poke_continuation_v02 as continuous
from experiments import run_matched_finite_poke_control_v02 as finite
ROOT=continuous.ROOT;LOCAL=ROOT/'runs/operator-dev/noisy-poke-recovery-operator-v02'
SOURCE='experiments/supervise_noisy_poke_recovery_v02.py';TEST='tests/test_supervise_noisy_poke_recovery_v02.py'
GiB=1024**3
CAPS=dict(remaining=[18,18060,28*GiB],summary=[18,300,4*GiB],continuous_audit=[16,3600,4*GiB],finite_evaluate=[17,1800,4*GiB],finite_audit=[16,1800,4*GiB])
OPERATOR_LIMITS=dict(CPU=19,GPU=False,threads=1,RSS_bytes=4*GiB,RAM_reserve_bytes=30*GiB)
sha,read=continuous.sha,continuous.read
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
def pid_directory_exists(pid):return(Path('/proc')/str(pid)).exists()
def retirement_proof(row):
    """Only absence/reuse retires an old nonleader; zombies/errors stay tracked."""
    try:
        current=identity(row['pid'])
        if current is None:return 'PID_absent'if not pid_directory_exists(row['pid'])else None
        if current['starttime']!=row['starttime']:return 'starttime_changed'
    except(OSError,ValueError,IndexError,KeyError):pass
    return None

def retirement_summary():
    return dict(count=0,SHA256_chain='0'*64,PID_absent=0,starttime_changed=0,algorithm='SHA256(previous_digest_bytes + canonical_identity_and_proof_JSON_UTF8)')
def compact_owned(previous,fresh,retired):
    """Retain all observed leaders and live/transitional identities, not tombstones.

    Identity keys include starttime, so PID reuse cannot replace old leader proof.
    Retire only proven-dead nonleaders into a constant-size count/hash chain.
    """
    merged={}
    for row in[*previous,*fresh]:
        key=(row['pid'],row['starttime'])
        if key not in merged or merged[key]['PGID']!=row['pid']:merged[key]=row
    kept=[];summary=dict(retired)
    for key,row in sorted(merged.items()):
        proof=None if row['PGID']==row['pid']else retirement_proof(row)
        if proof is None:kept.append(row);continue
        payload=json.dumps(dict(identity=row,proof=proof),sort_keys=True,separators=(',',':')).encode()
        summary['SHA256_chain']=hashlib.sha256(bytes.fromhex(summary['SHA256_chain'])+payload).hexdigest();summary['count']+=1;summary[proof]+=1
    return kept,summary

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
    return {**continuous.sources(),**finite.sources(),**{name:sha(ROOT/name)for name in(SOURCE,TEST)}}

def preapproval(digest):
    path=LOCAL/'root-preapproval.json';p=read(path)
    if not(sha(path)==digest and p.get('approved')is True and p.get('schema')=='noisy-poke-recovery-operator-preapproval-v02'and p.get('source_SHA256')==sources()and p.get('stage_caps')==CAPS and p.get('operator_resources')==OPERATOR_LIMITS):raise ValueError('Exact recovery source/resources/preapproval')
    exact(p['continuous_protocol']);exact(p['original_operator_reap'])
    if p['continuous_protocol']!=record(continuous.LOCAL/'protocol.json'):raise ValueError('Exact resource-only protocol')
    old=read(exact(p['original_operator_reap']))
    if not(old.get('reaped')is True and old.get('exit_code')==1 and old.get('all_recorded_owned_PID_groups_absent')is True):raise ValueError('Original failed operator honestly reaped/closed')
    return p

def guard():
    if set(os.sched_getaffinity(0))!={19}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Operator CPU19/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=OPERATOR_LIMITS['RSS_bytes']:raise MemoryError('Own operator4GiB')
    if continuous.available_memory()<OPERATOR_LIMITS['RAM_reserve_bytes']:raise MemoryError('Operator30GiB reserve')

# Generic lifecycle utility uses the unchanged available-memory API.
noisy=continuous
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
    folder.mkdir(parents=True,exist_ok=True);state=dict(phase=name,parent_protocol_SHA256=parent_sha,completed=False,reaped=False,command=cmd,source_SHA256=sources(),operator_preapproval=record(LOCAL/'root-preapproval.json'),resources=dict(CPU=cpu,wall_s=wall,RSS_bytes=rss),outputs=[],observed_owned=[],observed_owned_scope='All observed group leaders and live/transitional descendants; proven-dead nonleaders retired into count/hash chain',retired_nonleaders=retirement_summary());exclusive(statepath,state)
    token=folder/'start.json';bootstrap=command('experiments.supervise_noisy_poke_recovery_v02',['child','--token',str(token),'--command-json',json.dumps(cmd)],cpu);child=None;owned=[];start=time.monotonic()
    try:
        with(folder/'stdout-LOCAL.log').open('xb')as log:
            child=subprocess.Popen(bootstrap,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row=identity(child.pid)
            if not row or row['PGID']!=child.pid:raise RuntimeError('Exact owned child session before handshake')
            owned=[row];state['child_identity']=row;atomic(statepath,state);exclusive(token,dict(pid=child.pid,command=cmd))
            while child.poll()is None:
                fresh=descendants(row);owned,state['retired_nonleaders']=compact_owned(owned,fresh,state['retired_nonleaders']);state['observed_owned']=owned
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
    finally:
        owned,state['retired_nonleaders']=compact_owned(owned,[],state['retired_nonleaders'])
        state['wall_s']=time.monotonic()-start;state['observed_owned']=owned;atomic(statepath,state)
    return state

def commit(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('Changed deterministic existing commitment: '+str(path))
    else:exclusive(path,value)
    return record(path)

def pipeline(digest):
    p=preapproval(digest);guard();parent=p['parent_protocol_SHA256'];approval=record(LOCAL/'root-preapproval.json')
    def call(name,module,args,outputs):return stage(name,command(module,args,CAPS[name][0]),outputs,digest,parent_sha=parent)
    commit(continuous.LOCAL/'release.json',dict(approved=True,protocol=record(continuous.LOCAL/'protocol.json'),source_SHA256=continuous.sources(),resources=continuous.RESOURCES,parent_protocol_SHA256=parent,root_owned_operator_preapproval=approval))
    continuous.continuation_gate(parent)
    call('remaining',continuous.__name__,['run','--sha',parent],[continuous.LOCAL/'execution-remaining.json'])
    e=read(continuous.LOCAL/'execution-remaining.json');receipt=read(LOCAL/'stages/remaining/execution.json');ids=[e['controller_pid'],*[v['pid']for v in e['workers']]]
    if not(e.get('completed')is True and e.get('terminal')is True and 'failure'not in e and e['controller_pid']==receipt['child_identity']['pid']):raise ValueError('Exact successful new controller')
    for pid in ids:absent(pid)
    commit(continuous.LOCAL/'root-closure-remaining.json',dict(root_reaped=True,root_exit_code=0,root_owned_supervisor=True,external_subprocess_wait=True,all_owned_PID_and_groups_absent=True,execution=record(continuous.LOCAL/'execution-remaining.json'),protocol=record(continuous.LOCAL/'protocol.json'),source_SHA256=continuous.sources(),owned_pids=ids,operator_preapproval=approval,operator_stage_execution=record(LOCAL/'stages/remaining/execution.json')))
    call('summary',continuous.__name__,['summarize','--sha',parent],[continuous.LOCAL/'summary-LOCAL.json'])
    call('continuous_audit','experiments.audit_noisy_poke_continuation_v02',['--sha',parent],[continuous.LOCAL/'independent-terminal-review.json'])
    if read(continuous.LOCAL/'independent-terminal-review.json').get('audit_passed')is not True:raise ValueError('Full independent continuous integrity audit required')
    # Finite release is bound to the ORIGINAL pre-outcome64 sealed choices.
    _,_,_,origin=finite.original_choices_gate(parent)
    commit(finite.LOCAL/'release.json',dict(schema='matched-finite-poke-recovery-release-v02',approved=True,parent_protocol_SHA256=parent,source_SHA256=finite.sources(),resources=finite.LIMITS,origin=origin,verifier_rule=finite.PRIOR_RULE,continuous_audit=record(continuous.LOCAL/'independent-terminal-review.json'),root_owned_operator_preapproval=approval))
    call('finite_evaluate',finite.__name__,['evaluate','--parent-sha',parent],[finite.LOCAL/'evaluation-results-LOCAL.json',finite.LOCAL/'compact-summary.json'])
    path=LOCAL/'stages/finite_evaluate/execution.json';v=read(path);ids=[r['pid']for r in v['observed_owned']]
    for pid in ids:absent(pid)
    commit(finite.LOCAL/'root-closure-evaluate.json',dict(root_reaped=True,root_exit_code=0,root_owned_supervisor=True,external_subprocess_wait=True,all_owned_PID_and_groups_absent=True,parent_protocol_SHA256=parent,source_SHA256=finite.sources(),owned_pids=ids,operator_preapproval=approval,operator_stage_execution=record(path),result=record(finite.LOCAL/'evaluation-results-LOCAL.json')))
    call('finite_audit','experiments.audit_matched_finite_poke_control_v02',['--parent-sha',parent],[finite.LOCAL/'independent-terminal-review.json'])
    if read(finite.LOCAL/'independent-terminal-review.json').get('audit_passed')is not True:raise ValueError('Independent finite audit required')
    commit(LOCAL/'completed.json',dict(completed=True,operator_preapproval=approval,stages={name:record(LOCAL/'stages'/name/'execution.json')for name in CAPS},noisy_audit=record(continuous.LOCAL/'independent-terminal-review.json'),finite_audit=record(finite.LOCAL/'independent-terminal-review.json')))

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
    if v['pid']!=os.getpid()or v['command']!=cmd:raise ValueError('Exact owned child handshake')
    os.execv(cmd[0]if cmd[0].startswith('/')else'/usr/bin/'+cmd[0],cmd)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','child']);ap.add_argument('--sha');ap.add_argument('--token');ap.add_argument('--command-json');a=ap.parse_args()
    if a.action=='child':child(a.token,json.loads(a.command_json))
    else:
        def stop(*_):raise InterruptedError('Owned recovery interrupted; no partial retry')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);run(a.sha)
