"""Operational-only GPU environment recovery; unchanged frozen native CLI."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from scripts.compositional_dynamics_v17 import native_phystwin_cohort_adapter as adapter
ROOT=adapter.ROOT;sha=adapter.sha;write=adapter.write
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
OUT=BASE/'native-recovery';PIPE=BASE/'pipeline'
PLAN=ROOT/'reports/compositional-dynamics-v17/native-cohort-recovery-plan.json'
ENV=dict(CUDA_VISIBLE_DEVICES='0',PYNPUT_BACKEND='dummy',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
PYTHON=ROOT/'runs/compositional-dynamics-v17/render-env/bin/python'

def original_failures():
    rows=[];evidence={}
    for i in range(4):
        world=f'world-{i}';folder=PIPE/'stages'/f'native-fit-{world}';status=json.loads((folder/'status.json').read_text());log=(folder/'output.log').read_text()
        if status.get('state')!='terminal'or status.get('completed')is not False or status.get('exit_code')!=1 or 'RuntimeError: Physical GPU0 only'not in log:raise ValueError('Wait for all four original prefit GPU-guard failures')
        old=BASE/'native'/world
        if list(old.rglob('*.pth'))or list(old.rglob('fit-complete.json'))or (old/'fit').exists():raise ValueError('Original optimization artifacts exist; no automatic retry')
        plan=PIPE/'native-plans'/f'{world}.json';p=adapter.guard(plan)
        request=json.loads((folder/'request.json').read_text())
        expected=[str(PYTHON),'-m','scripts.compositional_dynamics_v17.native_phystwin_cohort_adapter','fit','--plan',str(plan)]
        if status['request_sha256']!=sha(folder/'request.json')or request['command']!=expected or request['stage']!=f'native-fit-{world}':raise ValueError('Original request binding mismatch')
        if p['world_name']!=world or ROOT/p['output_directory']!=old:raise ValueError('Original identity mismatch')
        for f in [folder/'status.json',folder/'output.log',folder/'request.json',plan]:evidence[str(f.relative_to(ROOT))]=sha(f)
        for query in p['forecast_setups']:
            failed=old/'forecasts'/query['name']/'prediction-commit.json';c=json.loads(failed.read_text())
            if c['completed']is not False or c['world_id']!=world or c['role']!=query['name']or c['protocol_sha256']!=request['bindings']['protocol_sha']or c['method_freeze_sha256']!=request['bindings']['freeze_sha']:raise ValueError('All eight original failed commits required')
            evidence[str(failed.relative_to(ROOT))]=sha(failed)
        rows.append(dict(world_id=world,original_plan=str(plan.relative_to(ROOT)),original_output=str(old.relative_to(ROOT))))
    return rows,evidence

def prepare():
    if PLAN.exists()or OUT.exists():raise ValueError('Preserve recovery')
    rows,evidence=original_failures()
    # Copy plans byte-semantically, changing precisely one routing field.
    for row in rows:
        source=ROOT/row['original_plan'];p=json.loads(source.read_text());p['output_directory']=str((OUT/row['world_id']).relative_to(ROOT))
        dest=OUT/'plans'/f"{row['world_id']}.json";write(dest,p)
        row.update(recovery_plan=str(dest.relative_to(ROOT)),recovery_plan_sha256=sha(dest),recovery_output=p['output_directory'])
        if {k:v for k,v in p.items()if k!='output_directory'}!={k:v for k,v in json.loads(source.read_text()).items()if k!='output_directory'}:raise ValueError('Nonrouting modification')
    deadline=json.loads((PIPE/'stages/native-fit-world-0/request.json').read_text())['deadline_epoch']
    sources={str(Path(__file__).relative_to(ROOT)):sha(__file__),str((ROOT/'tests/test_native_cohort_recovery_v17.py').relative_to(ROOT)):sha(ROOT/'tests/test_native_cohort_recovery_v17.py')}
    write(PLAN,dict(scope='Operational environment/routing repair only; no fitting recipe, input, seed, prior, source or selection change. Original prefit failures retained.',worlds=rows,environment=ENV,deadline_epoch=deadline,fit_timeout_s=3700,replay_timeout_s=300,source_sha256=sources,original_evidence_sha256=evidence,expected_prediction_commits=8,target_access=False))

def guard():
    p=json.loads(PLAN.read_text())
    if p['environment']!=ENV:raise ValueError('Environment changed')
    for group in ['source_sha256','original_evidence_sha256']:
        for path,digest in p[group].items():
            if sha(ROOT/path)!=digest:raise ValueError('Recovery provenance changed '+path)
    original_failures()
    for row in p['worlds']:
        if sha(ROOT/row['recovery_plan'])!=row['recovery_plan_sha256']:raise ValueError('Recovery routing changed')
        adapter.guard(ROOT/row['recovery_plan'])
    return p

def stage(command,folder,timeout,deadline):
    folder.mkdir(parents=True,exist_ok=False);start=time.time();record=dict(command=command,environment=ENV,started_epoch=start,completed=False)
    process=None
    try:
        remaining=min(timeout,deadline-time.time())
        if remaining<=0:raise TimeoutError('Original cohort deadline exhausted')
        with (folder/'output.log').open('wb')as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**ENV),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=process.pid
            try:record['exit_code']=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait();raise TimeoutError('Recovery stage timeout')
        record['completed']=record['exit_code']==0
    except Exception as exc:record['error']=repr(exc)
    record.update(terminal=True,seconds=time.time()-start);write(folder/'complete.json',record);return record

def gpu_ready(deadline):
    while time.time()<deadline:
        text=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free,util=map(int,text.strip().split(','))
        if free>=6144 and util<=90:return dict(time_epoch=time.time(),gpu=0,free_MiB=free,utilization_percent=util)
        time.sleep(min(10,max(0,deadline-time.time())))
    raise TimeoutError('GPU0 headroom unavailable within original deadline')

def run():
    p=guard();marker=OUT/'run-start.json'
    with marker.open('x')as f:json.dump(dict(plan_sha256=sha(PLAN),started_epoch=time.time()),f)
    mapping=[]
    for row in p['worlds']:
        guard();world=row['world_id'];plan=ROOT/row['recovery_plan'];out=ROOT/row['recovery_output'];original=json.loads((ROOT/row['original_plan']).read_text())
        cmd=[str(PYTHON),'-m','scripts.compositional_dynamics_v17.native_phystwin_cohort_adapter']
        census=gpu_ready(p['deadline_epoch']);write(OUT/'resource-census'/f'{world}.json',census)
        fitted=stage(cmd+['fit','--plan',str(plan)],OUT/'stages'/f'fit-{world}',p['fit_timeout_s'],p['deadline_epoch'])
        for query in original['forecast_setups']:
            name=query['name'];result=dict(completed=False,error='Recovery fit unavailable')
            if fitted['completed']:result=stage(cmd+['predict','--plan',str(plan),'--query',name],OUT/'stages'/f'{world}-{name}',p['replay_timeout_s'],p['deadline_epoch'])
            folder=out/'forecasts'/name;folder.mkdir(parents=True,exist_ok=True)
            oldcommit=ROOT/row['original_output']/'forecasts'/name/'prediction-commit.json'
            if not oldcommit.exists():raise ValueError('Original failed prediction commitment required')
            old=json.loads(oldcommit.read_text())
            if old['completed']:raise ValueError('Never replace successful original prediction')
            commit=dict(world_id=world,role=name,completed=False,terminal=True,protocol_sha256=old['protocol_sha256'],method_freeze_sha256=old['method_freeze_sha256'],recovery_plan_sha256=sha(PLAN),native_world_plan_sha256=sha(plan),original_prediction_commit_sha256=sha(oldcommit),error=result.get('error','Recovery command failed'))
            complete=folder/'complete.json'
            if result['completed']and complete.exists():
                raw=json.loads(complete.read_text());array=folder/'prediction.npz'
                if raw['completed']and raw['plan_sha256']==sha(plan)and sha(array)==raw['prediction_sha256']:commit.update(completed=True,error=None,native_complete_sha256=sha(complete),prediction_sha256=sha(array),path=str(array.relative_to(ROOT)))
            dest=folder/'prediction-commit.json';write(dest,commit)
            mapping.append(dict(world_id=world,role=name,original_commit=str(oldcommit.relative_to(ROOT)),original_commit_sha256=sha(oldcommit),recovery_commit=str(dest.relative_to(ROOT)),recovery_commit_sha256=sha(dest),completed=commit['completed']))
            write(OUT/'progress.json',dict(plan_sha256=sha(PLAN),rows=mapping))
    guard();write(OUT/'recovery-manifest.json',dict(terminal=True,plan_sha256=sha(PLAN),expected=8,rows=mapping,target_access=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);args=a.parse_args();prepare()if args.stage=='prepare'else run()
