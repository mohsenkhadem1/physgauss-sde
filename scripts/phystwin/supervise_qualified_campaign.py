"""Resumable, fail-closed v05 campaign runner. Raw outputs remain local."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.qualified_reference import BASE,ROOT,write_json

STATUS=BASE/'status'
PYTHON=ROOT/'.venv/bin/python'


def status(stage,**extra):
    row=dict(stage=stage,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    write_json(STATUS/'supervisor.json',row);print(json.dumps(row),flush=True)


def launch(script,args,gpu,label):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTORCH_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
    log=open(STATUS/(label+'.log'),'ab')
    process=subprocess.Popen([str(PYTHON),str(ROOT/'scripts/phystwin'/script),*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    log.close();return process


def wait_all(processes):
    codes=[p.wait() for p in processes]
    if any(codes):raise RuntimeError(f'Phase failed with exit codes {codes}; inspect stage logs. No downstream evaluation was run.')


def reference(split,mesh=4):
    workers=6 if split=='bank' else 3
    label=f'{split}-n{mesh}';status(label,workers=workers)
    wait_all([launch('qualified_reference.py',['--split',split,'--bank-mesh',mesh,'--workers',workers,'--worker',i,'--keep-going'],i%3,label+f'-{i}') for i in range(workers)])


def policy(stage):
    status('inference-'+stage);wait_all([launch('qualified_policy.py',['--stage',stage],0,'inference-'+stage)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pids',nargs='*',type=int,default=[]);a=p.parse_args()
    STATUS.mkdir(parents=True,exist_ok=True)
    lock=open(STATUS/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # Watch the current process identity, so a recycled PID cannot stall us.
        identities={pid:Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[19] for pid in a.wait_pids if Path(f'/proc/{pid}/stat').exists()}
        status('waiting-for-existing-checkpointed-jobs',pids=list(identities))
        while identities:
            for pid,identity in list(identities.items()):
                try:fields=Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()
                except FileNotFoundError:del identities[pid];continue
                if fields[19]!=identity or fields[0]=='Z':del identities[pid]
            if identities:time.sleep(5)
        # This verifies completion and retries only missing checkpointed fields.
        reference('development')
        reference('bank',8)
        status('select-surrogate')
        chosen=launch('qualified_policy.py',['--stage','select'],0,'select-surrogate').wait()
        if chosen:
            progress=BASE/'analysis/model-selection-progress.json'
            if not progress.exists() or json.loads(progress.read_text()).get('required_bank_mesh')!=12:
                raise RuntimeError('Surrogate selection failed; no calibration or test inference started')
            reference('bank',12);policy('select')
        policy('development')
        reference('calibration');policy('calibrate')
        reference('test');policy('test')
        status('oracle-audit');wait_all([launch('audit_qualified_oracle.py',[],0,'oracle-audit')])
        status('publish-local-report');wait_all([launch('summarize_qualified_support.py',[],0,'report')])
        status('render-held-force-video');wait_all([launch('render_qualified_predictions.py',[],0,'video')])
        status('complete',report='reports/phystwin-qualified-support-v05/REPORT.md')
    except BaseException as e:
        status('stopped-on-error',error=str(e));traceback.print_exc();raise

if __name__=='__main__':main()
