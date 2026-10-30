"""Bounded two-worker export/forecast queue; no evaluator import or scoring."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import concurrent.futures,json,subprocess,sys,time,os,threading
from scripts.prefix_ridge_switch_v13.run import ROOT,BASE,REPORT,PARENT,read,write,sha,guard,check_inputs
import numpy as np

def fallback(case,error):
    source=PARENT/'model-selection-evaluation'/case['cohort']/case['name']/'predictions.npz'
    check_inputs([source])
    with np.load(source,allow_pickle=False) as f:old=f['model_selection_selected'].copy()
    out=BASE/'forecasts'/case['cohort']/case['name'];out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'prediction.npz',predicted=old)
    write(out/'metadata.json',{'case':case,'switched':False,'reason':'Unavailable bounded attempt; exact archived fallback','attempt_error':error,'input_sha256':{str(source.relative_to(ROOT)):sha(source)},'prediction_sha256':sha(out/'prediction.npz'),'scope':'Exploratory fallback; no suffix target used'})

def attempt(case,index,deadline,p):
    state='complete';error=None
    try:
        remaining=min(p['compute_bound']['per_case_timeout_seconds'],deadline-time.time())
        if remaining<=0:raise TimeoutError('Total/deadline budget exhausted before attempt')
        end=time.time()+remaining
        for stage in ('export','forecast'):
            result=subprocess.run([sys.executable,'-m','scripts.prefix_ridge_switch_v13.run',stage,'--index',str(index)],cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=max(.01,end-time.time()))
            if result.returncode:raise RuntimeError((result.stderr+result.stdout)[-4000:])
    except subprocess.TimeoutExpired:error='Per-case timeout';state='timeout'
    except TimeoutError as e:error=str(e);state='timeout'
    except Exception as e:error=repr(e);state='failed'
    if state!='complete':fallback(case,error)
    folder=BASE/'forecasts'/case['cohort']/case['name']
    return {'case':case,'state':state,'prediction_sha256':sha(folder/'prediction.npz'),'metadata_sha256':sha(folder/'metadata.json'),'error':error}

def main():
    p,remaining=guard();BASE.mkdir(parents=True,exist_ok=True);started=BASE/'batch-start.json'
    if not started.exists():write(started,{'started_epoch':time.time(),'protocol_sha256':sha(REPORT/'protocol.json'),'method_freeze_sha256':sha(REPORT/'method-freeze.json')})
    start=read(started)
    if start['protocol_sha256']!=sha(REPORT/'protocol.json') or start['method_freeze_sha256']!=sha(REPORT/'method-freeze.json'):raise RuntimeError('Batch start provenance differs')
    deadline=min(start['started_epoch']+p['compute_bound']['total_execution_budget_seconds'],time.time()+remaining-10)
    manifest=BASE/'attempts.json';records={}
    if manifest.exists():
        old=read(manifest)
        if old['protocol_sha256']!=sha(REPORT/'protocol.json') or old['method_freeze_sha256']!=sha(REPORT/'method-freeze.json'):raise RuntimeError('Prior manifest mismatch')
        declared={c['name']:c for c in p['cases']}
        if len({r['case']['name'] for r in old['cases']})!=len(old['cases']):raise RuntimeError('Duplicate prior attempts')
        for r in old['cases']:
            if declared.get(r['case']['name'])!=r['case'] or r['state'] not in ('complete','failed','timeout'):raise RuntimeError('Undeclared/nonterminal old attempt')
            folder=BASE/'forecasts'/r['case']['cohort']/r['case']['name']
            if sha(folder/'prediction.npz')!=r['prediction_sha256'] or sha(folder/'metadata.json')!=r['metadata_sha256']:raise RuntimeError('Existing output changed')
            records[r['case']['name']]=r
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(2,p['compute_bound']['workers_max'])) as pool:
        futures={pool.submit(attempt,c,i,deadline,p):c for i,c in enumerate(p['cases']) if c['name'] not in records}
        for future in concurrent.futures.as_completed(futures):
            r=future.result();records[r['case']['name']]=r
            write(manifest,{'protocol_sha256':sha(REPORT/'protocol.json'),'method_freeze_sha256':sha(REPORT/'method-freeze.json'),'cases':[records[c['name']] for c in p['cases'] if c['name'] in records],'complete':len(records)==148})
if __name__=='__main__':main()
