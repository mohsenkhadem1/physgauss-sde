"""Uniform budget-exhausted prefix checkpoint continuation; no future scoring."""
import sys,json,time,shutil,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import *
from scripts.contact_benchmark_v12.data import load

def repair(path,job):
    data=load(job['spec'],job.get('validation',False));tag='validation' if job.get('validation') else 'full';out=BASE/'base'/tag/data['name']
    saved=np.load(out/'best.npz');progress=json.loads((out/'progress.json').read_text());best=progress['best']
    np.testing.assert_array_equal(saved['theta'],best['theta']);np.testing.assert_array_equal(saved['nodes'],data['nodes'])
    np.testing.assert_array_equal(saved['selected_frames'],np.unique(np.linspace(0,data['train_end']-1,36,dtype=int)))
    assert progress['fit_end']==data['train_end'] and np.isfinite(saved['theta']).all() and np.isfinite(best['loss']) and best['loss']<1000
    assert (saved['selected_frames']<data['train_end']).all()
    snapshot=BASE/'timeout-records'/f'{job["index"]:03d}';snapshot.mkdir(parents=True,exist_ok=True)
    write_json(snapshot/'original-job.json',job)
    for name in ['best.npz','progress.json']:shutil.copy2(out/name,snapshot/name)
    parameter_sha=sha256(out/'best.npz');job['timeout_repair_attempted']=True;job['state']='repairing';write_json(path,job)
    settings=dict(field=True,contact=True,nonconvex=data['dataset']=='PhysTwin',materials=6 if data['dataset']=='PhysTwin' else 1)
    if (out/'fit.json').exists():
        fit=json.loads((out/'fit.json').read_text());np.testing.assert_array_equal(fit['best']['theta'],saved['theta'])
        shutil.copy2(out/'fit.json',snapshot/'completed-fit.json')
        job['repair_phase']='prediction after completed fit'
    else:
        fit=dict(state='budget_exhausted_valid_checkpoint',name=data['name'],dataset=data['dataset'],fit_end=data['train_end'],selected_frames=saved['selected_frames'].tolist(),best=best,
            optimizer=dict(success=False,message='2400 second fitting-job budget exhausted; unchanged best finite prefix checkpoint retained',iterations=None),history=[best],failed_trials=progress.get('failures',[]),elapsed_seconds=job['elapsed_seconds'],**settings,
            timeout_checkpoint_sha256=parameter_sha,timeout_original_job=str(snapshot/'original-job.json'),interpretation='Point estimate with unfinished optimizer; not a convergence claim. No future-error selection.')
        write_json(out/'fit.json',fit);job['repair_phase']='unfinished optimization checkpoint'
    start=time.time();log=snapshot/'rollout.log'
    with open(log,'w') as handle:
        try:code=subprocess.run(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,timeout=1800).returncode
        except subprocess.TimeoutExpired:code=124
    assert sha256(out/'best.npz')==parameter_sha
    job.update(state='complete' if code==0 else 'failed',repair_returncode=code,repair_elapsed_seconds=time.time()-start,repair_log=str(log),repair_checkpoint_sha256=parameter_sha,optimizer_converged=fit['optimizer']['success'])
    write_json(path,job);print('TIMEOUT_CONTINUATION',job['index'],job['state'],flush=True)

def main(watch):
    while True:
        check_deadline();jobs=[]
        for path in sorted((BASE/'jobs').glob('*.json')):
            j=json.loads(path.read_text());jobs.append(j)
            if j['kind']=='base' and j['state']=='failed' and j.get('returncode')==124 and not j.get('timeout_repair_attempted'):
                try:repair(path,j)
                except Exception as error:
                    j.update(state='failed',timeout_repair_attempted=True,repair_validation_error=str(error));write_json(path,j);print('REPAIR_REJECTED',j['index'],str(error),flush=True)
        live=any(j['kind']=='base' and j['state'] in ['pending','running','repairing'] for j in jobs)
        if not watch or not live:break
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();main(a.watch)
