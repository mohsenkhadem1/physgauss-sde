"""Resumable, dependency-aware frozen evaluation across CPU and GPU workers."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,ROOT,protocol,within_deadline
from scripts.position_ood.runtime import CONFIG
from scripts.position_ood.seal import verify
from scripts.phystwin.sequential_reference import write

def jobs(settings):
    cohorts=protocol();config=json.loads(CONFIG.read_text());items=[]
    def add(name,kind,script,args,result,deps,priority=1,environment='main'):
        items.append(dict(name=name,kind=kind,script=script,args=list(map(str,args)),result=str(result),deps=list(map(str,deps)),priority=priority,environment=environment))
    for split in ['calibration','test_id','test_material','test_environment']:
        count=len(cohorts[split])
        for i in range(count):
            reference=BASE/f'fields/{split}-{i}.npz';directory=BASE/f'inverse/{split}-{i}';common=['--split',split,'--worker',i,'--workers',count,'--iterations',settings['iterations']]
            methods=['extrapolated']
            if i in config['expensive_baseline_indices'].get(split,[]):methods+=['extrapolated_material_frame','extrapolated_nominal']
            elif split=='test_id' and i<9:methods+=['extrapolated_material_frame']
            for method in methods:
                coarse={'extrapolated':'joint','extrapolated_material_frame':'material_frame','extrapolated_nominal':'fixed_spatial_field'}[method];coarse_path=directory/f'{coarse}-n2-p3.json';fine_path=directory/f'{method}-n3-p3.json';priority=1 if method=='extrapolated' else (2 if method=='extrapolated_material_frame' and split=='test_id' and i<9 else 4)
                add(f'{split}-{i}-{coarse}','cpu','run_inverse.py',common+['--method',coarse,'--mesh',2],coarse_path,[reference],priority)
                add(f'{split}-{i}-{method}','gpu','run_inverse.py',common+['--method',method,'--mesh',3],fine_path,[reference,coarse_path],priority)
            u=directory/'extrapolated-n3-p3-uncertainty.npz'
            add(f'{split}-{i}-uncertainty','cpu','uncertainty.py',['--split',split,'--object',i,'--method','extrapolated','--mesh',3],u,[directory/'extrapolated-n3-p3.json'],2)
            if split!='calibration':
                for mode in ['id','all']:
                    add(f'{split}-{i}-neural-{mode}','gpu','predict_selected.py',['--split',split,'--object',i,'--mode',mode],BASE/f'neural-predictions/{split}-{i}-selected-{mode}.npz',[reference],2)
            if i in config['expensive_baseline_indices'].get(split,[]):
                pt=settings['phystwin'];add(f'{split}-{i}-phystwin','gpu','phystwin.py',['--split',split,'--object',i,*pt['args']],BASE/f'phystwin/{split}-{i}/{pt["tag"]}/summary.json',[reference],3,'native')
                mpm=settings['mpm'];add(f'{split}-{i}-mpm','gpu','fit_mpm.py',['--split',split,'--object',i,*mpm['args']],BASE/f'{mpm["tag"]}/{split}-{i}/summary.json',[reference],3,'native')
                if i in config['secondary_mpm_field_indices'].get(split,[]):add(f'{split}-{i}-mpm-field','gpu','fit_mpm.py',['--split',split,'--object',i,*mpm['args'],'--spatial-field'],BASE/f'{mpm["tag"]}-field/{split}-{i}/summary.json',[reference],5,'native')
    cal=BASE/'calibration/extrapolated-n3.json';deps=[BASE/f'inverse/calibration-{i}/extrapolated-n3-p3-uncertainty.npz' for i in range(len(cohorts['calibration']))]
    add('calibrate','cpu','calibration.py',['--split','calibration'],cal,deps,2)
    from scripts.position_ood.study import OUT
    for split in ['test_id','test_material','test_environment']:
        deps=[cal]+[BASE/f'inverse/{split}-{i}/extrapolated-n3-p3-uncertainty.npz' for i in range(len(cohorts[split]))]
        add(f'{split}-coverage','cpu','calibration.py',['--split',split],OUT/f'{split}-extrapolated-coverage.json',deps,3)
    return sorted(items,key=lambda r:(r['priority'],r['name']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['cpu','gpu'],required=True);p.add_argument('--worker',type=int,default=0);a=p.parse_args();seal=verify();tasks=[]
    for job in jobs(seal['settings']):
        if job['kind']==a.kind:tasks.append(job)
        elif a.kind=='cpu' and job['script']=='run_inverse.py':
            # Spare CPU workers may take fine solves after their dependency work.
            tasks.append(dict(job,priority=job['priority']+10))
    tasks.sort(key=lambda r:(r['priority'],r['name']));directory=BASE/'jobs';directory.mkdir(exist_ok=True)
    while True:
        within_deadline();verify();pending=0;progress=False
        for job in tasks:
            output=Path(job['result']);status=directory/(job['name']+'.json')
            if output.exists():continue
            if status.exists() and json.loads(status.read_text()).get('state')=='failed':continue
            pending+=1
            if not all(Path(p).exists() for p in job['deps']):continue
            with (directory/(job['name']+'.lock')).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if output.exists():continue
                env=os.environ.copy();env['POSITION_INVERSE_DEVICE']='cpu' if a.kind=='cpu' else 'cuda:0'
                python=ROOT/('runs/phystwin-env/bin/python' if job['environment']=='native' else '.venv/bin/python');command=[str(python),str(ROOT/'scripts/position_ood'/job['script']),*job['args']];start=time.time();write(status,dict(state='running',worker=a.worker,kind=a.kind,job=job,start=start));print('JOB_START',job['name'],flush=True)
                with (directory/(job['name']+'.log')).open('a') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                success=result.returncode==0 and output.exists();write(status,dict(state='complete' if success else 'failed',worker=a.worker,kind=a.kind,job=job,returncode=result.returncode,elapsed_seconds=time.time()-start));print('JOB_DONE',job['name'],success,flush=True);progress=True
                break # Reconsider newly available higher-priority dependencies.
        if pending==0:break
        if not progress:time.sleep(20)
    write(directory/f'worker-{a.kind}-{a.worker}-done.json',dict(time=time.time(),scope='All eligible jobs complete or explicitly failed; inspect per-job states and declared completeness.'))
if __name__=='__main__':main()
