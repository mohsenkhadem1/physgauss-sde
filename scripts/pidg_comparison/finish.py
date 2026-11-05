"""Crash-resilient evaluator: validate all runs, seal choice, then open test."""
import json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-comparison-v01';PYTHON=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python'
NAMES=['full-pidg','full-ours-001','full-ours-010']
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def call(args,log):
    with log.open('w') as f:subprocess.run([str(PYTHON),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    started=time.time();ready=set()
    try:
        while len(ready)<3:
            if time.time()-started>4*3600:raise TimeoutError('Four-hour training watch limit reached; existing jobs left intact')
            for gpu,name in enumerate(NAMES):
                run=BASE/name;p=run/'run.json'
                if not p.exists() or name in ready:continue
                try:meta=json.loads(p.read_text())
                except json.JSONDecodeError:continue
                if meta['status']=='failed':raise RuntimeError(f'{name}: {meta.get("error")}')
                if meta['status']=='training_complete':
                    call(['scripts/pidg_comparison/evaluate.py','--run',str(run),'--gpu',str(gpu)],BASE/f'{name}-validation.log');ready.add(name)
            write(BASE/'status.json',{'phase':'training/validation','validation_complete':sorted(ready),'started':started,'updated':time.time()});time.sleep(20)
        candidates=[]
        for name in NAMES[1:]:
            metrics=json.loads((BASE/name/'validation/metrics.json').read_text())['metrics'];candidates.append({'run':name,'validation_foreground_psnr_db':metrics['foreground_psnr_db']})
        best=max(candidates,key=lambda r:r['validation_foreground_psnr_db'])['run']
        selection={'criterion':'Highest mean foreground PSNR on the 11 validation frames; no test-based tuning','candidates':candidates,'chosen':best,'test_runs':[str((BASE/NAMES[0]).resolve()),str((BASE/best).resolve())],'sealed_at':time.time()};write(BASE/'selection.json',selection)
        for gpu,run in enumerate(selection['test_runs']):call(['scripts/pidg_comparison/evaluate.py','--run',run,'--split','test','--gpu',str(gpu)],BASE/f'test-{gpu}.log')
        for gpu,run in enumerate(selection['test_runs']):call(['scripts/pidg_comparison/evaluate.py','--run',run,'--split','test','--render-width','640','--gpu',str(gpu)],BASE/f'test-highres-{gpu}.log')
        call(['scripts/pidg_comparison/compare.py','--root',str(BASE)],BASE/'comparison.log')
        write(BASE/'status.json',{'phase':'complete','selection':selection,'report':f'reports/{BASE.name}/results.json','finished':time.time()})
    except BaseException as e:
        write(BASE/'status.json',{'phase':'failed','error':repr(e),'finished':time.time()});raise
if __name__=='__main__':main()
