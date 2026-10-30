"""Bounded postprocessing after all scheduled experiments; does not revise models."""
from pathlib import Path
import sys,time,subprocess,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.coupled_contact_v13.campaign import BASE,ROOT,remaining,write

def main():
    while not (BASE/'complete.json').exists():
        if remaining()<=0:raise RuntimeError('Campaign deadline reached before experiment completion')
        time.sleep(min(20,remaining()))
    python=ROOT/'runs/phystwin-env/bin/python'
    for name in ['nominal_baseline','report','verify','verify_nominal','baseline_audit','render']:
        log=BASE/f'finish-{name}.log'
        with log.open('w') as f:
            result=subprocess.run([str(python),str(ROOT/f'scripts/coupled_contact_v13/{name}.py')],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,remaining()),cwd=ROOT)
        if result.returncode:
            write(BASE/'postprocessing-failed.json',dict(stage=name,returncode=result.returncode,log=str(log)));raise RuntimeError('Postprocessing failed: '+name)
    write(BASE/'postprocessing-complete.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Numerical verification and rendering complete; independent visual audit and publication still required'))
if __name__=='__main__':main()
