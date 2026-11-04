"""Wait for sealed corrected fits, then render recorded and unrecorded cameras."""
import concurrent.futures,json,os,subprocess,time,threading
from .campaign import ROOT,BASE,LABEL,CASES,write

def main():
    jobs=list(CASES);lock=threading.Lock()
    def worker(gpu):
        deadline=time.monotonic()+14400
        while True:
            with lock:
                if not jobs:return
                ready=[c for c in jobs if all((ROOT/'runs/real-transfer-v19/fits'/LABEL/c/m/'complete.json').exists() for m in ('neural','analytic'))]
                case=ready[0] if ready else None
                if case:jobs.remove(case)
            if case is None:
                if time.monotonic()>deadline:raise TimeoutError('Fits not complete within four hours')
                time.sleep(10);continue
            for module,tag in [('scripts.real_transfer_v20.render_audit','recorded'),('scripts.real_transfer_v22.novel_views','orbit')]:
                args=['--case',case,'--label',LABEL]+(['--nodes','240'] if tag=='recorded' else [])
                with (BASE/f'{case}-{tag}.log').open('w') as log:
                    try:r=subprocess.run([str(ROOT/'runs/deform360-reconstruction-env/bin/python'),'-u','-m',module,*args],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,timeout=1800);code=r.returncode
                    except subprocess.TimeoutExpired:code=124
                write(BASE/f'{case}-{tag}-status.json',dict(case=case,gpu=gpu,returncode=code));print(case,tag,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for f in [pool.submit(worker,g) for g in (0,1,2)]:f.result()
    subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'-m','scripts.real_transfer_v22.score'],cwd=ROOT,check=True)

if __name__=='__main__':main()
