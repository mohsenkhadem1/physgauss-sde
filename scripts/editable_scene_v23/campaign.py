import concurrent.futures,subprocess,os,json,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/editable-scene-v23'

def main():
    jobs=[(kind,E,.00005)for kind in ('lift_release','drop','force_pulse')for E in (6000,12000,24000)]+[('drop',12000,.000025)]
    lock=threading.Lock();status=[]
    def worker(gpu):
        while True:
            with lock:
                if not jobs:return
                kind,E,dt=jobs.pop(0)
            key=f'{kind}-{E}-dt{dt:g}'
            with (BASE/f'{key}.log').open('w')as log:
                try:r=subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'-u','-m','scripts.editable_scene_v23.pilot','--kind',kind,'--young',str(E),'--dt',str(dt),'--device',f'cuda:{gpu}'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=920);code=r.returncode
                except subprocess.TimeoutExpired:code=124
            with lock:
                status.append(dict(key=key,gpu=gpu,returncode=code));(BASE/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(key,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
        for f in [pool.submit(worker,g)for g in (0,1,2)]:f.result()

if __name__=='__main__':main()
