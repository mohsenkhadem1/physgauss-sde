"""One-factor development ablations; never score official test suffixes."""
import concurrent.futures,json,subprocess,threading,time
from pathlib import Path
from .development_run import ROOT,BASE,write,sha,export

CASES=['single_lift_sloth','single_lift_zebra']
RECIPES={
 'control':dict(nodes=120,iterations=50,fit_seconds=600,width=15.,floor='initial-minimum'),
 'longer_fit':dict(nodes=120,iterations=150,fit_seconds=1800,width=15.,floor='initial-minimum'),
 'mesh240':dict(nodes=240,iterations=50,fit_seconds=1800,width=15.,floor='initial-minimum'),
 'narrow_attachment':dict(nodes=120,iterations=50,fit_seconds=600,width=7.5,floor='initial-minimum'),
 'wide_attachment':dict(nodes=120,iterations=50,fit_seconds=600,width=30.,floor='initial-minimum'),
 'no_assumed_floor':dict(nodes=120,iterations=50,fit_seconds=600,width=15.,floor='disabled'),
}

def main():
    queue=BASE/'development-queue';queue.mkdir(parents=True,exist_ok=True)
    report=ROOT/'reports/real-transfer-v20'
    plan=dict(cases=CASES,recipes=RECIPES,method='neural',fit_frames=18,
              split='First 70% of official training prefix for fitting; remaining training prefix for validation. Official suffix excluded.',
              selection='One shared recipe chosen by equal-sequence validation tracking error, with complete rollout coverage; report all ablations.',
              limitation='All eight sequences historically exposed. This is development, not a sealed OOD claim.',
              sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('development_run.py'),ROOT/'scripts/real_transfer_v19/neural_fem.py',ROOT/'scripts/external_benchmark/fem.py']})
    path=report/'development-protocol.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Protocol changed')
    write(path,plan)
    # Avoid simultaneous workers racing to create the same prefix bundle.
    for case in CASES:
        for nodes in sorted({r['nodes'] for r in RECIPES.values()}):export(case,nodes)
    for case in CASES:
        for recipe in RECIPES:
            path=queue/f'{case}-{recipe}.json'
            if not path.exists():write(path,dict(case=case,recipe=recipe,state='pending'))
    lock=threading.Lock()
    def worker(device):
        while True:
            with lock:
                pending=[(p,json.loads(p.read_text())) for p in sorted(queue.glob('*.json')) if json.loads(p.read_text())['state']=='pending']
                if not pending:return
                path,job=pending[0];job.update(state='running',device=device,started=time.time());write(path,job)
            r=RECIPES[job['recipe']];limit=r['fit_seconds']+1800
            command=[str(ROOT/'runs/phystwin-env/bin/python'),'-u','-m','scripts.real_transfer_v20.development_run',
                     '--case',job['case'],'--label','development-'+job['recipe'],'--method','neural','--device',device,
                     '--nodes',str(r['nodes']),'--frames','18','--iterations',str(r['iterations']),
                     '--fit-seconds',str(r['fit_seconds']),'--total-seconds',str(limit),'--development',
                     '--attachment-width-mm',str(r['width']),'--floor-mode',r['floor']]
            with path.with_suffix('.log').open('w') as log:
                try:code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=limit+15).returncode
                except subprocess.TimeoutExpired:code=124
            job.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());write(path,job)
            print(job,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in [pool.submit(worker,f'cuda:{i}') for i in range(3)]:result.result()
    subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'-m','scripts.real_transfer_v20.score_development'],cwd=ROOT,check=True)

if __name__=='__main__':main()
