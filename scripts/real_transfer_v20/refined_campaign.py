"""Frozen development-selected recipe, matched neural/analytic rebenchmark."""
import concurrent.futures,json,subprocess,threading,time
from .development_run import ROOT,BASE,write,sha
from scripts.real_transfer_v19.phystwin_run import export
from scripts.real_transfer_v19.phystwin_campaign import CASES

LABEL='v20-selected-mesh240'

def main():
    selection=ROOT/'reports/real-transfer-v20/development-scores.json'
    decision=json.loads(selection.read_text())
    if not decision['complete'] or decision['selected_recipe']!='mesh240':raise ValueError('Unexpected development decision')
    protocol=dict(label=LABEL,nodes=240,iterations=50,fit_frames=18,fit_seconds=1800,
        selection_sha256=sha(selection),methods=['neural','analytic'],cases=CASES,
        scope='Frozen development-selected recipe on historically exposed eight sequences; not a sealed test.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/real_transfer_v19/phystwin_run.py',ROOT/'scripts/real_transfer_v19/neural_fem.py',ROOT/'scripts/external_benchmark/fem.py']})
    target=ROOT/'reports/real-transfer-v20/refined-protocol.json'
    if target.exists() and json.loads(target.read_text())!=protocol:raise ValueError('Protocol changed')
    write(target,protocol)
    ordered=['single_lift_sloth','single_lift_zebra']+[c for c in CASES if c not in ('single_lift_sloth','single_lift_zebra')]
    for case in ordered:export(case,240)
    jobs=[(case,method) for case in ordered for method in ('neural','analytic')];lock=threading.Lock()
    def worker(device):
        while True:
            with lock:
                if not jobs:return
                case,method=jobs.pop(0)
            out=ROOT/'runs/real-transfer-v19/fits'/LABEL/case/method
            if (out/'complete.json').exists():continue
            log=BASE/f'refined-{case}-{method}.log'
            with log.open('w') as f:
                try:code=subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),'-u','-m','scripts.real_transfer_v19.phystwin_run',
                    '--case',case,'--label',LABEL,'--method',method,'--device',device,'--nodes','240','--frames','18',
                    '--iterations','50','--fit-seconds','1800','--total-seconds','3600'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=3615).returncode
                except subprocess.TimeoutExpired:code=124
            write(BASE/f'refined-{case}-{method}-status.json',dict(returncode=code,finished=time.time()))
            print(case,method,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in [pool.submit(worker,f'cuda:{i}') for i in range(3)]:result.result()

if __name__=='__main__':
    from pathlib import Path
    main()
