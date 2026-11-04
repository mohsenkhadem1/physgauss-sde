"""Run the fixed-frame common-renderer audit as refined predictions finish."""
import json,os,subprocess,time
from .development_run import ROOT,BASE,write
from .refined_campaign import LABEL
from scripts.real_transfer_v19.phystwin_campaign import CASES

def main():
    pending=list(CASES);started=time.monotonic();statuses=[]
    while pending and time.monotonic()-started<10800:
        ready=[case for case in pending if all((ROOT/'runs/real-transfer-v19/fits'/LABEL/case/m/'complete.json').exists() for m in ('neural','analytic'))]
        if not ready:time.sleep(5);continue
        case=ready[0];pending.remove(case)
        target=BASE/'render-audit'/LABEL/case/'scores.json'
        if target.exists():statuses.append(dict(case=case,state='already_complete'));continue
        with (BASE/f'render-{case}.log').open('w') as log:
            try:code=subprocess.run([str(ROOT/'runs/deform360-reconstruction-env/bin/python'),'-u','-m','scripts.real_transfer_v20.render_audit','--case',case],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),
                stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code=124
        statuses.append(dict(case=case,state='complete' if code==0 else 'failed',returncode=code))
        write(BASE/'render-campaign-status.json',dict(statuses=statuses,pending=pending));print(case,code,flush=True)
    rows=[]
    for case in CASES:
        path=BASE/'render-audit'/LABEL/case/'scores.json'
        if path.exists():rows.append(json.loads(path.read_text()))
    write(ROOT/'reports/real-transfer-v20/render-scores.json',dict(cases=rows,complete=len(rows)==len(CASES),
        scope='Three predetermined future frames per sequence, all three initial appearance cameras. Common frozen Gaussian renderer; not full-video or unseen-camera evaluation. All methods receive the same appearance asset, occlusion mask treatment and local skinning.'))

if __name__=='__main__':main()
