"""Durable postprocessing after all bowl workers finish; never changes protocol."""
import json,os,time,traceback,hashlib
import numpy as np
from pathlib import Path
from .bowl_campaign import REPORT,ROOT,INFERENCE_N,REFERENCE_N,DEADLINE_UNIX,THETA,generate,seal_banks,predict,score_saved,verify_banks,verify_sources,protocol_hash
from .bowl_report import main as report
from .bowl_audit import audit_file
from .bowl_publication import main as publication

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    status=REPORT/'finalization.json'
    try:
        roles=('inference-and-target','reference-push','reference-rotation')
        while True:
            workers=[json.loads((REPORT/f'worker-{role}.json').read_text()) for role in roles]
            running=[w for w in workers if w['status']=='running' and alive(w['pid'])]
            if not running:break
            status.write_text(json.dumps(dict(status='waiting-for-workers',pid=os.getpid(),running=len(running),time_unix=time.time()),indent=2))
            if time.time()>=DEADLINE_UNIX:raise RuntimeError('Campaign deadline reached')
            time.sleep(30)
        status.write_text(json.dumps(dict(status='aggregating-banks',pid=os.getpid()),indent=2))
        verify_sources()
        missing=[];invalid=[]
        for n in (INFERENCE_N,REFERENCE_N):
            for h in range(len(THETA)):
                for interaction in ('calibration_push','calibration_tilt','heldout_push'):
                    path=ROOT/f'fem-n{n}'/f'h{h:02d}-{interaction}.npz'
                    if not path.exists():missing.append(str(path));continue
                    try:
                        with np.load(path) as r:
                            assert str(r['protocol_sha256'])==protocol_hash()
                            assert r['frames'].shape==(66,(n+1)**3,3)
                            assert r['reaction_averages'].shape==(66,2,3)
                            assert np.isclose(r['times'][-1],1.3,rtol=0,atol=1e-12)
                            assert np.isfinite(r['frames']).all() and float(r['min_j'])>.02
                    except Exception as e:invalid.append(dict(path=str(path),error=repr(e)))
        if missing or invalid:
            status.write_text(json.dumps(dict(status='requires-review',missing=missing,invalid=invalid,
                retained_failure_records=[str(p) for p in ROOT.rglob('*failure.json')],
                action='STOP: no automatic CPU retries, no exclusions, no scoring of incomplete banks',time_unix=time.time()),indent=2))
            return
        for n in (INFERENCE_N,REFERENCE_N):
            bank=ROOT/f'fem-n{n}-bank.npz'
            if not bank.exists():
                if generate('fem',n) is None:raise RuntimeError('Required bank failed; all attempted failures retained, no accuracy claim')
        if not (REPORT/'bank-seal.json').exists():seal_banks()
        else:verify_banks()
        status.write_text(json.dumps(dict(status='prospective-predictions',pid=os.getpid()),indent=2))
        if not (REPORT/'prospective-predictions.json').exists():predict()
        if not (REPORT/'case-results.json').exists():score_saved()
        report()
        status.write_text(json.dumps(dict(status='all-state-geometry-audit',pid=os.getpid()),indent=2))
        files=sorted(list((ROOT/f'fem-n{INFERENCE_N}').glob('h*.npz'))+list((ROOT/f'fem-n{REFERENCE_N}').glob('h*.npz')))
        rows=[]
        for path in files:
            rows.append(audit_file(path))
            (REPORT/'all-state-geometry-audit.json').write_text(json.dumps(dict(files=rows,total_visibility_checks=sum(r['visibility_checks'] for r in rows),total_visibility_failures=sum(len(r['visibility_failures']) for r in rows),completed_files=len(rows),expected_files=len(files)),indent=2))
        status.write_text(json.dumps(dict(status='rendering-figures-and-videos',pid=os.getpid()),indent=2))
        publication()
        status.write_text(json.dumps(dict(status='results-audits-and-figures-complete',pid=os.getpid(),time_unix=time.time(),summary_sha256=hashlib.sha256((REPORT/'summary.json').read_bytes()).hexdigest()),indent=2))
    except Exception:
        status.write_text(json.dumps(dict(status='requires-review',pid=os.getpid(),traceback=traceback.format_exc(),time_unix=time.time()),indent=2));raise

if __name__=='__main__':main()
