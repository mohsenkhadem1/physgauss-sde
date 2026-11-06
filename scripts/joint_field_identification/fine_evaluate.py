"""Frozen paired coarse/fine inverse comparison on independent finer-mesh truth."""
from pathlib import Path
import argparse,json,time,hashlib,datetime
import numpy as np
from . import bank
from .campaign import run_case,load_bank
ROOT=bank.ROOT;BASE=ROOT/'runs/joint-field-fine-v03';REPORT=ROOT/'reports/joint-field-fine-v03'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',choices=['block','step'],required=True);a=ap.parse_args()
    plan=REPORT/'evaluation-plan.json'
    if not plan.exists():raise RuntimeError('Freeze evaluation plan before execution')
    cfg=json.loads(plan.read_text());ph=digest(plan)
    for p,h in cfg['source_sha256'].items():assert digest(ROOT/p)==h,f'Changed source {p}'
    deadline=datetime.datetime.fromisoformat(cfg['deadline_utc']);bp=BASE/f'{a.shape}.npz';tp=BASE/a.shape/'fine-truth.npz'
    while not (bp.exists() and tp.exists()):
        if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Campaign deadline')
        time.sleep(30)
    truth=dict(np.load(tp));assert len(truth['y'])==16
    fixed=json.loads((ROOT/f'reports/joint-field-identification-v01/{a.shape}/fixed-schedules.json').read_text())
    for model in ['coarse8','fine24']:
        data=load_bank(a.shape) if model=='coarse8' else dict(np.load(bp));m=bank.maps(data)
        out=REPORT/'evaluation'/a.shape/model;out.mkdir(parents=True,exist_ok=True)
        for i in range(16):
            path=out/f'case-{i:03d}.json'
            if path.exists():assert json.loads(path.read_text())['evaluation_plan_sha256']==ph;continue
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Campaign deadline')
            start=time.time();doc=run_case(data,m,truth,i,fixed,4,128,2_000_000+(a.shape=='step')*10000+100*i)
            doc.update(shape=a.shape,case=i,model=model,seconds=time.time()-start,evaluation_plan_sha256=ph,truth_sha256=digest(tp),bank_sha256=digest(ROOT/f'runs/joint-field-identification-v01/{a.shape}.npz' if model=='coarse8' else bp))
            path.write_text(json.dumps(doc,indent=2));print(a.shape,model,i,doc['seconds'],flush=True)
if __name__=='__main__':main()
