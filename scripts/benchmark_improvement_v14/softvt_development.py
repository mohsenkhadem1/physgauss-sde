"""Development-only fixed prototype evaluation; predictions persisted before scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload,load_scene_payload
from scripts.benchmark_improvement_v14.softvt_slip import predict
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/benchmark-improvement-v14/softvt-development'
REPORT=ROOT/'reports/benchmark-improvement-v14'

def main():
    cases=[c for c in json.loads((ROOT/'reports/prefix-ridge-switch-v13/protocol.json').read_text())['cases'] if c['cohort']=='development' and 'case' not in c['spec']]
    assert len(cases)==20
    REPORT.mkdir(parents=True,exist_ok=True)
    plan=dict(cases=cases,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/benchmark_improvement_v14/softvt_slip.py']},scope='Development20 only; fixed8frame gravity slip and half-extent release; no suffix tuning within run')
    (REPORT/'softvt-development-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for c in cases:
        name=c['name'];folder=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name
        data=load_payload(folder/'full.npz');scene=load_scene_payload(folder/'scene.npz')
        oldpath=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'/name
        with np.load(oldpath/'predictions.npz') as a:base=a['model_selection_selected'].copy()
        pred,meta=predict(data,base,dict(scene=scene));out=BASE/name;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'prediction.npz',predicted=pred)
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(name,'forecast saved; trigger',meta['trigger'],flush=True)
    from scripts.contact_benchmark_v12.data import load
    from scripts.external_benchmark.evaluate import expand_prediction
    from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
    rows=[]
    for c in cases:
        data=load(c['spec']);out=BASE/c['name'];pred=np.load(out/'prediction.npz')['predicted'];score=safe_score(data,expand_prediction(data,pred),data['train_end'],len(data['times']))
        old=json.loads((ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'/c['name']/'metrics.json').read_text())
        row=dict(case=c,proposal=score,baseline={k:old['results'][k] for k in ['model_selection_selected','prefix_calibrated_rigid']},metadata=json.loads((out/'metadata.json').read_text()))
        rows.append(row);print(c['name'],'world',score['summary']['coordinate_rmse_mm'],flush=True)
    (REPORT/'softvt-development-results.json').write_text(json.dumps(dict(scope=plan['scope'],cases=rows),indent=2)+'\n')
if __name__=='__main__':main()
