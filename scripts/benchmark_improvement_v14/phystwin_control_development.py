"""Exactly one development case; predictions and SHORT validation before suffix scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.benchmark_improvement_v14.phystwin_control_strain import predict
from scripts.contact_benchmark_v12.object_frame import adapt,_observed_prediction
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/benchmark-improvement-v14';B=ROOT/'runs/benchmark-improvement-v14/phystwin-control-development'
def main():
 name='single_lift_sloth';folder=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name;full=load_payload(folder/'full.npz');short=load_payload(folder/'short.npz')
 original=ROOT/'runs/contact-benchmark-v12/evaluation/development'/name;meta=json.loads((original/'prediction-metadata.json').read_text())['object_frame']
 with np.load(original/'predictions.npz')as a:base=a['object_frame'].copy()
 with np.load(ROOT/'runs/contact-benchmark-v12/base/validation'/name/'prediction.npz')as a:shortbase=a['predicted'].copy()
 shortanchor,_=adapt(short,shortbase,method=meta['method'],strength=meta['strength'])
 plan=dict(case=name,scope='Development only, ONE controller-driven affine logstrain candidate, SHORT-fit correction validation; original adapter configuration previously selected on same validation interval',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/benchmark_improvement_v14/phystwin_control_strain.py']})
 (R/'phystwin-control-plan.json').write_text(json.dumps(plan,indent=2)+'\n');B.mkdir(parents=True,exist_ok=True)
 validation,vm=predict(short,shortanchor);prediction,pm=predict(full,base)
 np.savez_compressed(B/'predictions.npz',original=base,predicted=prediction,short_original=shortanchor,short_predicted=validation)
 cut=short['train_end'];end=full['train_end'];valid=full['observed_mask'][cut:end]&np.isfinite(full['observed'][cut:end]).all(2)
 vals={k:float(np.sqrt(np.mean((_observed_prediction(full,a[cut:end])[valid]-full['observed'][cut:end][valid])**2))*1000)for k,a in [('original',shortanchor),('candidate',validation)]}
 from scripts.contact_benchmark_v12.data import load
 from scripts.external_benchmark.evaluate import expand_prediction
 from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
 data=load(dict(case=name));scores={k:safe_score(data,expand_prediction(data,a),end,len(data['times']))['summary']for k,a in [('original',base),('candidate',prediction)]}
 out=dict(plan=plan,validation_coordinate_mm=vals,suffix=scores,full_metadata=pm,short_metadata=vm);(R/'phystwin-control-development.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(validation=vals,suffix=scores),indent=2))
if __name__=='__main__':main()
