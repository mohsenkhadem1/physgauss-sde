"""Single prescribed centroid law; SHORT calibration and2% validation gate."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.object_frame import _observed_prediction
from scripts.benchmark_improvement_v14.phystwin_control_centroid import predict
ROOT=Path(__file__).resolve().parents[2]
def main():
 report=ROOT/'reports/benchmark-improvement-v14';folder=ROOT/'runs/prefix-ridge-switch-v13/inputs/single_lift_sloth'
 full=load_payload(folder/'full.npz');short=load_payload(folder/'short.npz');cut=short['train_end'];end=full['train_end']
 path=ROOT/'runs/benchmark-improvement-v14/phystwin-control-development/predictions.npz'
 with np.load(path)as f:base=f['original'].copy();anchor=f['short_original'].copy()
 plan=dict(case='single_lift_sloth',law='Terminal shared-observation centroid translation transported in existing controller frame, no tunable hyperparameters',fit_interval=[0,cut],validation_interval=[cut,end],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/benchmark_improvement_v14/phystwin_control_centroid.py']},scope='Development only; original adapter configuration previously selected on same validation interval')
 (report/'phystwin-centroid-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 candidate,meta=predict(short,anchor);valid=full['observed_mask'][cut:end]&np.isfinite(full['observed'][cut:end]).all(2)
 scores={key:float(np.sqrt(np.mean((_observed_prediction(full,arr[cut:end])[valid]-full['observed'][cut:end][valid])**2))*1000)for key,arr in [('original',anchor),('candidate',candidate)]}
 accept=bool(scores['candidate']<.98*scores['original']);record=dict(plan=plan,short_fit=meta,validation_coordinate_mm=scores,strict_two_percent_pass=accept,decision='Evaluate development suffix once'if accept else'Stop branch; original retained; no suffix evaluation')
 if accept:
  prediction,fullmeta=predict(full,base);out=ROOT/'runs/benchmark-improvement-v14/phystwin-centroid-development';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'prediction.npz',predicted=prediction)
  from scripts.contact_benchmark_v12.data import load
  from scripts.external_benchmark.evaluate import expand_prediction
  from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
  data=load(dict(case='single_lift_sloth'));record['full_fit']=fullmeta;record['suffix']=safe_score(data,expand_prediction(data,prediction),end,len(data['times']))['summary']
 (report/'phystwin-centroid-development.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
