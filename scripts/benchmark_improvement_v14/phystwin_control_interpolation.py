"""One development diagnostic: prefix-validation fitted convex correction weight."""
import json
from pathlib import Path
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.object_frame import _observed_prediction
ROOT=Path(__file__).resolve().parents[2]
def main():
    report=ROOT/'reports/benchmark-improvement-v14';folder=ROOT/'runs/prefix-ridge-switch-v13/inputs/single_lift_sloth'
    full=load_payload(folder/'full.npz');short=load_payload(folder/'short.npz');cut=short['train_end'];end=full['train_end']
    artifact=ROOT/'runs/benchmark-improvement-v14/phystwin-control-development/predictions.npz'
    with np.load(artifact)as f:arrays={k:f[k].copy()for k in f.files}
    a=_observed_prediction(full,arrays['short_original'][cut:end]);b=_observed_prediction(full,arrays['short_predicted'][cut:end]);target=full['observed'][cut:end]
    valid=full['observed_mask'][cut:end]&np.isfinite(target).all(2)
    delta=(b-a)[valid];residual=(target-a)[valid];weight=float(np.clip(np.sum(delta*residual)/max(np.sum(delta*delta),1e-20),0.,1.))
    old=float(np.sqrt(np.mean(residual**2)));new=float(np.sqrt(np.mean((residual-weight*delta)**2)))
    accept=bool(weight>0 and new<.98*old)
    record=dict(case='single_lift_sloth',scope='Development-only empirical residual interpolation; weight fit on permitted SHORT-to-FULL validation only, same interval used to assess threshold so not an untouched validation set',weight=weight,validation_original_mm=old*1000,validation_interpolated_mm=new*1000,strict_two_percent_pass=accept,decision='Proceed single development suffix score'if accept else'Stop branch; retain original; no suffix score')
    (report/'phystwin-control-interpolation-plan.json').write_text(json.dumps(record,indent=2)+'\n')
    if accept:
        prediction=arrays['original']+weight*(arrays['predicted']-arrays['original']);prediction[0]=arrays['original'][0]
        path=artifact.parent/'interpolated.npz';np.savez_compressed(path,predicted=prediction)
        from scripts.contact_benchmark_v12.data import load
        from scripts.external_benchmark.evaluate import expand_prediction
        from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
        data=load(dict(case='single_lift_sloth'));record['suffix']=safe_score(data,expand_prediction(data,prediction),end,len(data['times']))['summary']
    (report/'phystwin-control-interpolation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
