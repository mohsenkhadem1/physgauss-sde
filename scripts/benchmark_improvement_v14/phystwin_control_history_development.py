"""Predeclared nested-prefix history diagnostic, development case only."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.object_frame import _observed_prediction
from scripts.external_benchmark.data import interpolate_matrix
from scripts.benchmark_improvement_v14.phystwin_control_history import LAGS,choose_lag,marker_fit,support
ROOT=Path(__file__).resolve().parents[2]
def main():
 r=ROOT/'reports/benchmark-improvement-v14';f=ROOT/'runs/prefix-ridge-switch-v13/inputs/single_lift_sloth';full=load_payload(f/'full.npz');short=load_payload(f/'short.npz');cut=short['train_end'];end=full['train_end']
 plan=dict(case='single_lift_sloth',grid=list(LAGS),rule='Choose lag by first70%SHORT→last30%SHORT; refit entireSHORT; require outer41:59 RMSE<.98original; fixedchosenlag refitFULL if pass',ridge=.1,initial_reference='Preserve original expanded frame0',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/benchmark_improvement_v14/phystwin_control_history.py']})
 (r/'phystwin-history-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 with np.load(ROOT/'runs/benchmark-improvement-v14/phystwin-control-development/predictions.npz')as a:base=a['original'].copy();shortbase=a['short_original'].copy()
 tau,meta=choose_lag(short);markers=marker_fit(short,cut,tau)+short['observed_initial'][None];old=_observed_prediction(full,shortbase[cut:end]);valid=full['observed_mask'][cut:end]&np.isfinite(full['observed'][cut:end]).all(2)
 olderr=float(np.sqrt(np.mean((old[valid]-full['observed'][cut:end][valid])**2))*1000);newerr=float(np.sqrt(np.mean((markers[cut:end][valid]-full['observed'][cut:end][valid])**2))*1000);passed=newerr<.98*olderr
 record=dict(plan=plan,lag_seconds=tau,inner_selection=meta,outer_validation_original_mm=olderr,outer_validation_history_mm=newerr,passes_two_percent=passed)
 if passed:
  init=full['support_geometry'];A=interpolate_matrix(full['nodes'],init);reference=init+A@(base[0]-full['nodes']);pred=support(full,marker_fit(full,end,tau),reference);out=ROOT/'runs/benchmark-improvement-v14/phystwin-history-development';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'predictions.npz',predicted=pred)
  from scripts.contact_benchmark_v12.data import load
  from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
  data=load(dict(case='single_lift_sloth'));record['suffix']=safe_score(data,pred,end,len(data['times']))['summary']
 else:record['decision']='Stop without suffix scoring'
 (r/'phystwin-history-development.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
