"""Single prespecified Gaussian diagnostic; development case only."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.benchmark_improvement_v14.phystwin_gaussian_residual import predict,SETTINGS
from scripts.contact_benchmark_v12.data import load,predictor_payload,trim_prefix
from scripts.contact_benchmark_v12.model_selection import _validation_window,_score
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
from scripts.contact_benchmark_v12.feasibility import repair,inherited_elements,jacobians
from scripts.external_benchmark.evaluate import expand_prediction
REPORT=ROOT/'reports/benchmark-improvement-v14'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def topology(data,p,e):
 vals=[jacobians(data['nodes'],x,e) for x in p if np.isfinite(x).all()]
 return dict(failed_frames=int((~np.isfinite(p).all((1,2))).sum()),inverted_frames=int(sum((j<=0).any() for j in vals)),below_cutoff_frames=int(sum((j<=.03+1e-8).any() for j in vals)),minimum_J=float(min(j.min() for j in vals)))
def main():
 paths={tag:ROOT/'runs/contact-benchmark-v12/base'/tag/'single_lift_sloth/prediction.npz' for tag in ('full','validation')}
 hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('phystwin_gaussian_*.py')}
 plan=dict(case='single_lift_sloth',scope='One development hypothesis; fixed settings before outcomes; no other seven outcomes',settings=SETTINGS,source_sha256=hashes,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values()})
 (REPORT/'phystwin-gaussian-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 d=load({'case':'single_lift_sloth'});full=predictor_payload(d);end=d['train_end'];cut=int(.7*end);short=_validation_window(predictor_payload(trim_prefix(d,cut)),end);e=inherited_elements(full)
 base=np.load(paths['full'])['predicted'];shortbase=np.load(paths['validation'])['predicted'][:end]
 rows={};shortpred,sm=predict(short,shortbase);pred,fm=predict(full,base)
 for label,a,b in [('raw',shortpred,pred),('unchanged_global_guard',repair(short,shortbase,shortpred,elements=e)[0],repair(full,base,pred,elements=e)[0])]:
  val,bad=_score(full,a,short,cut,end);rows[label]=dict(validation_rmse_mm=val*1000,validation_failed=bad,suffix=safe_score(d,expand_prediction(d,b),end,len(d['times']))['summary'],topology=topology(full,b,e))
 out=dict(plan=plan,results=rows,short_metadata=sm,full_metadata=fm,original_validation_mm=13.938729085992833,original_manual_mm=34.07648095788899,original_shape_mm=17.33294258936813)
 (REPORT/'phystwin-gaussian-development.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
