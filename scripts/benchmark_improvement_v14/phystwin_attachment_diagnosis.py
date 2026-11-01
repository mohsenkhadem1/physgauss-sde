"""Development-only independent SHORT/full physics fits. No other case targets."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.contact_benchmark_v12.data import load,predictor_payload,trim_prefix
from scripts.contact_benchmark_v12.model_selection import _validation_window,_score
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
from scripts.contact_benchmark_v12.object_frame import adapt
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.benchmark_improvement_v14.phystwin_attachment import fit,predict
OUT=ROOT/'runs/benchmark-improvement-v14/phystwin-attachment-development';REPORT=ROOT/'reports/benchmark-improvement-v14'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('phystwin_attachment*.py')}
 write(OUT/'plan.json',dict(case='single_lift_sloth',scope='Development only',settings=dict(materials=6,frames=36,iterations=100,width_bounds_mm=[7.5,30],initial_width_mm=15,fit_budget_seconds=1800,predict_budget_seconds=1200,device='cuda:1'),source_sha256=sources))
 data=load({'case':'single_lift_sloth'});full=predictor_payload(data);end=full['train_end'];cut=int(.7*end);short=_validation_window(predictor_payload(trim_prefix(data,cut)),end)
 output={};predictions={}
 for tag,payload in [('short',short),('full',full)]:
  theta,fitting=fit(payload);write(OUT/(tag+'-fit.json'),fitting);prediction,meta=predict(payload,theta);np.savez_compressed(OUT/(tag+'-prediction.npz'),predicted=prediction,theta=theta);write(OUT/(tag+'-prediction.json'),meta);predictions[tag]=prediction;output[tag]=dict(fit=fitting,prediction=meta)
 # Exact previous seven-adapter menu; selector uses SHORT only.
 rows=[]
 for method,strength in [('none',0.)]+[(m,s) for m in ('pose','residual') for s in (.25,.5,1.)]:
  p,_=adapt(short,predictions['short'],method=method,strength=strength);score,bad=_score(full,p,short,cut,end)
  rows.append(dict(method=method,strength=strength,validation_mm=float(score)*1000 if np.isfinite(score) else None,failed=bad))
 best=min(rows,key=lambda r:float('inf') if r['validation_mm'] is None else r['validation_mm'])
 if rows[0]['validation_mm'] is not None and not (best['validation_mm'] is not None and best['validation_mm']<.98*rows[0]['validation_mm']):best=rows[0]
 chosen,meta=adapt(full,predictions['full'],method=best['method'],strength=best['strength'])
 np.savez_compressed(OUT/'selected-prediction.npz',predicted=chosen)
 output.update(validation=rows,selected=best,selection_before_suffix_scoring=True,scope='Single development case, no held-out confirmation',accepted_original_manual_mm=34.07648095788899,source_sha256=sources,input_sha256={data['source']:sha(Path(data['source']))})
 # Suffix evaluation kept outside model and strictly after the fixed-prefix selection.
 output['selected_suffix']=safe_score(data,expand_prediction(data,chosen),end,len(data['times']))['summary']
 output['fem_suffix']=safe_score(data,expand_prediction(data,predictions['full']),end,len(data['times']))['summary']
 write(REPORT/'phystwin-attachment-development.json',output);print(json.dumps(dict(selected=best,selected_suffix=output['selected_suffix'],fem_suffix=output['fem_suffix'])),flush=True)
if __name__=='__main__':main()
