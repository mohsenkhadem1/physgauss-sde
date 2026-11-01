"""One development case only: localized repair and genuine SHORT validation."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.contact_benchmark_v12.data import load,predictor_payload,trim_prefix
from scripts.contact_benchmark_v12.experiment import base_path
from scripts.contact_benchmark_v12.model_selection import _validation_window,_score
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
from scripts.contact_benchmark_v12.object_frame import adapt
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.benchmark_improvement_v14.phystwin_local_guard import predict
CASE='single_lift_sloth';REPORT=ROOT/'reports/benchmark-improvement-v14'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=load({'case':CASE});full=predictor_payload(data);end=data['train_end'];cut=max(3,int(end*.7));short=_validation_window(predictor_payload(trim_prefix(data,cut)),end);paths=[base_path(data,'development'),base_path(data,'development',True)]
 with np.load(paths[0]) as f:base=f['predicted'].copy()
 with np.load(paths[1]) as f:shortbase=f['predicted'][:end].copy()
 rows=[];oldraw,_=adapt(short,shortbase,method='residual',strength=.5);oldscore=_score(full,oldraw,short,cut,end)[0]*1000
 for method in ('local_residual','local_response_residual'):
  for strength in (.25,.5,1.):
   shortpred,sm=predict(short,shortbase,method,strength);score,bad=_score(full,shortpred,short,cut,end)
   pred,meta=predict(full,base,method,strength);suffix=safe_score(data,expand_prediction(data,pred),end,len(data['times']))['summary']
   row=dict(method=method,strength=strength,short_response_gain=sm['response_gain'],full_response_gain=meta['response_gain'],validation_rmse_mm=1000*score,validation_failed_frames=bad,suffix=suffix,modified_frames=sum(r.get('modified_nodes',0)>0 for r in meta['guard']['rows']),global_fallback_frames=sum(r.get('global_fallback',False) for r in meta['guard']['rows']),failed_frames=meta['guard']['failed_frames']);rows.append(row);print(json.dumps(row),flush=True)
 output=dict(scope='Only single_lift_sloth development; candidate shortlist fixed before this diagnostic; no held-out tuning',cut=cut,end=end,accepted_raw_primary_validation_rmse_mm=oldscore,accepted_raw_primary_manual_mm=34.07648095788899,candidates=rows,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('phystwin*.py')})
 (REPORT/'phystwin-local-development.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
