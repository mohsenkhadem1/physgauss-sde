"""Independent SHORT cached-fit validation of development-only shape hypotheses."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.contact_benchmark_v12.data import load,predictor_payload,trim_prefix
from scripts.contact_benchmark_v12.experiment import base_path
from scripts.contact_benchmark_v12.model_selection import _score,_validation_window
from scripts.benchmark_improvement_v14.phystwin_shape import predict
CASE='single_lift_sloth';REPORT=ROOT/'reports/benchmark-improvement-v14'
def main():
 data=load({'case':CASE});end=data['train_end'];cut=max(3,int(end*.7));full=predictor_payload(data);short=predictor_payload(trim_prefix(data,cut));path=base_path(data,'development',True)
 with np.load(path) as a:base=a['predicted'].copy()
 # Slice public inputs and inherited SHORT prediction at validation end; no
 # postvalidation prediction state/failure enters validation scores.
 short=_validation_window(short,end);base=base[:end];rows=[]
 for method in ('shape_gain_pose','shape_gain_residual','shape_gain_none','shape_axis_gain_pose','shape_spatial_gain_pose'):
  pred,meta=predict(short,base,method);error,failures=_score(full,pred,short,cut,end)
  rows.append(dict(method=method,short_gain=meta['deformation_gain'],validation_rmse_mm=1000*error,failures=failures,guard_modified_frames=meta['guard']['modified_frames']))
 oldpath=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'/CASE/'prediction-metadata.json';old=json.loads(oldpath.read_text())['selected_configuration'];oldscore=old['validation_rmse_m']*1000
 out=dict(scope='Development SHORT-fit prefix validation only; fixed cutoff41/59; no held-out outcomes; known public input window ends at59',cut=cut,end=end,archived_selected_validation_rmse_mm=oldscore,candidates=rows,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,oldpath]},source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('phystwin*.py')})
 (REPORT/'phystwin-development-short-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
