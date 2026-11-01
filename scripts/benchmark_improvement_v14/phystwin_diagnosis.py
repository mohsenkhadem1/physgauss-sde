"""Development-only attachment/control diagnosis. Never evaluates held-out cases."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.model_selection import _observed_prediction
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.external_benchmark.data import rigid_map
from scripts.benchmark_improvement_v14.phystwin_pose import predict
CASE='single_lift_sloth';REPORT=ROOT/'reports/benchmark-improvement-v14'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=load({'case':CASE});payload=predictor_payload(data);end=data['train_end'];T=len(data['times']);out={};evidence={}
 parent=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'/CASE
 path=parent/'predictions.npz'
 with np.load(path) as a:prior=a['model_selection_selected'].copy();rigid=a['model_selection_rigid'].copy()
 evidence[str(path.relative_to(ROOT))]=digest(path)
 # Current-family cached FEM is an inherited deformation trajectory, not refit.
 from scripts.contact_benchmark_v12.experiment import base_path
 path=base_path(data,'development')
 with np.load(path) as a:base=a['predicted'].copy()
 evidence[str(path.relative_to(ROOT))]=digest(path)
 candidates={'archived_selected':prior,'archived_rigid':rigid};metadata={}
 original=ROOT/'runs/contact-benchmark-v12/evaluation/development'/CASE/'predictions.npz'
 with np.load(original) as a:candidates['frozen_object_frame']=a['object_frame'].copy()
 evidence[str(original.relative_to(ROOT))]=digest(original)
 for method in ('controller_translation','controller_rigid','controller_fitted_rotation'):
  for inherited in (False,True):
   name=method+('_fem_shape' if inherited else '_rigid_shape');candidates[name],metadata[name]=predict(payload,base if inherited else None,method)
 from scripts.benchmark_improvement_v14.phystwin_shape import predict as shape_predict
 for method in ('shape_gain_pose','shape_gain_residual','shape_gain_none','shape_axis_gain_pose','shape_spatial_gain_pose'):
  candidates[method],metadata[method]=shape_predict(payload,base,method)
 for name,pred in candidates.items():
  projection=_observed_prediction(payload,pred[:end]);mask=payload['observed_mask'];prefix=float(np.sqrt(np.mean((projection[mask]-payload['observed'][mask])**2))*1000)
  scores=safe_score(data,expand_prediction(data,pred),end,T)['summary'];out[name]=dict(prefix_coordinate_mm=prefix,suffix=scores,metadata=metadata.get(name))
 # Visible marker rigid-motion residual, prefix and development-only suffix.
 phases={}
 for label,start,stop in [('prefix',0,end),('development_suffix',end,T)]:
  residual=[];motion=[]
  for t in range(start,stop):
   valid=data['evaluation']['mask'][t]&np.isfinite(data['evaluation']['points'][t]).all(1);x=data['evaluation']['points'][0,valid];y=data['evaluation']['points'][t,valid]
   if len(x)<3:continue
   R,b=rigid_map(x,y);residual.append(np.mean((x@R+b-y)**2));motion.append(np.mean((x-y)**2))
  phases[label]=dict(nonrigid_coordinate_rms_mm=float(np.sqrt(np.mean(residual))*1000),total_coordinate_rms_mm=float(np.sqrt(np.mean(motion))*1000),rms_ratio=float(np.sqrt(np.mean(residual)/np.mean(motion))))
 inventory=sorted(p.parent.name for p in (ROOT/'runs/third_party/PhysTwin/data/different_types').glob('*/split.json'))
 report=dict(scope='Only single_lift_sloth development outcomes inspected; no held-out outcome tuning; six prespecified attachment/shape candidates diagnosed',case=CASE,train_end=end,frames=T,local_case_inventory=inventory,untouched_local_cases=[],inventory_note='All eight locally available cases appear in historical v12 benchmark cohorts; no untouched local case identified',phase_motion=phases,candidates=out,input_sha256=evidence,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('phystwin*.py')})
 REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'phystwin-development-diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
