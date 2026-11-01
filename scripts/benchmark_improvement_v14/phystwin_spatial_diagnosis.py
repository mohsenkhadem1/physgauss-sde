"""Development-only spatial localization of accepted original prediction errors."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from scripts.contact_benchmark_v12.data import load
from scripts.external_benchmark.data import rigid_map
from scripts.contact_benchmark_v12.object_frame import _observed_prediction

def main():
 d=load({'case':'single_lift_sloth'});p=ROOT/'runs/contact-benchmark-v12/evaluation/development/single_lift_sloth/predictions.npz'
 with np.load(p) as f:x=f['object_frame']
 # Observation interpolation at all times is a fixed initial-coordinate map.
 pred=_observed_prediction(d,x);truth=d['evaluation']['points'];mask=d['evaluation']['mask']
 dist=cKDTree(d['controls'][0]).query(d['observed_initial'])[0];edges=np.quantile(dist,[0,.25,.5,.75,1]);bins=np.clip(np.searchsorted(edges[1:-1],dist,side='right'),0,3)
 rows=[];sums=np.zeros((4,2))
 for t in range(d['train_end'],len(pred)):
  valid=mask[t]&np.isfinite(pred[t]).all(1)&np.isfinite(truth[t]).all(1)
  a=pred[t,valid];b=truth[t,valid];R,c=rigid_map(a,b);aligned=a@R+c
  rows.append(dict(frame=t,coordinate_rmse_mm=float(np.sqrt(np.mean((a-b)**2))*1000),proper_aligned_coordinate_rmse_mm=float(np.sqrt(np.mean((aligned-b)**2))*1000),marker_centroid_error_mm=float(np.linalg.norm(a.mean(0)-b.mean(0))*1000),rotation_error_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))))
  for k in range(4):
   v=valid&(bins==k);sums[k]+=[float(np.sum((pred[t,v]-truth[t,v])**2)),int(v.sum())*3]
 out=dict(scope='single_lift_sloth development only; visible tracked-marker diagnostics, not semantic anatomy or mass COM',frames=rows,quartiles=[dict(controller_initial_distance_range_mm=(edges[k:k+2]*1000).tolist(),coordinate_rmse_mm=float(np.sqrt(sums[k,0]/sums[k,1])*1000),valid_coordinates=int(sums[k,1])) for k in range(4)],summary={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='frame'},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (ROOT/'reports/benchmark-improvement-v14/phystwin-spatial-development.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary']));print(json.dumps(out['quartiles']))
if __name__=='__main__':main()
