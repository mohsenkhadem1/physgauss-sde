"""Population deformation of all48 independent settled reference responses.
Uniform weights on the fixed81 query points; not volume/mass quadrature.
"""
import argparse,json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/physical-refinement-v15';BASE=ROOT/'runs/physical-refinement-v15'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((REPORT/'final-protocol.json').read_text());f=json.loads((REPORT/'method-freeze.json').read_text());scores=json.loads((REPORT/'reserved-scores.json').read_text());commit=json.loads((BASE/'reserved/forecasts/complete.json').read_text())
 if len(commit['rows'])!=2688 or len(scores['rows'])!=7680 or scores['method_freeze_sha256']!=sha(REPORT/'method-freeze.json') or f['protocol_sha256']!=sha(REPORT/'final-protocol.json'):raise ValueError('Complete frozen forecasts and scores required')
 rest=np.stack(np.meshgrid(np.linspace(0,.24,9),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3);centered=rest-rest.mean(0);rows=[];evidence={}
 def bind(path):evidence[str(path.relative_to(ROOT))]=sha(path)
 for world in p['reserved_worlds']:
  for i in range(5,9):
   folder=BASE/'reserved/worlds'/world['name']/'n6'/f'pose-{i:03d}';record=json.loads((folder/'complete.json').read_text());path=folder/'private-state.npz'
   if sha(path)!=record['artifact_sha256']['private-state.npz'] or sha(path)!=scores['evidence_sha256'][str(path.relative_to(ROOT))]:raise ValueError('Reference response changed')
   with np.load(path) as z:field=z['field'].copy()
   if field.shape!=(81,3) or not np.isfinite(field).all():raise ValueError('Invalid reference response')
   points=rest+field;x=points-points.mean(0);u,_,v=np.linalg.svd(x.T@centered);d=np.eye(3);d[-1,-1]=np.linalg.det(u@v);residual=x@(u@d@v)-centered
   total=float(np.sqrt(np.mean(field**2)));deform=float(np.sqrt(np.mean(residual**2)));ratio=deform/total if total else None
   rows.append(dict(world=world['name'],initial_camera=world['initial_camera'],response_index=i-5,pose_m_rad=record['pose_m_rad'],total_displacement_coordinate_rms_mm=1000*total,nonrigid_coordinate_rms_mm=1000*deform,nonrigid_to_total_rms_ratio=ratio,maximum_nonrigid_point_distance_mm=1000*float(np.linalg.norm(residual,axis=1).max()),translation_centroid_distance_mm=1000*float(np.linalg.norm(field.mean(0)))))
   bind(path);bind(folder/'complete.json')
 def summary(rs):
  values=np.array([[r['total_displacement_coordinate_rms_mm'],r['nonrigid_coordinate_rms_mm']] for r in rs]);ratios=np.array([r['nonrigid_to_total_rms_ratio'] for r in rs if r['nonrigid_to_total_rms_ratio'] is not None])
  return dict(count=len(rs),mean_total_rms_mm=float(values[:,0].mean()),mean_nonrigid_rms_mm=float(values[:,1].mean()),nonrigid_rms_range_mm=[float(values[:,1].min()),float(values[:,1].max())],pooled_total_rms_mm=float(np.sqrt(np.mean(values[:,0]**2))),pooled_nonrigid_rms_mm=float(np.sqrt(np.mean(values[:,1]**2))),mean_rms_ratio=float(ratios.mean()),rms_ratio_range=[float(ratios.min()),float(ratios.max())])
 for path in (REPORT/'final-protocol.json',REPORT/'method-freeze.json',REPORT/'reserved-scores.json',BASE/'reserved/forecasts/complete.json',Path(__file__)):bind(path)
 result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,all_declared_responses_included=True,expected=48,rows=rows,all=summary(rows),by_response={str(i):summary([r for r in rows if r['response_index']==i]) for i in range(4)},evidence_sha256=evidence,interpretation='Displacement from undeformed rest at four independently reset settled commands; not a time-trajectory measure. Ratio is RMS ratio, not fraction of variance. No low-deformation cases excluded.')
 (REPORT/'response-deformation-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['all']))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 if not args.execute:parser.error('Explicit post-score authorization required')
 main()
