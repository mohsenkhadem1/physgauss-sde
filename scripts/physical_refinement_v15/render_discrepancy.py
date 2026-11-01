"""Read-only development n4/n6 rendered-feature discrepancy at fixed nominal noise."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.physical_refinement_v15.banks import checked_npz,BASE,CAMERAS
from scripts.physical_refinement_v15.study import development_actions
from scripts.physical_refinement_v15.sensor import SensorSettings
from scripts.physical_refinement_v15.feature_likelihood import covariance
ROOT=Path(__file__).resolve().parents[2]
def main():
 settings=SensorSettings();rows=[];evidence={}
 for action in development_actions():
  for camera in CAMERAS:
   data,hashes=checked_npz(BASE/'render-bank/parallel_lower-g2-r0'/action,camera+'.npz');evidence.update(hashes)
   with data:C=covariance(data['rgb'],data['depth'],settings.depth_sd_m,settings.rgb_sd)
   for i in range(4):
    values=[];images=[]
    for level in [4,6]:
     data,hashes=checked_npz(BASE/'world-render-bank'/f'development-{i}'/f'n{level}'/action,camera+'.npz');evidence.update(hashes)
     with data:values.append(data['features'].copy());images.append(dict(rgb=data['rgb'].copy(),depth=data['depth'].copy()))
    delta=values[0]-values[1];d2=float(delta@np.linalg.solve(C,delta));depth=np.arange(3,64,4);rgb=np.array([j for j in range(64)if j%4!=3]);Cd=C[np.ix_(depth,depth)];Cr=C[np.ix_(rgb,rgb)]
    joint=np.isfinite(images[0]['depth'])&np.isfinite(images[1]['depth']);changed=np.any(images[0]['rgb']!=images[1]['rgb'],axis=2)
    rows.append(dict(world=f'development-{i}',action=action,camera=camera,mahalanobis_squared=d2,gaussian_KL_nats=d2/2,depth_D2=float(delta[depth]@np.linalg.solve(Cd,delta[depth])),rgb_D2=float(delta[rgb]@np.linalg.solve(Cr,delta[rgb])),rgb_changed_pixels=int(changed.sum()),silhouette_changed_pixels=int(np.count_nonzero(np.isfinite(images[0]['depth'])!=np.isfinite(images[1]['depth']))),joint_depth_rmse_m=float(np.sqrt(np.mean((images[0]['depth'][joint]-images[1]['depth'][joint])**2)))))
 result=dict(scope='Same developmentworld parameters/actions, n4vsn6geometry rasterized by samefixedrenderer. Uses EXACTpublicnominalcovariance, no newnoise orparameters. LargeD2 indicates combinedmechanical/rendering approximation relative to sensor; doesnot alone prove aliasingcause.',count=len(rows),max_D2=max(r['mahalanobis_squared']for r in rows),median_D2=float(np.median([r['mahalanobis_squared']for r in rows])),over_one=sum(r['mahalanobis_squared']>1 for r in rows),over_ten=sum(r['mahalanobis_squared']>10 for r in rows),rows=rows,evidence_sha256=evidence)
 result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=ROOT/'reports/physical-refinement-v15/render-discrepancy.json';p.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k not in ['rows','evidence_sha256']});print('top',sorted(rows,key=lambda r:r['mahalanobis_squared'],reverse=True)[:4])
if __name__=='__main__':main()
