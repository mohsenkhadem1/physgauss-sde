"""Development-only fixed-footprint renderer qualification; no production changes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scripts.physical_refinement_v15.render_bank import ROOT,BASE,REPORT,CAMERAS,sha,write,render_geometry
from scripts.physical_refinement_v15.observations import Camera
from scripts.physical_refinement_v15.feature_likelihood import features,covariance,FAR_CLIP
from scripts.physical_refinement_v15.study import development_actions
from scripts.physical_refinement_v15.sensor import SensorSettings
from scripts.physical_refinement_v15.banks import checked_npz
PAIRS=(('initial','left'),('design_opposed_translation','left'),('probe_right_bend','right'),('design_right_bend','right'),('design_opposed_translation','oblique'))
OUT=BASE/'supersampling-diagnostic'
def pixel_average(image,factor):
 h,w=image['depth'].shape;h//=factor;w//=factor
 rgb=image['rgb'].reshape(h,factor,w,factor,3).mean((1,3))
 # Average the existing feature's clipped depth signal, including background.
 depth=np.where(np.isfinite(image['depth']),image['depth'],FAR_CLIP).reshape(h,factor,w,factor).mean((1,3))
 return dict(rgb=rgb,depth=depth)
def job(spec):
 world,action,camname,level,factor=spec;index=list(development_actions()).index(action)
 folder=BASE/'worlds'/world/f'n{level}'/f'pose-{index:03d}';path=folder/'sensor-geometry.npz';complete=folder/'complete.json'
 record=json.loads(complete.read_text());assert record['artifact_sha256'][path.name]==sha(path)
 evidence={str(p.relative_to(ROOT)):sha(p) for p in (path,complete)}
 cam=CAMERAS[camname];high=Camera(cam.eye,cam.target,cam.width*factor,cam.height*factor,cam.focal*factor)
 started=time.monotonic();image,_=render_geometry(path,high);image=pixel_average(image,factor);elapsed=time.monotonic()-started
 target=OUT/world/action/camname/f'n{level}-s{factor}.npz';target.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(target,**image,features=features(**image))
 assert all(sha(ROOT/p)==digest for p,digest in evidence.items())
 return dict(spec=spec,seconds=elapsed,path=str(target.relative_to(ROOT)),sha256=sha(target),evidence_sha256=evidence)
def main():
 source_paths=[Path(__file__),*[ROOT/'scripts/physical_refinement_v15'/p for p in ('render_bank.py','observations.py','feature_likelihood.py','study.py','sensor.py','banks.py')]]
 sources={str(p.relative_to(ROOT)):sha(p)for p in source_paths}
 specs=[(f'development-{i}',a,c,n,s)for i in range(4)for a,c in PAIRS for n in (4,6)for s in (2,4)]
 plan=dict(scope='Development only; five fixed action/camera pairs across all four worlds, including previously diagnosed worst views and initial/oblique controls.',pairs=PAIRS,worlds=list(range(4)),factors=[1,2,4],jobs=specs,source_sha256=sources,noise='Unchanged factor1 public nominal covariance for all comparisons.',sampling='Uniform s×s pixel-center rays; mean RGB and far-clipped depth within each original pixel. This is feature-space footprint integration, not a calibrated mixed-return depth camera.',criteria='Report every D2 and fixed-geometry factor convergence; no outcome-dependent noise or factor selection.',workers=4)
 planpath=REPORT/'supersampling-diagnostic-plan.json'
 if planpath.exists():raise RuntimeError('Preserve existing diagnostic plan/results')
 write(planpath,plan)
 started=time.monotonic()
 with ProcessPoolExecutor(max_workers=4)as pool:outputs=list(pool.map(job,specs))
 assert all(sha(ROOT/p)==digest for p,digest in sources.items())
 evidence={str(planpath.relative_to(ROOT)):sha(planpath)};lookup={tuple(r['spec']):r for r in outputs};rows=[];settings=SensorSettings()
 for i in range(4):
  world=f'development-{i}'
  for action,cam in PAIRS:
   nominal,hs=checked_npz(BASE/'render-bank/parallel_lower-g2-r0'/action,cam+'.npz');evidence.update(hs)
   with nominal:C=covariance(nominal['rgb'],nominal['depth'],settings.depth_sd_m,settings.rgb_sd)
   vals={};runtime={}
   for n in (4,6):
    old,hs=checked_npz(BASE/'world-render-bank'/world/f'n{n}'/action,cam+'.npz');evidence.update(hs)
    with old:vals[n,1]=old['features'].copy()
    for s in (2,4):
     r=lookup[world,action,cam,n,s];evidence.update(r['evidence_sha256']);evidence[r['path']]=r['sha256'];runtime[f'n{n}_s{s}']=r['seconds']
     with np.load(ROOT/r['path'])as p:vals[n,s]=p['features'].copy()
   def distance(a,b):
    d=a-b;return float(d@np.linalg.solve(C,d))
   rows.append(dict(world=world,action=action,camera=cam,mesh_D2={str(s):distance(vals[4,s],vals[6,s])for s in (1,2,4)},fixed_geometry_renderer_D2={f'n{n}_{a}_to_{b}':distance(vals[n,a],vals[n,b])for n in (4,6)for a,b in ((1,2),(2,4))},render_seconds=runtime))
 result=dict(plan_sha256=sha(planpath),source_sha256=sources,evidence_sha256=evidence,rows=rows,total_seconds=time.monotonic()-started,total_renders=len(outputs),note='Low n4/n6 discrepancy alone is insufficient: fixed-geometry renderer convergence is reported separately. No production remedy adopted.')
 write(REPORT/'supersampling-diagnostic.json',result)
 print(json.dumps({s:{'max':max(r['mesh_D2'][s]for r in rows),'median':float(np.median([r['mesh_D2'][s]for r in rows]))}for s in ('1','2','4')}));print('seconds',result['total_seconds'])
if __name__=='__main__':main()
