"""Independent recomputation (gate G8) for the PAC-NeRF/GIC Torus comparison.
Reads archived predictions only; imports no training code. Adds the native ground-truth
particle metric (released PAC-NeRF simulation_data, frame i = video frame i at 24 fps).
"""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from skimage.metrics import structural_similarity
R=Path(__file__).resolve().parents[2];B=R/'runs/gap-crossmethod-v02';REP=R/'reports/gap-crossmethod-v02'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ply(p):
 v=PlyData.read(p)['vertex'];return np.stack([v[k] for k in 'xyz'],-1).astype(np.float64)
def chamfer(a,ta,b,tb):
 da=tb.query(a)[0];db=ta.query(b)[0];return float((da.mean()+db.mean())/2)
def main():
 out=dict(scope='Independent recomputation from archived predictions; no training code imported.',checks=[])
 # 1. PAC-NeRF held camera image metrics
 d=np.load(B/'pacnerf-predictions/camera0.npz');pred,gt=d['prediction'].astype(np.float64),d['reference'].astype(np.float64)
 reported=json.loads((REP/'pacnerf-prefix-results.json').read_text())['rows']
 rows=[]
 for i in range(14):
  err=(pred[i]-gt[i])**2;mask=gt[i].min(-1)<.95
  psnr=-10*np.log10(err.mean());fg=-10*np.log10(err[mask].mean());ss=structural_similarity(pred[i],gt[i],data_range=1.,channel_axis=-1)
  assert abs(psnr-reported[i]['psnr_db'])<1e-3 and abs(fg-reported[i]['foreground_psnr_db'])<1e-3,i
  rows.append(dict(frame=i,psnr_db=psnr,foreground_psnr_db=fg,ssim=float(ss)))
 out['checks'].append('PAC-NeRF per-frame full/foreground PSNR reproduce reported values within 1e-3 dB (14 frames)')
 base=json.loads((REP/'pacnerf-baseline-results.json').read_text())
 for r in base['rows']:
  img=pred[r['frame']] if r['method']=='native_forecast' else pred[7]
  assert abs(-10*np.log10(((img-gt[r['frame']])**2).mean())-r['psnr_db'])<1e-3
 out['checks'].append('PAC-NeRF frozen-frame7 baseline PSNR reproduced (12 rows)')
 split=lambda rs,k,a,b:float(np.mean([r[k] for r in rs if a<=r['frame']<=b]))
 out['pacnerf_image']=dict(rows=rows,**{f'{s}_{k}':split(rows,k,a,b) for s,(a,b) in dict(heldview_fit_frames_0_7=(0,7),heldview_future_8_13=(8,13)).items() for k in ['psnr_db','foreground_psnr_db','ssim']})
 # 2. Ground-truth particle Chamfer
 GT=[ply(B/f'pacnerf-inputs/simulation_data/torus/{i}.ply') for i in range(14)];T=[cKDTree(g) for g in GT]
 out['gt_particles']=dict(count=len(GT[0]),sha256_frame0=sha(B/'pacnerf-inputs/simulation_data/torus/0.ply'))
 traj={}
 traj['PAC-NeRF native forecast (scored, native dt)']=[ply(B/f'pacnerf-work/checkpoint/torus-prefix-native/simulation/{i}.ply') for i in range(14)]
 traj['PAC-NeRF replay, fitted params, dt/4']=list(np.load(B/'pacnerf-counterfactual/original-fitted/predictions.npz')['xyz'].astype(np.float64))
 traj['PAC-NeRF replay, paper E/nu, dt/4']=list(np.load(B/'pacnerf-counterfactual/original-paper_parameters/predictions.npz')['xyz'].astype(np.float64))
 g=B/'gic-output/torus/forecast14.npz'
 if g.exists():traj['GIC native forecast']=list(np.load(g)['xyz'].astype(np.float64))
 geo={}
 for name,xs in traj.items():
  assert len(xs)==14 and all(np.isfinite(x).all() for x in xs),name
  tr=[cKDTree(x) for x in xs];c=[chamfer(xs[i],tr[i],GT[i],T[i])*1000 for i in range(14)]
  frozen=[chamfer(xs[7],tr[7],GT[i],T[i])*1000 for i in range(8,14)]
  geo[name]=dict(chamfer_mm_per_frame=c,fit_frames_0_7_mean_mm=float(np.mean(c[:8])),future_8_13_mean_mm=float(np.mean(c[8:])),final_frame13_mm=c[13],
   frozen_frame7_future_mean_mm=float(np.mean(frozen)),particles=len(xs[0]),centroid_err_frame13_mm=float(np.linalg.norm(xs[13].mean(0)-GT[13].mean(0))*1000))
 # Count-matched Chamfer: nearest-neighbour distance falls with particle density, so raw values
 # favour denser reconstructions. Subsample every trajectory (same indices across frames) to N.
 N=min(len(x[0]) for x in traj.values());rng=np.random.default_rng(0)
 for name,xs in traj.items():
  idx=rng.choice(len(xs[0]),N,replace=False);sub=[x[idx] for x in xs]
  c=[chamfer(sub[i],cKDTree(sub[i]),GT[i],T[i])*1000 for i in range(14)]
  geo[name].update(count_matched_N=N,count_matched_fit_0_7_mm=float(np.mean(c[:8])),count_matched_future_8_13_mm=float(np.mean(c[8:])),count_matched_final_frame13_mm=c[13])
 floor=[]
 for i in range(14):
  s=GT[i][rng.choice(len(GT[i]),N,replace=False)];floor.append(chamfer(s,cKDTree(s),GT[i],T[i])*1000)
 out['gt_subsample_floor_mm']=dict(N=N,mean=float(np.mean(floor)),note='Chamfer of a random N-point subsample of ground truth against full ground truth: sampling floor, not achievable error zero')
 out['geometry_chamfer']=geo
 out['geometry_definition']='Symmetric mean nearest-neighbour distance (mm) between predicted particles and released ground-truth simulation particles; not material correspondence. frozen_frame7 = hold predicted frame-7 particles fixed.'
 out['archive_sha256']={str(p.relative_to(R)):sha(p) for p in [B/'pacnerf-predictions/camera0.npz',B/'pacnerf-counterfactual/original-fitted/predictions.npz',B/'pacnerf-counterfactual/original-paper_parameters/predictions.npz']+([g] if g.exists() else [])}
 # 3. Paired camera-0 foreground image metric. Backgrounds differ (PAC white, GIC black) so full-frame
 # PSNR is not comparable; reference foreground pixels are verified identical across pipelines.
 renders={'PAC-NeRF native forecast':(pred,gt)}
 for name,f in [('GIC deformation network (no physics)','camera0-deformnet-render.npz'),('GIC native physics forecast','camera0-physics-render.npz')]:
  p=B/'gic-output/torus'/f
  if p.exists():
   z=np.load(p);assert list(z['frames'])==list(range(14));renders[name]=(z['prediction'].astype(np.float64),z['reference'].astype(np.float64));out['archive_sha256'][str(p.relative_to(R))]=sha(p)
 masks=[gt[i].min(-1)<.95 for i in range(14)]
 for name,(pp,rr) in renders.items():
  for i in range(14):assert np.abs(rr[i][masks[i]]-gt[i][masks[i]]).max()<1e-6,(name,i)
 out['checks'].append(f'camera-0 reference foreground pixels identical across {len(renders)} render archives')
 paired={}
 for name,(pp,rr) in renders.items():
  fg=[float(-10*np.log10(((pp[i]-rr[i])**2)[masks[i]].mean())) for i in range(14)]
  fz=[float(-10*np.log10(((pp[7]-rr[i])**2)[masks[i]].mean())) for i in range(8,14)]
  paired[name]=dict(foreground_psnr_db_per_frame=fg,fit_frames_0_7_mean_db=float(np.mean(fg[:8])),future_8_13_mean_db=float(np.mean(fg[8:])),frozen_frame7_future_mean_db=float(np.mean(fz)))
 out['paired_foreground_image']=dict(definition='Camera 0 (never fitted), mean per-frame PSNR over reference-foreground pixels min(RGB on white)<0.95; each method vs its own reference, references identical on these pixels',rows=paired)
 for k,v in paired.items():print(f"{k:45s} fg fit {v['fit_frames_0_7_mean_db']:.2f}  future {v['future_8_13_mean_db']:.2f}  frozen7 {v['frozen_frame7_future_mean_db']:.2f}")
 ge=REP/'gic-physics-evaluation.json'
 if ge.exists():out['gic_physics_evaluation_sha256']=sha(ge)
 (REP/'crossmethod-independent-audit.json').write_text(json.dumps(out,indent=2))
 for k,v in geo.items():print(f"{k:45s} fit {v['fit_frames_0_7_mean_mm']:.2f}  future {v['future_8_13_mean_mm']:.2f}  frozen7 {v['frozen_frame7_future_mean_mm']:.2f}  f13 {v['final_frame13_mm']:.2f}  n={v['particles']} | matched fit {v['count_matched_fit_0_7_mm']:.2f} future {v['count_matched_future_8_13_mm']:.2f} f13 {v['count_matched_final_frame13_mm']:.2f} centroid13 {v['centroid_err_frame13_mm']:.2f}")
 print('GT subsample floor',out['gt_subsample_floor_mm'])
 print({k:v for k,v in out['pacnerf_image'].items() if k!='rows'})
if __name__=='__main__':main()
