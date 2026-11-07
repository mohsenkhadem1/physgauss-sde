"""Identical-camera released-video reproduction metrics, no alignment fitting."""
from pathlib import Path
import sys,json
import cv2,numpy as np
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.gap_first_v01.drop_protocol import verify
BASE=ROOT/'runs/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop'
def load_video(p):
 cap=cv2.VideoCapture(str(p));frames=[]
 while True:
  ok,x=cap.read()
  if not ok:break
  frames.append(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))
 cap.release();return np.array(frames)
def score(gt,pred):
 assert gt.shape==pred.shape
 diff=(gt.astype(float)-pred.astype(float))**2;mask=np.max(gt,axis=-1)>8;union=mask|(np.max(pred,axis=-1)>8);ssim,smap=structural_similarity(gt,pred,channel_axis=-1,data_range=255,full=True)
 def metric(v):
  mse=float(v.mean());return dict(mse=mse,psnr_db=None if mse==0 else float(10*np.log10(255**2/mse)),exact=mse==0)
 return dict(full=metric(diff),foreground=metric(diff[mask]),union=metric(diff[union]),ssim=float(ssim),foreground_ssim=float(smap[mask].mean()),foreground_pixels=int(mask.sum()))
def average_finite(values):
 v=[x for x in values if x is not None];return float(np.mean(v)) if v else None
def main():
 protocol=verify();results=[]
 for asset in protocol['assets']:
  for case in ['native-predicted','native-groundtruth']:
   try:
    folder=BASE/'cases'/asset/case;status=json.loads((folder/'result.json').read_text());assert status['complete'] and status['frames']==30
    rgb=np.load(folder/'trajectory.npz')['rgb'];rows=[]
    for view in range(4):
     gt=load_video(BASE/'assets'/asset/'sample_00000'/f'view{view}.mp4');assert gt.shape==rgb[:,view].shape,(gt.shape,rgb[:,view].shape)
     for frame in range(30):rows.append(dict(view=view,frame=frame,**score(gt[frame],rgb[frame,view])))
    summaries=[]
    for split,ids in [('conditioning',range(20)),('withheld_recorded',range(20,30)),('frame0',range(1))]:
     rr=[r for r in rows if r['frame'] in ids]
     summaries.append(dict(split=split,n_frames_per_view=len(ids),n_views=4,full_psnr_mean_db=average_finite([r['full']['psnr_db'] for r in rr]),foreground_psnr_mean_db=average_finite([r['foreground']['psnr_db'] for r in rr]),union_psnr_mean_db=average_finite([r['union']['psnr_db'] for r in rr]),ssim=float(np.mean([r['ssim'] for r in rr])),foreground_ssim=float(np.mean([r['foreground_ssim'] for r in rr]))))
    d=dict(asset=asset,case=case,complete=True,rows=rows,summary=summaries,limits=['Released dataset training/test membership unknown; no generalization benchmark claim.','Given released Gaussian appearance; native video physical predictor and native mechanics/render replay, not new single-image appearance inference.','Native demo anchors sampled withseed123; releaseddataset anchor graph may differ. Groundtruth parameter replay is pipeline control, not learned baseline.','Native fourcamera ordering assumed fromreleasedrender helper; frame0 metric retained to expose mismatches. No per-frame alignment or camera optimization.','Foreground maxRGB>8 fixed onreference; unionmask also reported. MP4 compression retained.'])
   except Exception as e:d=dict(asset=asset,case=case,complete=False,error=repr(e),retained_in_cohort=True)
   (REPORT/f'{asset}-{case}-video-metrics.json').write_text(json.dumps(d,indent=2,allow_nan=False));results.append(d);print(asset,case,d.get('summary',d.get('error')),flush=True)
 (REPORT/'native-video-metrics-summary.json').write_text(json.dumps(dict(declared_cases=6,completed=sum(x['complete'] for x in results),summaries=[{k:v for k,v in x.items() if k!='rows'} for x in results]),indent=2))
if __name__=='__main__':main()
