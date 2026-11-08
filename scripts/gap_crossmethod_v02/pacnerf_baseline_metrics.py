"""Predeclared frozen-last-prediction image baseline; no held-frame fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'runs/gap-crossmethod-v02';R=ROOT/'reports/gap-crossmethod-v02';P=R/'pacnerf-baseline-protocol.json';p=json.loads(P.read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['source_sha256']
assert json.loads((R/'pacnerf-prefix-status.json').read_text())['phase']=='complete'
archive=B/'pacnerf-predictions/camera0.npz';d=np.load(archive);pred,gt=d['prediction'],d['reference'];rows=[]
for frame in range(8,14):
 for method,image in [('native_forecast',pred[frame]),('frozen_predicted_frame7',pred[7])]:
  diff=(image-gt[frame])**2;mask=np.min(gt[frame],axis=-1)<.95;mse=float(diff.mean());fg=float(diff[mask].mean());ssim=float(structural_similarity(image,gt[frame],data_range=1.,channel_axis=-1))
  rows.append(dict(frame=frame,method=method,full_mse=mse,psnr_db=float(-10*np.log10(mse)),foreground_mse=fg,foreground_psnr_db=float(-10*np.log10(fg)),ssim=ssim))
summary={}
for method in ['native_forecast','frozen_predicted_frame7']:
 rr=[r for r in rows if r['method']==method];summary[method]={k:float(np.mean([r[k] for r in rr])) for k in ['psnr_db','foreground_psnr_db','ssim','full_mse','foreground_mse']}
(R/'pacnerf-baseline-results.json').write_text(json.dumps(dict(scope=p['scope'],rows=rows,summary=summary,protocol_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),prediction_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))
