"""Evaluate frozen GIC appearance checkpoint on held camera, separately by time.
Not the physical forecast: that stage is scored separately once identified.
"""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from argparse import Namespace,ArgumentParser
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from scripts.gap_crossmethod_v02 import gic_native_entry
from scene import Scene,GaussianModel,DeformModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from PIL import Image

def main():
 parser=ArgumentParser();parser.add_argument('--iteration',type=int,default=40000);args=parser.parse_args()
 model=R/'runs/gap-crossmethod-v02/gic-output/torus';out=R/'output/gap-crossmethod-v02/gic-appearance';out.mkdir(parents=True,exist_ok=True)
 cfg=eval((model/'cfg_args').read_text(),{'Namespace':Namespace})
 gaussians=GaussianModel(cfg.sh_degree);scene=Scene(cfg,gaussians,load_iteration=args.iteration,shuffle=False)
 deform=DeformModel(cfg);deform.load_weights(str(model),args.iteration)
 pipe=Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
 bg=torch.zeros(3,device='cuda');records=[];imgs={}
 with torch.no_grad():
  for view in sorted(scene.getTestCameras(),key=lambda x:float(x.fid)):
   view.load2device();d=deform.step(gaussians.get_xyz,view.fid.reshape(1,1))
   pred=render(view,gaussians,pipe,bg,*d,False)['render'].clamp(0,1);gt=view.original_image
   mse=(pred-gt).square().mean();psnr=float(-10*torch.log10(mse));ss=float(ssim(pred,gt));frame=round(float(view.fid)*24)
   frozen_d=deform.step(gaussians.get_xyz,torch.tensor([[7/24]],device='cuda'))
   frozen=render(view,gaussians,pipe,bg,*frozen_d,False)['render'].clamp(0,1)
   records.append(dict(frame=frame,time=float(view.fid),split='held-view' if frame<8 else 'held-view-and-future',psnr_db=psnr,ssim=ss,frozen_model_frame7_psnr_db=float(-10*torch.log10((frozen-gt).square().mean())),frozen_model_frame7_ssim=float(ssim(frozen,gt))))
   imgs[frame]=(pred.permute(1,2,0).cpu().numpy(),gt.permute(1,2,0).cpu().numpy())
   pair=torch.cat([gt,pred,(pred-gt).abs()],dim=2).permute(1,2,0).cpu().numpy()
   Image.fromarray((pair*255).round().astype('uint8')).save(out/f'frame{frame:02d}.png');view.load2device('cpu')
 fr=sorted(imgs);np.savez_compressed(model/'camera0-deformnet-render.npz',frames=np.array(fr),prediction=np.stack([imgs[f][0] for f in fr]),reference=np.stack([imgs[f][1] for f in fr]))
 report=dict(method='Native GIC deformation-network appearance stage, NOT mechanics forecast',iteration=args.iteration,records=records,aggregates={s:{k:float(np.mean([r[k] for r in records if r['split']==s])) for k in ['psnr_db','ssim','frozen_model_frame7_psnr_db','frozen_model_frame7_ssim']} for s in ['held-view','held-view-and-future']},checkpoint_sha256=hashlib.sha256((model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply').read_bytes()).hexdigest(),scope='Black-background segmented full-frame RGB; no mask-only metric; no target frame used in training')
 (R/f'reports/gap-crossmethod-v02/gic-appearance-{args.iteration}.json').write_text(json.dumps(report,indent=2));print(report['aggregates'])
if __name__=='__main__':main()
