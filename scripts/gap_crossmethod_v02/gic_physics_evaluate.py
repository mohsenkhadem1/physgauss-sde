"""Native fitted GIC mechanics forecast, then native fixed-particle appearance fit.
Only fitting-prefix cameras enter either fit. Targets scored after fitting.
"""
import sys,json,hashlib,time
from pathlib import Path
from argparse import Namespace,ArgumentParser
import numpy as np
import torch,taichi as ti,trimesh
from scipy.spatial import cKDTree
from PIL import Image
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from scripts.gap_crossmethod_v02 import gic_native_entry
from simulator import Simulator
from new_trajectory import load_pcd_file,gen_xyz_list
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from arguments import OptimizationParams
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.general_utils import safe_state

def main():
 safe_state(False);model=R/'runs/gap-crossmethod-v02/gic-output/torus';out=R/'output/gap-crossmethod-v02/gic-physics';out.mkdir(parents=True,exist_ok=True)
 cfg=eval((model/'cfg_args').read_text(),{'Namespace':Namespace});phys=Namespace(**json.loads((model/'0-pred.json').read_text()))
 pipe=Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
 ti.init(arch=ti.cuda,debug=False,fast_math=False,device_memory_fraction=.4)
 vol=load_pcd_file(str(model),40000);sim=Simulator(phys,vol)
 xyz=gen_xyz_list(sim,14,diff=False);assert sim.succeed() and all(torch.isfinite(x).all() for x in xyz)
 pred=np.stack([x.cpu().numpy() for x in xyz]);np.savez_compressed(model/'forecast14.npz',xyz=pred,initial=vol.cpu().numpy())
 # Author's released application retrains appearance with positions fixed to mechanics.
 parser=ArgumentParser();op=OptimizationParams(parser);opt=op.extract(parser.parse_args(['--reg_scale','--reg_alpha']))  # same flags as native train_dynamic run
 del sim
 scene=train_gs_with_fixed_pcd(vol,cfg,opt,pipe,[7000,40000],[7000,10000,20000,30000,40000],[x-vol for x in xyz[:8]],phys.fps,grid_size=phys.density_grid_size)
 bg=torch.zeros(3,device='cuda');records=[];imgs={}
 with torch.no_grad():
  for view in sorted(scene.getTestCameras(),key=lambda x:float(x.fid)):
   view.load2device();frame=round(float(view.fid)*24)
   image=render(view,scene.gaussians,pipe,bg,xyz[frame]-vol,0.,0.,False)['render'].clamp(0,1);gt=view.original_image
   row=dict(frame=frame,split='held-view' if frame<8 else 'held-view-and-future',psnr_db=float(-10*torch.log10((image-gt).square().mean())),ssim=float(ssim(image,gt)))
   frozen=render(view,scene.gaussians,pipe,bg,xyz[7]-vol,0.,0.,False)['render'].clamp(0,1)
   row.update(frozen_model_frame7_psnr_db=float(-10*torch.log10((frozen-gt).square().mean())),frozen_model_frame7_ssim=float(ssim(frozen,gt)))
   gtpath=R/f'runs/gap-crossmethod-v02/pacnerf-inputs/simulation_data/torus/{frame}.ply'
   if gtpath.exists():
    target=np.asarray(trimesh.load(gtpath,process=False).vertices);a=cKDTree(target).query(pred[frame])[0];b=cKDTree(pred[frame]).query(target)[0]
    row.update(chamfer_mean_distance_m=float((a.mean()+b.mean())/2),chamfer_mean_squared_m2=float((np.square(a).mean()+np.square(b).mean())/2))
    fa=cKDTree(target).query(pred[7])[0];fb=cKDTree(pred[7]).query(target)[0]
    row.update(frozen_model_frame7_chamfer_mean_distance_m=float((fa.mean()+fb.mean())/2))
   imgs[frame]=(image.permute(1,2,0).cpu().numpy(),gt.permute(1,2,0).cpu().numpy(),view.gt_alpha_mask[0].cpu().numpy())
   records.append(row);pair=torch.cat([gt,image,(image-gt).abs()],2).permute(1,2,0).cpu().numpy();Image.fromarray(np.uint8(np.clip(pair*255,0,255))).save(out/f'frame{frame:02d}.png');view.load2device('cpu')
 fr=sorted(imgs);np.savez_compressed(model/'camera0-physics-render.npz',frames=np.array(fr),prediction=np.stack([imgs[f][0] for f in fr]),reference=np.stack([imgs[f][1] for f in fr]),alpha=np.stack([imgs[f][2] for f in fr]))
 report=dict(method='Native GIC fitted MPM forecast and native fixed-particle Gaussian appearance fit',records=records,parameters=vars(phys),fitting_frames=list(range(8)),fitting_cameras=list(range(1,11)),test_camera=0,appearance_iterations=40000,scope='Native known-density/gravity/collider assumptions; RGB black-background fullframe; geometry symmetric all-particle nearest-distance Chamfer, not established material correspondences',aggregates={s:{k:float(np.mean([r[k] for r in records if r['split']==s])) for k in records[0] if k not in ['frame','split']} for s in ['held-view','held-view-and-future']})
 (R/'reports/gap-crossmethod-v02/gic-physics-evaluation.json').write_text(json.dumps(report,indent=2));print(report['aggregates'])
if __name__=='__main__':main()
