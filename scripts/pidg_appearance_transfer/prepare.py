"""Preserve PIDG fitted appearance exactly at a fitting-frame binding reference."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from types import SimpleNamespace
from scene_PIDG import GaussianModel,DeformModel
from gaussian_renderer import render as native
from scripts.pidg_mechanics.appearance import camera_frame
from scripts.mechanics_gaussian_bridge.bridge import render
from utils.general_utils import quaternion_to_matrix
run=ROOT/'runs/pidg-comparison-v02/refined-pidg';out=ROOT/'runs/pidg-appearance-transfer-v01';cfg={};exec((run/'config.py').read_text(),cfg);g=GaussianModel(3);g.load_ply(str(run/'model/point_cloud/iteration_5000/point_cloud.ply'),og_number_points=100000);d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(run/'model'),5000)
doc=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_train.json').read_text());f=doc['frames'][0];cam,gt,mask,frame=camera_frame(f,doc)
with torch.no_grad():
 v=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),f['time'],device='cuda'));xyz=(quaternion_to_matrix(v['d_xyz'][:,:4])@g.get_xyz[...,None]).squeeze(-1)+v['d_xyz'][:,4:]
 q=g.get_rotation+v['d_rotation'];w,x,y,z=q.unbind(-1)
 # Match released rasterizer's raw (unnormalized) quaternion covariance exactly.
 Q=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
 scales=g.get_scaling+v['d_scaling'];cov=Q@torch.diag_embed(scales.square())@Q.transpose(-1,-2)
 state={'centers':xyz,'covariance':cov,'rotation':torch.eye(3,device='cuda').expand(len(xyz),3,3)}
 before=native(cam,g,SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False),torch.zeros(3,device='cuda'),v['d_xyz'],v['d_rotation'],v['d_scaling'],False)['render']
 after=render(cam,g,state,torch.zeros(3,device='cuda'))['render'];maxerr=float((before-after).abs().max());rmse=float((before-after).square().mean().sqrt());assert rmse<1e-5,(maxerr,rmse)
 torch.save({'xyz':xyz.cpu(),'covariance':cov.cpu(),'features':g.get_features.cpu(),'opacity':g.get_opacity.cpu(),'reference_frame':frame},out/'transferred-appearance.pt')
 (out/'binding-reference-audit.json').write_text(json.dumps({'passed':True,'reference_frame':f['file_path'],'reference_is_fitting_image':True,'gaussians':len(xyz),'maximum_rgb_difference':maxerr,'rgb_rmse':rmse,'original_psnr':float(-10*(before-gt).square().mean().log10()),'transferred_psnr':float(-10*(after-gt).square().mean().log10()),'source':'Matched local PIDG baseline; same fitting images; fixed fitted SH/opacity/shape','covariance_detail':'Explicitly matches upstream unnormalized quaternion convention rather than silently normalizing.'},indent=2))
 print((out/'binding-reference-audit.json').read_text())
