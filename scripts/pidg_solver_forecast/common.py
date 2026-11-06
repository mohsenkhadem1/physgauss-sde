"""Shared prefix checkpoint/camera utilities. No hidden test-image reads."""
from pathlib import Path
import json
import numpy as np
import torch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-solver-forecast-v01';DATA=BASE/'data'
def load(scene,arm='native-base'):
 from scene_PIDG import GaussianModel,DeformModel
 run=BASE/scene/arm;meta=json.loads((run/'run.json').read_text());assert meta['status']=='training_complete';assert 'pidg-solver-forecast-v01' in meta['arguments']['data_root'];steps=meta['arguments']['steps'];cfg={};exec((run/'config.py').read_text(),cfg);g=GaussianModel(3);g.load_ply(str(run/'model/point_cloud'/f'iteration_{steps}'/'point_cloud.ply'),og_number_points=100000);d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(run/'model'),steps);return g,d

def camera(scene,f,width=320,load_image=True):
 from PIL import Image
 from scene_PIDG.cameras import Camera
 from utils.graphics_utils import focal2fov,fov2focal
 doc=json.loads((DATA/scene/'fit.json').read_text());raw=DATA/scene/'released';im=Image.open(raw/(f['file_path']+'.png')).convert('RGBA') if load_image else None
 height=round(im.height*width/im.width) if im else round(width*9/16)
 if im:
  arr=np.asarray(im.resize((width,height),Image.Resampling.LANCZOS)).copy()/255.;gt=torch.tensor(np.floor(arr[:,:,:3]*arr[:,:,3:4]*255)/255,device='cuda',dtype=torch.float32).permute(2,0,1);mask=torch.tensor(arr[:,:,3]>.5,device='cuda')
 else:gt=torch.zeros(3,height,width,device='cuda');mask=None
 matrix=np.linalg.inv(np.asarray(f['transform_matrix']));R=-matrix[:3,:3].T;R[:,0]=-R[:,0];T=-matrix[:3,3];fx=doc['camera_angle_x'];fy=focal2fov(fov2focal(fx,width),height);cam=Camera(f['file_path'],R,T,fx,fy,gt,None,Path(f['file_path']).name,0,fid=f['time']);return cam,gt,mask

def native(g,d,cam,t):
 from gaussian_renderer import render
 pk=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),t,device='cuda'));return render(cam,g,SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False),torch.zeros(3,device='cuda'),pk['d_xyz'],pk['d_rotation'],pk['d_scaling'],False)

def positions(g,d,t):
 from utils.general_utils import quaternion_to_matrix
 pk=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),t,device='cuda'));return (quaternion_to_matrix(pk['d_xyz'][:,:4])@g.get_xyz[...,None]).squeeze(-1)+pk['d_xyz'][:,4:]

def raw_rotation(q):
 # Exactly matches the upstream CUDA covariance convention (no quaternion normalization).
 r,x,y,z=q.unbind(-1)
 return torch.stack([1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def appearance_at(g,d,t):
 pk=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),t,device='cuda'));xyz=positions(g,d,t);rot=raw_rotation(g.get_rotation+pk['d_rotation']);scale=g.get_scaling+pk['d_scaling'];cov=rot@torch.diag_embed(scale.square())@rot.transpose(-1,-2)
 return dict(centers=xyz.detach(),covariance=cov.detach(),features=g.get_features.detach(),opacity=g.get_opacity.detach())
