import os
"""Gaussian appearance bound to the paper's solved tetrahedral mechanics states."""
from pathlib import Path
import json,numpy as np,torch
from types import SimpleNamespace
from scripts.pidg_mechanics.fit_mechanics import payload
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.mechanics_gaussian_bridge.tetra import TetraBinding
from scripts.mechanics_gaussian_bridge.bridge import render
ROOT=Path(__file__).resolve().parents[2]

class Appearance(torch.nn.Module):
 active_sh_degree=1;max_sh_degree=1
 def __init__(self,cache=True):
  super().__init__();self.active_sh_degree=self.max_sh_degree=int(os.environ.get('PIDG_SH_DEGREE','1'));base=ROOT/os.environ.get('PIDG_MECHANICS_ROOT','runs/pidg-mechanics-v01');g=np.load(base/'training-geometry.npz');states=np.load(base/'mechanical-states.npz');fit=json.loads((base/'mechanics-fit.json').read_text());count=768;i=np.arange(count);z=1-2*(i+.5)/count;phi=i*np.pi*(3-np.sqrt(5));q=np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z];self.bindings=[];self.models=[];colors=[];normals=[];bodyids=[];self.cache=[];self.reference_radials=[]
  positions=[];covariances=[];rotations=[];offsets=[]
  for b in range(6):
   data=payload(g,b,states['times']);m=NeuralFEM(data,device='cuda:0',materials=1);self.models.append(m);radius=g['radii'][b];centers=g['centers'][b]+q*radius*.90
   cov=radius**2*(.105**2*(np.eye(3)[None]-q[:,:,None]*q[:,None,:])+.018**2*q[:,:,None]*q[:,None,:]);binding=TetraBinding(m,centers,cov);self.bindings.append(binding);self.reference_radials.append(m.t(q*radius))
   parts=[]
   with torch.no_grad():
    for nodes in states['positions'][:,b]:parts.append({k:v.float().detach() for k,v in binding.transport(m.t(nodes)).items()})
   positions.append(torch.stack([p['centers'] for p in parts]));covariances.append(torch.stack([p['covariance'] for p in parts]));rotations.append(torch.stack([p['rotation'] for p in parts]));colors.append(np.tile(g['colors'][b],(count,1)));bodyids.extend([b]*count)
   # Small bounded canonical radial appearance offset; mechanics is unchanged.
   F=torch.einsum('tnai,naJ->tniJ',torch.tensor(states['positions'][:,b],device='cuda',dtype=torch.double)[:,binding.ids],binding.grad)
   offsets.append(torch.einsum('tnij,nj->tni',F.float(),torch.tensor(q*radius,device='cuda',dtype=torch.float32)))
  self.register_buffer('node_states',torch.tensor(states['positions'],device='cuda',dtype=torch.float64));self.base_parameters=[m.t(row['parameters']) for m,row in zip(self.models,fit)]
  self.register_buffer('positions',torch.cat(positions,1));self.register_buffer('covariances',torch.cat(covariances,1));self.register_buffer('rotations',torch.cat(rotations,1));self.register_buffer('offsets',torch.cat(offsets,1));self.register_buffer('bodyids',torch.tensor(bodyids,device='cuda'))
  dc=(torch.tensor(np.concatenate(colors),device='cuda',dtype=torch.float32)-.5)/.28209479177387814;self.features=torch.nn.Parameter(torch.cat([dc[:,None],torch.zeros(len(dc),(self.max_sh_degree+1)**2-1,3,device='cuda')],1));self.radius_raw=torch.nn.Parameter(torch.zeros(6,device='cuda'));self.scale_raw=torch.nn.Parameter(torch.zeros(6,device='cuda'));self.opacity_raw=torch.nn.Parameter(torch.full((len(dc),1),1.,device='cuda'))
 @property
 def get_features(self):return self.features
 @property
 def get_opacity(self):return .65+.34*self.opacity_raw.sigmoid()
 def state(self,frame):
  radius=.25*self.radius_raw.tanh()[self.bodyids];scale=torch.exp(.3*self.scale_raw.tanh()[self.bodyids])
  return dict(centers=self.positions[frame]+radius[:,None]*self.offsets[frame],covariance=self.covariances[frame]*scale[:,None,None]**2,rotation=self.rotations[frame])
 def image(self,camera,frame):return render(camera,self,self.state(frame),torch.zeros(3,device='cuda'))['render']

def camera_frame(f,doc,width=320,angle=0):
 from PIL import Image
 from scene_PIDG.cameras import Camera
 from utils.graphics_utils import focal2fov,fov2focal
 base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction';path=base/'released'/(f['file_path']+'.png');im=Image.open(path).convert('RGBA');height=round(im.height*width/im.width);a=np.asarray(im.resize((width,height),Image.Resampling.LANCZOS)).copy()/255
 gt=torch.tensor(np.floor(a[:,:,:3]*a[:,:,3:4]*255)/255,device='cuda',dtype=torch.float32).permute(2,0,1);mask=torch.tensor(a[:,:,3]>.5,device='cuda')
 pose=np.array(f['transform_matrix']);rad=np.deg2rad(angle);rot=np.array([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]);pose[:3,:3]=rot@pose[:3,:3];pose[:3,3]=rot@pose[:3,3];matrix=np.linalg.inv(pose);R=-matrix[:3,:3].T;R[:,0]=-R[:,0];T=-matrix[:3,3];fx=doc['camera_angle_x'];fy=focal2fov(fov2focal(fx,width),height);idx=int(Path(f['file_path']).name)-1
 return Camera(f['file_path'],R,T,fx,fy,gt,None,str(idx),0,fid=f['time']),gt,mask,idx
