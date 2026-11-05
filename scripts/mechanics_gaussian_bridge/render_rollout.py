"""Render sealed paper mechanics states through the official PIDG rasterizer."""
import argparse,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical-output',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import numpy as np,torch
 from PIL import Image
 from types import SimpleNamespace
 from scene_PIDG import GaussianModel
 from scene_PIDG.cameras import Camera
 from utils.graphics_utils import focal2fov
 from utils.general_utils import quaternion_to_matrix
 from scripts.mechanics_gaussian_bridge.bridge import Q2Binding,render
 from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
 from scripts.physical_refinement_v15.observations import Camera as PhysicalCamera
 folder=Path(a.physical_output);seal=json.loads((folder/'complete.json').read_text());assert seal['completed'] and seal['law_type']=='learned'
 for name,key in [('geometry.npz','geometry_sha256'),('nodal-state.npz','nodal_state_sha256')]:assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==seal[key]
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'report.json').exists():raise ValueError('Preserve completed rendering')
 checkpoint=ROOT/'runs/compositional-dynamics-v17/matched-appearance-native-resolution/model/point_cloud/iteration_10000/point_cloud.ply'
 from plyfile import PlyData
 ply=PlyData.read(checkpoint).elements[0]
 def columns(names):return torch.tensor(np.stack([ply[n] for n in names],-1),device='cuda',dtype=torch.float32)
 def numbered(prefix):return sorted([n for n in ply.data.dtype.names if n.startswith(prefix)],key=lambda n:int(n.rsplit('_',1)[1]))
 dc=columns(numbered('f_dc_')).reshape(-1,1,3);rest=columns(numbered('f_rest_')).reshape(-1,3,15).transpose(1,2)
 pc=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,get_xyz=columns(['x','y','z']),get_scaling=columns(numbered('scale_')).exp(),get_rotation=torch.nn.functional.normalize(columns(numbered('rot_')),dim=-1),get_features=torch.cat([dc,rest],1),get_opacity=columns(['opacity']).sigmoid())
 # Some author appearance checkpoints store isotropic scale in one component.
 scale=pc.get_scaling
 if scale.shape[-1]==1:scale=scale.expand(-1,3)
 rot=quaternion_to_matrix(pc.get_rotation);cov=rot@torch.diag_embed(scale.square())@rot.transpose(-1,-2)
 geo=np.load(folder/'geometry.npz');states=np.load(folder/'nodal-state.npz');bindings=[]
 for body in range(states['positions_m'].shape[1]):
  elems=geo[f'elements_{body}'];n=round((len(elems)/4)**(1/3));assert 4*n**3==len(elems)
  model=SimpleNamespace(geometry=GeometrySpec(dimensions=(.12,.06,.06)),rest=geo[f'rest_{body}'],elements=elems,cells=(4*n,n,n),axes=(np.linspace(0,.24,4*n+1),np.linspace(-.04,.04,n+1),np.linspace(-.04,.04,n+1)))
  bindings.append(Q2Binding(model,pc.get_xyz.detach().cpu().numpy(),cov.detach().cpu().numpy(),device='cuda',dtype=torch.float32))
 count=len(bindings)
 class Appearance:
  active_sh_degree=pc.active_sh_degree
  max_sh_degree=pc.max_sh_degree
  get_features=pc.get_features.repeat(count,1,1)
  get_opacity=pc.get_opacity.repeat(count,1)
 appearance=Appearance();records=[];frames=[0,len(states['times_s'])//2,len(states['times_s'])-1]
 eyes=[(.25,-.42,.25),(-.22,-.22,.15),(.22,.30,.20)];gradient_norm=None
 for frame in frames:
  nodes=torch.tensor(states['positions_m'][frame].real,device='cuda',dtype=torch.float32,requires_grad=True)
  parts=[b.transport(nodes[i]) for i,b in enumerate(bindings)];state={k:torch.cat([s[k] for s in parts]) for k in parts[0]}
  for j,eye in enumerate(eyes):
   cam=PhysicalCamera(eye,(.08,0.,.035),512,384,400.);origin,R=cam.pose()
   view=Camera('mechanics',R.T,-R@origin,focal2fov(400.,512),focal2fov(400.,384),torch.zeros(3,384,512,device='cuda'),None,'mechanics',0,fid=float(states['times_s'][frame]))
   package=render(view,appearance,state,torch.full((3,),245/255,device='cuda'));rgb=package['render'];assert torch.isfinite(rgb).all()
   if gradient_norm is None:
    gradient=torch.autograd.grad(rgb.square().mean(),nodes,retain_graph=True)[0];assert torch.isfinite(gradient).all() and gradient.norm()>0;gradient_norm=float(gradient.norm())
   path=out/f'frame{frame:03d}-view{j}.png';Image.fromarray((rgb.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(path)
   records.append({'frame':frame,'time_s':float(states['times_s'][frame]),'eye':eye,'image':path.name,'minimum_J':float(state['determinant'].min().detach())})
 report={'scope':'Actual saved learned-material Q2 rollout connected to PIDG rasterizer; no PIDG motion network. Connection validation, not Balls Reaction benchmark.','physical_output':str(folder),'physical_state_sha256':seal['nodal_state_sha256'],'appearance_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'render_to_mechanical_nodes_gradient_norm':gradient_norm,'gaussians':len(pc.get_xyz)*count,'records':records}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
