"""Shared dense appearance and calibrated cameras; no motion MLP in this renderer."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.solver_driven_gaussians.binding import DenseBinding
from scripts.solver_driven_gaussians.dynamics import DynamicFEM
from scripts.mechanics_gaussian_bridge.bridge import render

ROOT=Path(__file__).resolve().parents[2]
APPEARANCE=ROOT/'runs/compositional-dynamics-v17/matched-appearance-native-resolution/model/point_cloud/iteration_10000/point_cloud.ply'

def load_appearance():
    from plyfile import PlyData
    from utils.general_utils import quaternion_to_matrix
    p=PlyData.read(APPEARANCE).elements[0]
    def columns(names):return torch.tensor(np.stack([p[n] for n in names],-1),device='cuda',dtype=torch.float32)
    def fields(prefix):return sorted([n for n in p.data.dtype.names if n.startswith(prefix)],key=lambda n:int(n.rsplit('_',1)[1]))
    opacity=columns(['opacity']).sigmoid();keep=opacity[:,0]>.1
    xyz=columns(['x','y','z'])[keep];scales=columns(fields('scale_')).exp()[keep]
    if scales.shape[1]==1:scales=scales.expand(-1,3)
    rotation=torch.nn.functional.normalize(columns(fields('rot_'))[keep],dim=-1)
    R=quaternion_to_matrix(rotation);cov=R@torch.diag_embed(scales.square())@R.transpose(-1,-2)
    dc=columns(fields('f_dc_')).reshape(-1,1,3);rest=columns(fields('f_rest_')).reshape(-1,3,15).transpose(1,2)
    appearance=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,get_xyz=xyz,get_features=torch.cat([dc,rest],1)[keep],get_opacity=opacity[keep],get_scaling=scales,get_rotation=rotation)
    appearance.get_covariance=lambda scaling_modifier=1.:torch.stack([cov[:,0,0],cov[:,0,1],cov[:,0,2],cov[:,1,1],cov[:,1,2],cov[:,2,2]],-1)*scaling_modifier**2
    return appearance,cov

def cameras(width=320,height=240):
    from scripts.physical_refinement_v15.observations import Camera as PhysicalCamera
    from scene_PIDG.cameras import Camera
    from utils.graphics_utils import focal2fov
    result=[]
    for i,eye in enumerate([(.24,-.35,.20),(-.19,-.24,.12),(.20,.30,.17),(-.12,.25,.22)]):
        p=PhysicalCamera(eye,(.06,0.,-.035),width,height,width*1.0);origin,R=p.pose()
        cam=Camera('controlled',R.T,-R@origin,focal2fov(width,width),focal2fov(width,height),torch.zeros(3,height,width,device='cuda'),None,str(i),i,fid=0.)
        cam.source_eye=eye
        result.append(cam)
    return result

class MechanicalRenderer:
    def __init__(self,cells=(8,4,4),width=320):
        self.model=DynamicFEM(cells=cells,device='cuda',dtype=torch.float32)
        self.appearance,self.cov=load_appearance()
        self.binding=DenseBinding(self.model,self.appearance.get_xyz.cpu().numpy(),self.cov.cpu().numpy())
        self.cameras=cameras(width,round(width*.75));self.background=torch.full((3,),.96,device='cuda')
    def state(self,nodes):return self.binding.transport(self.model.t(nodes))
    def image(self,nodes,view=0):return render(self.cameras[view],self.appearance,self.state(nodes),self.background)['render'].clamp(0,1)
