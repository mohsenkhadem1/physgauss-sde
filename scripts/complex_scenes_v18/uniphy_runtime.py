"""Load released UniPhy weights and construct the unmodified single-object MPM.

Caller owns Taichi initialization, physical input, observations and latent fitting.
No reference mechanics or private material descriptor is imported here.
"""
import sys
import numpy as np
import torch
from torch import nn
from scripts.complex_scenes_v18.native_smoke import VENDOR,load_uniphy_classes


def models():
    Stress,Fproj=load_uniphy_classes();folder=VENDOR/'UniPhy/ckpt/pretrained_stress_fproj_models_latent_space'
    state=torch.load(folder/'ckpt.pth',map_location='cpu',weights_only=True)
    bank=torch.load(folder/'traj_latent_weights_only.pt',map_location='cpu',weights_only=True)
    latent=nn.Embedding.from_pretrained(bank.mean(0,keepdim=True).cuda(),freeze=False);result=[]
    for Class,key in [(Stress,'stress'),(Fproj,'fproj')]:
        selected={k.replace('module.','').replace('_'+key,'').replace(key+'_model.',''):v for k,v in state.items()if key in k}
        model=Class(activation='gelu',hidden_size=selected['fc1.weight'].shape[0],device='cuda',embed_dim=1024,trajectory_latent=None).cuda()
        model.load_state_dict(selected,strict=True)
        for p in model.parameters():p.requires_grad_(False)
        model.trajectory_latent=latent;result.append(model)
    return result,latent,bank


def wrapper(x,target,models,*,dt,steps,chunk=64,density=1000.,gravity=(0,0,0),boundaries=None,forward_only=False):
    if steps>=chunk and not forward_only:raise ValueError('Loss adapter requires a single complete chunk; native forward chunking is available')
    sys.path.insert(0,str(VENDOR/'UniPhy/gradio_demo'))
    from mpmwrapper import MPMWrapperLearnableStress
    # Analytic mu/lam are unused by the learned F/stress stepping path.
    mat=dict(material_type='MPMSimulator.elasticity',rho=density,velocity=[0,0,0],mu=384615.4,lam=576923.2,yield_stress=1.,plastic_viscosity=0.,friction_alpha=0.,inv_dt=1/dt,inv_frame_dt=1/(dt*steps))
    capacity=256*((len(x)+255)//256)
    w=MPMWrapperLearnableStress({'object':dict(material=mat,geometry={})},dict(dtype='float32',gravity=list(gravity),BC=boundaries or {}),dict(num_frames=1),np.asarray(x,np.float32),np.asarray(target,np.float32),None,capacity,chunk,models[0],None,None,1024,models[1])
    w.initialize_particles();return w
