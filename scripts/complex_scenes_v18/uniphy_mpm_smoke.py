"""Bounded native author MPM forward/backward test, not a benchmark result."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.complex_scenes_v18.native_smoke import VENDOR,REPORT,load_uniphy_classes


def run():
    import taichi as ti
    ti.init(arch=ti.cuda,device_memory_GB=2,debug=False)
    sys.path.insert(0,str(VENDOR/'UniPhy/gradio_demo'))
    from mpmwrapper import MPMWrapperLearnableStress
    Stress,Fproj=load_uniphy_classes()
    folder=VENDOR/'UniPhy/ckpt/pretrained_stress_fproj_models_latent_space'
    state=torch.load(folder/'ckpt.pth',map_location='cpu',weights_only=True)
    bank=torch.load(folder/'traj_latent_weights_only.pt',map_location='cpu',weights_only=True)
    latent=nn.Embedding.from_pretrained(bank.mean(0,keepdim=True).cuda(),freeze=False)
    models=[]
    for Class,key in [(Stress,'stress'),(Fproj,'fproj')]:
        selected={k.replace('module.','').replace('_'+key,'').replace(key+'_model.',''):v for k,v in state.items()if key in k}
        model=Class(activation='gelu',hidden_size=selected['fc1.weight'].shape[0],device='cuda',embed_dim=1024,trajectory_latent=None).cuda()
        model.load_state_dict(selected,strict=True)
        for p in model.parameters():p.requires_grad_(False)
        model.trajectory_latent=latent;models.append(model)
    x=np.stack(np.meshgrid(*[np.linspace(.4,.42,3)]*3,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    steps=10;dt=1/50000
    mat=dict(material_type='MPMSimulator.elasticity',rho=1000.,velocity=[0,0,0],mu=384615.4,lam=576923.2,yield_stress=1.,plastic_viscosity=0.,friction_alpha=0.,inv_dt=1/dt,inv_frame_dt=1/(dt*steps))
    wrapper=MPMWrapperLearnableStress({'object':dict(material=mat,geometry={})},dict(dtype='float32',gravity=[0,-9.8,0],BC={}),dict(num_frames=1),x,np.repeat(x[:,None,:],33,axis=1),None,256,32,models[0],None,None,1024,models[1])
    wrapper.initialize_particles();wrapper.simulator_variables_initialize()
    wrapper.update_traj_latent_learnable(latent.weight[0].detach())
    sim=wrapper.simulator
    for i in range(steps):
        sim.advance_F_stress(i,0)
        if not sim.cfl_satisfy[None]:raise RuntimeError('Native CFL guard failed at '+str(i))
    positions=sim.x.to_numpy()[:len(x),:steps+1]
    assert np.isfinite(positions).all()
    wrapper.compute_loss();sim.loss.grad[None]=1.;wrapper.compute_loss.grad()
    sim.advance_grad_F_stress(0,0)
    grad=latent.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    return dict(passed=True,particles=len(x),steps=steps,dt_s=dt,loss=float(sim.loss[None]),maximum_displacement_m=float(np.linalg.norm(positions[:,-1]-x,axis=-1).max()),latent_gradient_norm=float(grad.norm()),cfl_passed=bool(sim.cfl_satisfy[None]),kind='Unmodified author MPM and pretrained constitutive forward/backward; mean latent; no fitting')


if __name__=='__main__':
    torch.set_num_threads(1);t=time.monotonic()
    out=REPORT/'uniphy-mpm-smoke.json'
    if out.exists():raise RuntimeError('Preserve previous report before retry')
    try:r=run()
    except Exception as e:r=dict(passed=False,error=str(e),traceback=traceback.format_exc())
    r.update(elapsed_s=time.monotonic()-t,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['passed']else 1)
