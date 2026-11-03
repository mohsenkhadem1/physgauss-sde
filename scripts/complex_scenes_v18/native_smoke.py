"""Exercise pinned author physics modules before writing benchmark adapters.

Spring-Gaus: original class, exact dense Torch KNN compatibility function in
place of unavailable PyTorch3D KNN. No spring, damping, boundary or step changes.
UniPhy: AST-load the two unchanged author demo NN class definitions, avoiding
interactive Gradio/Houdini imports. This tests constitutive weights, not MPM.
"""
import argparse,ast,hashlib,json,sys,types,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'runs/complex-scenes-v18/vendor'
REPORT=ROOT/'reports/complex-scenes-v18'


def exact_knn(p1,p2,lengths1=None,lengths2=None,K=1,version=-1,return_nn=False,return_sorted=True):
    if lengths1 is not None or lengths2 is not None or not return_sorted:raise ValueError('Unsupported KNN compatibility option')
    d=(p1[:,:,None,:]-p2[:,None,:,:]).square().sum(-1)
    dist,idx=d.topk(K,dim=-1,largest=False,sorted=True)
    near=p2[torch.arange(len(p2),device=p2.device)[:,None,None],idx]if return_nn else None
    return dist,idx,near


def load_spring():
    if 'pytorch3d.ops' not in sys.modules:
        package=types.ModuleType('pytorch3d');ops=types.ModuleType('pytorch3d.ops');ops.knn_points=exact_knn
        package.ops=ops;sys.modules['pytorch3d']=package;sys.modules['pytorch3d.ops']=ops
    sys.path.insert(0,str(VENDOR/'Spring-Gaus'))
    from lib.models.spring_mass.Spring_Mass import Spring_Mass
    return Spring_Mass


def load_uniphy_classes():
    from einops.layers.torch import Rearrange
    p=VENDOR/'UniPhy/gradio_demo/infer_material_latent.py';tree=ast.parse(p.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name in ('FprojNN','StressNN')]
    assert len(selected)==2
    namespace={'torch':torch,'nn':nn,'Rearrange':Rearrange,'__name__':'author_uniphy_constitutive'}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),namespace)
    return namespace['StressNN'],namespace['FprojNN']


def uniphy_smoke():
    Stress,Fproj=load_uniphy_classes();folder=VENDOR/'UniPhy/ckpt/pretrained_stress_fproj_models_latent_space'
    state=torch.load(folder/'ckpt.pth',map_location='cpu',weights_only=True)
    # Tensor-only conversion made with modern Torch's restricted allowlist loader.
    bank=torch.load(folder/'traj_latent_weights_only.pt',map_location='cpu',weights_only=True)
    latent=nn.Embedding.from_pretrained(bank.mean(0,keepdim=True).to('cuda'),freeze=False)
    models=[]
    for Class,key in [(Stress,'stress'),(Fproj,'fproj')]:
        selected={k.replace('module.','').replace('_'+key,'').replace(key+'_model.',''):v for k,v in state.items()if key in k}
        hidden=selected['fc1.weight'].shape[0]
        # Author constructors use different positional parameter ordering.
        model=Class(activation='gelu',hidden_size=hidden,device='cuda',embed_dim=bank.shape[1],trajectory_latent=None).cuda()
        model.load_state_dict(selected,strict=True)
        for p in model.parameters():p.requires_grad_(False)
        model.trajectory_latent=latent;models.append(model)
    F=torch.eye(3,device='cuda')[None].repeat(8,1,1);F[:,0,0]=torch.linspace(.8,1.2,8,device='cuda');F[:,0,1]=.1
    C=torch.zeros_like(F);U,S,V=torch.svd(F)
    stress=models[0](F,C,0);proj=models[1](F,U,V,0)
    loss=stress.square().mean()+proj.square().mean();loss.backward()
    assert torch.isfinite(stress).all()and torch.isfinite(proj).all()and torch.isfinite(latent.weight.grad).all()
    return dict(kind='Author pretrained constitutive smoke, not native MPM benchmark',stress_shape=list(stress.shape),projected_F_shape=list(proj.shape),latent_bank_shape=list(bank.shape),latent_gradient_norm=float(latent.weight.grad.norm()),gpu_peak_bytes=torch.cuda.max_memory_allocated(),passed=True)


def spring_smoke():
    from yacs.config import CfgNode as CN
    Class=load_spring();cfg=CN(dict(K_NEIGHBORS=12,K_BINDING=8,N_STEP=2,PRETRAINED=None,INIT_VELOCITY=[0,0,0],G=[0,0,-9.8],FITTING=True,FIX_MASS=True,FIX_K=True,SINGLE_K=True,DAMPING=True,FIX_DAMP=False,SOFT_K=False,SPRING_BC=False,UNLINEAR_FORCE=True,DATA=dict(DT=.0001,BC=[[[0,0,-1],[0,0,1]]],GLOBAL_K=.05,GLOBAL_M=.27/125,GLOBAL_DAMP=.00001)))
    torch.manual_seed(2026);xyz=torch.rand((125,3),device='cuda')*.06+torch.tensor([0,0,.1],device='cuda')
    model=Class(cfg,xyz,init_velocity=[0,0,0]).cuda();v=torch.zeros_like(xyz)
    K=10**model.global_k.expand_as(model.origin_len)/(model.origin_len+model.eps)
    damp=10**model.damp/(model.origin_len+model.eps);mass=10**model.global_m
    x0=xyz.clone();xyz=xyz.clone();xyz[:,0]*=1.01
    for _ in range(10):xyz,v=model.step(xyz,v,K,mass,torch.sigmoid(model.rebound_k),torch.sigmoid(model.fric_k),damp,1e-5)
    loss=(xyz-x0).square().sum();loss.backward()
    assert torch.isfinite(xyz).all()and torch.isfinite(model.global_k.grad).all()
    return dict(kind='Author spring force and integrator with exact Torch KNN compatibility, not fitted benchmark',nodes=125,steps=10,parameter_gradient=float(model.global_k.grad),passed=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('method',choices=['uniphy','spring']);a=ap.parse_args();torch.set_num_threads(1);t=time.monotonic()
    try:r=(uniphy_smoke if a.method=='uniphy'else spring_smoke)()
    except Exception as e:r=dict(passed=False,error=f'{type(e).__name__}: {e}')
    r.update(seconds=time.monotonic()-t,torch_version=torch.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/(a.method+'-native-smoke.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if not r['passed']:raise SystemExit(1)
