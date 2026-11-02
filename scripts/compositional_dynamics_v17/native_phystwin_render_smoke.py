"""Tiny author-renderer execution probe; no appearance training or benchmark claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys, json, hashlib, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
AUTHOR=ROOT/'runs/compositional-dynamics-v17/PhysTwin-author'
sys.path.insert(0,str(AUTHOR))
import torch
import numpy as np
from PIL import Image
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic()
    out=ROOT/'runs/compositional-dynamics-v17/native-render-smoke';out.mkdir(parents=True,exist_ok=True)
    pc=GaussianModel(0)
    pc._xyz=torch.tensor([[-.08,0,1],[.08,0,1],[0,.08,1]],device='cuda',requires_grad=True)
    pc._scaling=torch.full((3,3),float(np.log(.045)),device='cuda')
    pc._rotation=torch.tensor([[1.,0,0,0]]*3,device='cuda')
    pc._opacity=torch.full((3,1),3.,device='cuda')
    camera=SimpleNamespace(K=torch.tensor([[100.,0,64],[0,100,48],[0,0,1]],device='cuda'),image_width=128,image_height=96,world_view_transform=torch.eye(4,device='cuda'))
    colors=torch.eye(3,device='cuda')
    result=render(camera,pc,SimpleNamespace(),torch.ones(3,device='cuda'),override_color=colors,use_gsplat=True)
    loss=result['render'][:3].square().mean();loss.backward()
    image=result['render'].detach().cpu().numpy();depth=result['depth'].detach().cpu().numpy()
    grad=pc._xyz.grad.detach().cpu().numpy()
    assert image.shape==(4,96,128) and np.isfinite(image).all() and np.isfinite(depth).all()
    assert np.isfinite(grad).all() and np.linalg.norm(grad)>0 and image[3].max()>.5
    np.savez_compressed(out/'render.npz',rgba=image,depth=depth,xyz_gradient=grad)
    Image.fromarray((image[:3].transpose(1,2,0).clip(0,1)*255).astype('uint8')).save(out/'render.png')
    report=dict(passed=True,scope='Unmodified author render() with synthetic Gaussians, forward and backward only; not full trained PhysTwin pipeline or held-view benchmark.',author_commit='81c718790a37e5e0102eb77af2c6edd34a9db25f',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__,wall_seconds=time.monotonic()-start,gradient_norm=float(np.linalg.norm(grad)),alpha_max=float(image[3].max()),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),AUTHOR/'gaussian_splatting/gaussian_renderer/__init__.py',AUTHOR/'gaussian_splatting/scene/gaussian_model.py']},artifacts={str(p.relative_to(ROOT)):sha(p) for p in [out/'render.npz',out/'render.png']})
    (ROOT/'reports/compositional-dynamics-v17/native-phystwin-render-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
