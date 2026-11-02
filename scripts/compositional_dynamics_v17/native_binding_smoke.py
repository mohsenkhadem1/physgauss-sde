"""Input-only unchanged native K16 binding translation check, no image targets."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['PYNPUT_BACKEND']='dummy'
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,AUTHOR,sha,write
sys.path.insert(0,str(AUTHOR))
import torch
from scripts.compositional_dynamics_v17.matched_gaussian_states import native_states

def main():
    grid=np.stack(np.meshgrid(*([np.linspace(-.02,.02,3)]*3),indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    displacement=np.array([.003,-.002,.001],np.float32);points=np.stack([grid,grid+displacement])
    centers=torch.tensor([[.003,.005,.001],[-.008,.004,-.002],[.009,-.01,.003]],device='cuda');dc=torch.zeros((3,3),device='cuda');q=torch.tensor([[1.,0,0,0]]*3,device='cuda');opacity=torch.ones((3,1),device='cuda')*.9
    out=native_states(points,centers,dc,q,opacity);error=float(np.max(abs(out['centers_m'][1]-centers.cpu().numpy()-displacement)))
    assert error<1e-6 and np.array_equal(out['centers_m'][0],centers.cpu().numpy())
    write(ROOT/'reports/compositional-dynamics-v17/native-gaussian-binding-smoke.json',dict(passed=True,translation_max_error_m=error,scope='Author K16 motion interpolation on artificial27particle cloud/3Gaussians; no private scene, fit or reference image',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),ROOT/'scripts/compositional_dynamics_v17/matched_gaussian_states.py',AUTHOR/'gs_render_dynamics.py',AUTHOR/'gaussian_splatting/dynamic_utils.py']}))
if __name__=='__main__':main()
