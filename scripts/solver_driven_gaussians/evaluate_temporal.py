"""Calibration continuation of temporal adapters; no changed-action claims."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(2);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from types import SimpleNamespace
from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
from scene_PIDG import DeformModel
from gaussian_renderer import render
from utils.loss_utils import ssim
torch.set_num_threads(2);r=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v02';truth=np.load(base/'reference/calibration.npz');rows=[]
cfg={};exec((EXTERNAL/'upstream/arguments/pidg/rubber_dack.py').read_text(),cfg);pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
for method in ['pidg','hybrid']:
 model=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=True,reg_temporal_able=True);c=torch.load(base/('rgb-'+method)/'best.pt',map_location='cuda',weights_only=False);model.grid4d.load_state_dict(c['grid']);model.deform.load_state_dict(c['network'])
 with torch.no_grad():
  for frame in range(0,len(truth['times']),4):
   t=float(truth['times'][frame]);x=r.appearance.get_xyz;d=model.step(x/.1,torch.full((len(x),1),t/.42,device='cuda'));d['d_xyz']=torch.cat([d['d_xyz'][:,:4],d['d_xyz'][:,4:]*.1],-1)
   for view in range(4):
    target=r.image(truth['x'][frame],view);pred=render(r.cameras[view],r.appearance,pipe,r.background,d['d_xyz'],d['d_rotation'],d['d_scaling'],False)['render'].clamp(0,1);mask=(target-.96).abs().max(0).values>.03;union=mask|((pred-.96).abs().max(0).values>.03);err=(pred-target).square()
    rows.append(dict(method=method,time_s=t,view=view,forecast=t>.14,split='novel' if view>=2 else 'familiar',psnr=float(-10*err.mean().clamp_min(1e-12).log10()),foreground_psnr=float(-10*err[:,mask].mean().clamp_min(1e-12).log10()),union_psnr=float(-10*err[:,union].mean().clamp_min(1e-12).log10()),ssim=float(ssim(pred,target))))
summary=[]
for method in ['pidg','hybrid']:
 for forecast in [False,True]:
  for split in ['familiar','novel']:
   rr=[x for x in rows if x['method']==method and x['forecast']==forecast and x['split']==split];summary.append(dict(method=method,forecast=forecast,views=split,**{k:float(np.mean([x[k] for x in rr])) for k in ['psnr','foreground_psnr','union_psnr','ssim']}))
report=dict(summary=summary,frames=rows,scope='Same fixed appearance and observed calibration prefix; released PIDG temporal architecture/TV adaptation, not the complete PIDG recipe. Continuation of one calibration sequence only; no claims about intervention capabilities of PIDG.')
(ROOT/'reports/solver-driven-gaussians-v02/temporal-results.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
